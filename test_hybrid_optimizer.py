import json
import ijson
import statistics

from pathlib import Path

from src.optimization.travel_time_loader import (
    load_route_travel_times,
)

from src.optimization.travel_time_optimizer import (
    calculate_route_travel_time,
    nearest_neighbor_by_travel_time,
    two_opt_by_travel_time,
)


ROOT = Path(__file__).resolve().parent

BASE = (
    ROOT
    / "dataset"
    / "almrrc2021-data-training"
    / "model_build_inputs"
)

ROUTE_FILE = BASE / "route_data.json"
SEQUENCE_FILE = BASE / "actual_sequences.json"
TRAVEL_FILE = BASE / "travel_times.json"


SAMPLE_SIZE = 20
MIN_STOPS = 50


with open(ROUTE_FILE) as f:
    routes = json.load(f)

with open(SEQUENCE_FILE) as f:
    sequences = json.load(f)


# --------------------------------------------------
# Use the 20 WORST routes from the existing
# 100-route benchmark.
# --------------------------------------------------

import pandas as pd

benchmark = pd.read_csv(
    "benchmark_results_100.csv"
)

worst_routes = (
    benchmark
    .sort_values(
        "actual_improvement"
    )
    .head(SAMPLE_SIZE)
    ["route_id"]
    .tolist()
)


print("===== HYBRID OPTIMIZER TEST =====")
print(
    "Testing worst",
    len(worst_routes),
    "routes from benchmark."
)
print()


wanted = set(worst_routes)

travel_matrices = {}


print("Streaming travel-time file...")


with open(TRAVEL_FILE, "rb") as f:

    for route_id, matrix in ijson.kvitems(f, ""):

        if route_id in wanted:

            travel_matrices[route_id] = matrix

            print(
                f"Loaded "
                f"{len(travel_matrices)}/"
                f"{len(wanted)}"
            )

        if len(travel_matrices) == len(wanted):
            break


print()
print("===== RESULTS =====")
print()


current_results = []
hybrid_results = []


for route_id in worst_routes:

    if route_id not in travel_matrices:
        continue

    sequence_data = sequences[
        route_id
    ]["actual"]

    travel_times = travel_matrices[
        route_id
    ]


    actual_route = [
        stop_id

        for stop_id, _ in sorted(
            sequence_data.items(),
            key=lambda item: item[1]
        )

        if stop_id in travel_times
    ]


    if len(actual_route) < 4:
        continue


    start = actual_route[0]


    actual_time = calculate_route_travel_time(
        actual_route,
        travel_times
    )


    # ------------------------------------------
    # Current method
    # ------------------------------------------

    nn_route = nearest_neighbor_by_travel_time(
        start,
        actual_route,
        travel_times
    )

    current_route = two_opt_by_travel_time(
        nn_route,
        travel_times
    )

    current_time = calculate_route_travel_time(
        current_route,
        travel_times
    )


    # ------------------------------------------
    # Hybrid method:
    #
    # 1. Refine actual route
    # 2. Refine NN route
    # 3. Choose best
    # ------------------------------------------

    actual_refined = two_opt_by_travel_time(
        actual_route,
        travel_times
    )

    actual_refined_time = (
        calculate_route_travel_time(
            actual_refined,
            travel_times
        )
    )


    hybrid_route = (
        actual_refined
        if actual_refined_time < current_time
        else current_route
    )


    hybrid_time = calculate_route_travel_time(
        hybrid_route,
        travel_times
    )


    current_improvement = (
        (
            actual_time
            - current_time
        )
        / actual_time
        * 100
    )


    hybrid_improvement = (
        (
            actual_time
            - hybrid_time
        )
        / actual_time
        * 100
    )


    actual_refined_improvement = (
        (
            actual_time
            - actual_refined_time
        )
        / actual_time
        * 100
    )


    current_results.append(
        current_improvement
    )

    hybrid_results.append(
        hybrid_improvement
    )


    print(
        f"{route_id} | "
        f"{len(actual_route)} stops | "
        f"Current: {current_improvement:.2f}% | "
        f"Actual-refined: "
        f"{actual_refined_improvement:.2f}% | "
        f"Hybrid: {hybrid_improvement:.2f}%"
    )


# --------------------------------------------------
# Summary
# --------------------------------------------------

print()
print("==============================================")
print("HYBRID SUMMARY")
print("==============================================")


if hybrid_results:

    print(
        "Routes evaluated:",
        len(hybrid_results)
    )

    print(
        "Current average improvement:",
        f"{statistics.mean(current_results):.2f}%"
    )

    print(
        "Hybrid average improvement:",
        f"{statistics.mean(hybrid_results):.2f}%"
    )

    print(
        "Current worst:",
        f"{min(current_results):.2f}%"
    )

    print(
        "Hybrid worst:",
        f"{min(hybrid_results):.2f}%"
    )

    print(
        "Hybrid improved routes:",
        sum(
            value > 0
            for value in hybrid_results
        ),
        "/",
        len(hybrid_results)
    )

    print(
        "Hybrid worsened routes:",
        sum(
            value < 0
            for value in hybrid_results
        ),
        "/",
        len(hybrid_results)
    )