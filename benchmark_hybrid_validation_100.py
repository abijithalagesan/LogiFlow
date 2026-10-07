import json
import ijson
import statistics
import time

import pandas as pd

from pathlib import Path

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


SAMPLE_SIZE = 100


print("===== LOGIFLOW HYBRID 100-ROUTE BENCHMARK =====")
print()


# --------------------------------------------------
# Load route data
# --------------------------------------------------

with open(ROUTE_FILE) as f:
    routes = json.load(f)

with open(SEQUENCE_FILE) as f:
    sequences = json.load(f)


# Use exactly the same 100 routes as the previous
# benchmark.
import random

eligible_routes = [
    route_id
    for route_id, route in routes.items()
    if len(route["stops"]) >= 50
]

rng = random.Random(123)

route_ids = rng.sample(
    eligible_routes,
    SAMPLE_SIZE
)

wanted = set(route_ids)

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
print(
    "Loaded selected travel-time matrices:",
    len(travel_matrices)
)

print()


# --------------------------------------------------
# Results
# --------------------------------------------------

results = []


for index, route_id in enumerate(
    route_ids,
    start=1
):

    if route_id not in travel_matrices:
        continue


    route_info = routes[route_id]

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


    start_stop = actual_route[0]


    # --------------------------------------------------
    # Actual
    # --------------------------------------------------

    actual_time = calculate_route_travel_time(
        actual_route,
        travel_times
    )


    # --------------------------------------------------
    # NN + 2-opt
    # --------------------------------------------------

    nn_route = nearest_neighbor_by_travel_time(
        start_stop,
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


    # --------------------------------------------------
    # Historical route + 2-opt
    # --------------------------------------------------

    refined_actual_route = two_opt_by_travel_time(
        actual_route,
        travel_times
    )

    refined_actual_time = (
        calculate_route_travel_time(
            refined_actual_route,
            travel_times
        )
    )


    # --------------------------------------------------
    # Hybrid
    #
    # Keep the best route among:
    #
    # 1. Historical route + 2-opt
    # 2. NN + 2-opt
    # 3. Historical route itself
    # --------------------------------------------------

    candidates = [
        (
            actual_route,
            actual_time,
            "historical"
        ),
        (
            current_route,
            current_time,
            "nn_2opt"
        ),
        (
            refined_actual_route,
            refined_actual_time,
            "historical_2opt"
        ),
    ]


    hybrid_route, hybrid_time, hybrid_source = min(
        candidates,
        key=lambda item: item[1]
    )


    # --------------------------------------------------
    # Improvements
    # --------------------------------------------------

    current_improvement = (
        (
            actual_time
            - current_time
        )
        / actual_time
        * 100
        if actual_time > 0
        else 0.0
    )


    hybrid_improvement = (
        (
            actual_time
            - hybrid_time
        )
        / actual_time
        * 100
        if actual_time > 0
        else 0.0
    )


    nn_to_2opt = (
        (
            nn_route
        )
    )


    results.append(
        {
            "route_id": route_id,
            "station": route_info[
                "station_code"
            ],
            "stops": len(actual_route),

            "actual_time":
                actual_time,

            "current_time":
                current_time,

            "hybrid_time":
                hybrid_time,

            "current_improvement":
                current_improvement,

            "hybrid_improvement":
                hybrid_improvement,

            "hybrid_source":
                hybrid_source,
        }
    )


    print(
        f"[{index}] "
        f"{route_info['station_code']} | "
        f"{len(actual_route)} stops | "
        f"Current: {current_improvement:.2f}% | "
        f"Hybrid: {hybrid_improvement:.2f}% | "
        f"Source: {hybrid_source}"
    )


# --------------------------------------------------
# Summary
# --------------------------------------------------

df = pd.DataFrame(results)


print()
print("==============================================")
print("HYBRID 100-ROUTE SUMMARY")
print("==============================================")


current_improvements = (
    df["current_improvement"].tolist()
)

hybrid_improvements = (
    df["hybrid_improvement"].tolist()
)


print(
    "Routes evaluated:",
    len(df)
)

print(
    "Current average improvement:",
    f"{statistics.mean(current_improvements):.2f}%"
)

print(
    "Hybrid average improvement:",
    f"{statistics.mean(hybrid_improvements):.2f}%"
)

print(
    "Current median improvement:",
    f"{statistics.median(current_improvements):.2f}%"
)

print(
    "Hybrid median improvement:",
    f"{statistics.median(hybrid_improvements):.2f}%"
)

print(
    "Current worst:",
    f"{min(current_improvements):.2f}%"
)

print(
    "Hybrid worst:",
    f"{min(hybrid_improvements):.2f}%"
)

print(
    "Current improved:",
    sum(
        value > 0
        for value in current_improvements
    ),
    "/",
    len(df)
)

print(
    "Hybrid improved:",
    sum(
        value > 0
        for value in hybrid_improvements
    ),
    "/",
    len(df)
)

print(
    "Current worsened:",
    sum(
        value < 0
        for value in current_improvements
    ),
    "/",
    len(df)
)

print(
    "Hybrid worsened:",
    sum(
        value < 0
        for value in hybrid_improvements
    ),
    "/",
    len(df)
)


# --------------------------------------------------
# Save
# --------------------------------------------------

df.to_csv(
    "benchmark_hybrid_100.csv",
    index=False
)

print()
print(
    "Saved results to benchmark_hybrid_100.csv"
)