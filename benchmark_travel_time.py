import json
import random
import statistics
import time
import ijson

from pathlib import Path

from src.optimization.travel_time_optimizer import (
    calculate_route_travel_time,
    nearest_neighbor_by_travel_time,
    optimize_route_by_travel_time,
)


ROOT = Path(__file__).resolve().parent

BASE = (
    ROOT
    / "dataset"
    / "almrrc2021-data-training"
    / "model_build_inputs"
)

ROUTE_DATA_FILE = BASE / "route_data.json"
SEQUENCE_FILE = BASE / "actual_sequences.json"
TRAVEL_TIME_FILE = BASE / "travel_times.json"


# --------------------------------------------------
# Benchmark configuration
# --------------------------------------------------

SAMPLE_SIZE = 100
RANDOM_SEED = 42
MIN_STOPS = 50


# --------------------------------------------------
# Load route metadata
# --------------------------------------------------

print("===== LOGIFLOW MULTI-ROUTE BENCHMARK =====")
print()

with open(ROUTE_DATA_FILE) as f:
    routes = json.load(f)

with open(SEQUENCE_FILE) as f:
    sequences = json.load(f)


# Select routes with enough stops.
eligible_routes = [
    route_id
    for route_id, route in routes.items()
    if len(route["stops"]) >= MIN_STOPS
]


rng = random.Random(RANDOM_SEED)

sample_ids = rng.sample(
    eligible_routes,
    min(SAMPLE_SIZE, len(eligible_routes))
)


print("Total routes available:", len(routes))
print("Eligible routes:", len(eligible_routes))
print("Routes selected:", len(sample_ids))
print()


# --------------------------------------------------
# Load selected travel-time matrices
# in ONE streaming pass over the 1.7 GB file.
# --------------------------------------------------

wanted = set(sample_ids)

travel_matrices = {}

print("Streaming travel-time file...")
print("This may take some time on the first pass.")
print()


with open(TRAVEL_TIME_FILE, "rb") as f:

    for route_id, matrix in ijson.kvitems(f, ""):

        if route_id in wanted:

            travel_matrices[route_id] = matrix

            print(
                f"Loaded {len(travel_matrices)}/"
                f"{len(wanted)}: {route_id}"
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
# Benchmark
# --------------------------------------------------

results = []


for index, route_id in enumerate(
    sample_ids,
    start=1
):

    if route_id not in travel_matrices:
        print(
            f"[{index}] Skipping {route_id}: "
            "travel-time matrix not found."
        )
        continue


    route_info = routes[route_id]

    sequence_data = sequences[
        route_id
    ]["actual"]

    travel_times = travel_matrices[
        route_id
    ]


    # Build actual route.
    actual_route = [
        stop_id

        for stop_id, _ in sorted(
            sequence_data.items(),
            key=lambda item: item[1]
        )

        if stop_id in travel_times
    ]


    if len(actual_route) < 4:

        print(
            f"[{index}] Skipping {route_id}: "
            "not enough valid stops."
        )

        continue


    start_stop = actual_route[0]


    # ----------------------------------------------
    # Actual route travel time
    # ----------------------------------------------

    actual_time = calculate_route_travel_time(
        actual_route,
        travel_times
    )


    # ----------------------------------------------
    # Nearest Neighbor
    # ----------------------------------------------

    nn_start = time.perf_counter()

    nn_route = nearest_neighbor_by_travel_time(
        start_stop,
        actual_route,
        travel_times
    )

    nn_time = calculate_route_travel_time(
        nn_route,
        travel_times
    )

    nn_runtime = (
        time.perf_counter()
        - nn_start
    )


    # ----------------------------------------------
    # Nearest Neighbor + Directed 2-opt
    # ----------------------------------------------

    optimization_start = time.perf_counter()

    optimized_result = (
        optimize_route_by_travel_time(
            start_stop,
            actual_route,
            travel_times
        )
    )

    optimization_runtime = (
        time.perf_counter()
        - optimization_start
    )


    optimized_time = optimized_result[
        "optimized_travel_time"
    ]


    # ----------------------------------------------
    # Improvements
    # ----------------------------------------------

    actual_to_optimized = (
        (
            actual_time
            - optimized_time
        )
        / actual_time
        * 100
        if actual_time > 0
        else 0.0
    )


    nn_to_optimized = (
        (
            nn_time
            - optimized_time
        )
        / nn_time
        * 100
        if nn_time > 0
        else 0.0
    )


    results.append(
        {
            "route_id": route_id,
            "station": route_info[
                "station_code"
            ],
            "stops": len(actual_route),
            "route_score": route_info[
                "route_score"
            ],
            "actual_time": actual_time,
            "nn_time": nn_time,
            "optimized_time": optimized_time,
            "actual_improvement":
                actual_to_optimized,
            "nn_improvement":
                nn_to_optimized,
            "runtime_seconds":
                optimization_runtime,
        }
    )


    print(
        f"[{index}] "
        f"{route_info['station_code']} | "
        f"{len(actual_route)} stops | "
        f"Actual: {actual_time:.1f} | "
        f"NN: {nn_time:.1f} | "
        f"2-opt: {optimized_time:.1f} | "
        f"Improvement: {actual_to_optimized:.2f}% | "
        f"Runtime: {optimization_runtime:.2f}s"
    )


# --------------------------------------------------
# Summary
# --------------------------------------------------

print()
print("==============================================")
print("BENCHMARK SUMMARY")
print("==============================================")


if not results:

    print("No benchmark results were produced.")

else:

    improvements = [
        row["actual_improvement"]
        for row in results
    ]

    nn_improvements = [
        row["nn_improvement"]
        for row in results
    ]

    runtimes = [
        row["runtime_seconds"]
        for row in results
    ]

    improved_routes = sum(
        value > 0
        for value in improvements
    )

    unchanged_routes = sum(
        value == 0
        for value in improvements
    )

    worsened_routes = sum(
        value < 0
        for value in improvements
    )


    print(
        "Routes evaluated:",
        len(results)
    )

    print(
        "Average improvement:",
        f"{statistics.mean(improvements):.2f}%"
    )

    print(
        "Median improvement:",
        f"{statistics.median(improvements):.2f}%"
    )

    print(
        "Best improvement:",
        f"{max(improvements):.2f}%"
    )

    print(
        "Worst improvement:",
        f"{min(improvements):.2f}%"
    )

    print()

    print(
        "Routes improved:",
        improved_routes
    )

    print(
        "Routes unchanged:",
        unchanged_routes
    )

    print(
        "Routes worsened:",
        worsened_routes
    )

    print()

    print(
        "Average NN → 2-opt improvement:",
        f"{statistics.mean(nn_improvements):.2f}%"
    )

    print(
        "Average optimization runtime:",
        f"{statistics.mean(runtimes):.2f}s"
    )

    print()

    print("Per-route results:")
    print()

    for row in results:

        print(
            f"{row['station']} | "
            f"{row['stops']} stops | "
            f"Actual {row['actual_time']:.1f} | "
            f"Optimized {row['optimized_time']:.1f} | "
            f"{row['actual_improvement']:.2f}%"
        )
        import pandas as pd

results_df = pd.DataFrame(results)

results_df.to_csv(
    "benchmark_results_100.csv",
    index=False
)

print()
print(
    "Saved results to benchmark_results_100.csv"
)