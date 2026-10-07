import json

from src.optimization.travel_time_loader import (
    load_route_travel_times
)

from src.optimization.travel_time_optimizer import (
    optimize_route_by_travel_time
)


BASE = (
    "dataset/"
    "almrrc2021-data-training/"
    "model_build_inputs/"
)

ROUTE_ID = (
    "RouteID_00143bdd-0a6b-49ec-bb35-36593d303e77"
)


with open(BASE + "actual_sequences.json") as f:
    sequences = json.load(f)


sequence_data = sequences[ROUTE_ID]["actual"]

actual_route = [
    stop_id
    for stop_id, _ in sorted(
        sequence_data.items(),
        key=lambda item: item[1]
    )
]


print("===== LOGIFLOW TRAVEL-TIME OPTIMIZER =====")
print("Route:", ROUTE_ID)
print("Stops:", len(actual_route))


print("\nLoading travel-time matrix...")

travel_times = load_route_travel_times(
    ROUTE_ID
)


result = optimize_route_by_travel_time(
    actual_route[0],
    actual_route,
    travel_times
)


print("\nStarting stop:", actual_route[0])

print(
    "Original travel time:",
    result["original_travel_time"]
)

print(
    "Optimized travel time:",
    result["optimized_travel_time"]
)

print(
    "Travel time saved:",
    result["travel_time_saved"]
)

print(
    "Improvement:",
    result["improvement_percent"],
    "%"
)


print("\nFirst 15 original stops:")
print(result["original_route"][:15])


print("\nFirst 15 optimized stops:")
print(result["optimized_route"][:15])