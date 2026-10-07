import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.optimization.route_optimizer import (
    calculate_route_distance,
    nearest_neighbor,
    two_opt
)

BASE = Path("dataset/almrrc2021-data-training/model_build_inputs")

with open(BASE / "route_data.json") as f:
    routes = json.load(f)

with open(BASE / "actual_sequences.json") as f:
    sequences = json.load(f)

route_id = next(iter(routes))

route_data = routes[route_id]
actual_sequence = sequences[route_id]["actual"]

locations = {
    stop_id: (stop["lat"], stop["lng"])
    for stop_id, stop in route_data["stops"].items()
}

actual_route = [
    stop_id
    for stop_id, _ in sorted(
        actual_sequence.items(),
        key=lambda item: item[1]
    )
]

initial_route = nearest_neighbor(
    actual_route[0],
    actual_route,
    locations
)

optimized_route = two_opt(initial_route, locations)

actual_distance = calculate_route_distance(
    actual_route, locations
)

optimized_distance = calculate_route_distance(
    optimized_route, locations
)

improvement = (
    (actual_distance - optimized_distance)
    / actual_distance * 100
)

print("\n===== LOGIFLOW REAL DATA TEST =====")
print("Route ID:", route_id)
print("Station:", route_data["station_code"])
print("Number of stops:", len(actual_route))

print("\nActual route distance:", round(actual_distance, 2), "km")
print("Optimized distance:", round(optimized_distance, 2), "km")
print("Distance reduction:", round(improvement, 2), "%")

print("\nFirst 15 actual stops:")
print(actual_route[:15])

print("\nFirst 15 optimized stops:")
print(optimized_route[:15])
