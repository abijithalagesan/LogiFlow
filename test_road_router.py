import json

from src.optimization.road_router import get_road_route


BASE = "dataset/almrrc2021-data-training/model_build_inputs/"


with open(BASE + "route_data.json") as f:
    routes = json.load(f)

with open(BASE + "actual_sequences.json") as f:
    sequences = json.load(f)


route_id = next(iter(routes))

route_data = routes[route_id]

locations = {
    stop_id: (
        stop["lat"],
        stop["lng"]
    )
    for stop_id, stop in route_data["stops"].items()
}

actual_route = [
    stop_id
    for stop_id, _ in sorted(
        sequences[route_id]["actual"].items(),
        key=lambda item: item[1]
    )
]


print("===== LOGIFLOW ROAD ROUTING TEST =====")
print("Route:", route_id)
print("Station:", route_data["station_code"])
print("Stops:", len(actual_route))

result = get_road_route(
    actual_route,
    locations
)

print()
print("Road distance:", result["distance_km"], "km")
print("Road duration:", result["duration_min"], "minutes")
print("Geometry points:", len(result["geometry"]))

print()
print("First 3 geometry points:")

for point in result["geometry"][:3]:
    print(point)