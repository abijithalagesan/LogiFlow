import json
import time
import ijson

from src.optimization.dijkstra_router import (
    build_graph,
    dijkstra,
    astar,
)


BASE = "dataset/almrrc2021-data-training/model_build_inputs"

ROUTE_DATA = f"{BASE}/route_data.json"
SEQUENCES = f"{BASE}/actual_sequences.json"
TRAVEL_TIMES = f"{BASE}/travel_times.json"


# --------------------------------------------------
# 1. Load one real route
# --------------------------------------------------

print("Loading route data...")

with open(ROUTE_DATA, "r") as f:
    routes = json.load(f)

route_id = next(iter(routes))

route = routes[route_id]
stops = route["stops"]

print("Route ID:", route_id)
print("Number of stops:", len(stops))


# --------------------------------------------------
# 2. Get actual route sequence
# --------------------------------------------------

with open(SEQUENCES, "r") as f:
    sequences = json.load(f)

actual = sequences[route_id]["actual"]

ordered_stops = [
    stop_id
    for stop_id, _ in sorted(
        actual.items(),
        key=lambda x: x[1]
    )
]

ordered_stops = [
    stop_id
    for stop_id in ordered_stops
    if stop_id in stops
]

start = ordered_stops[0]
goal = ordered_stops[-1]

print("Start:", start)
print("Goal:", goal)


# --------------------------------------------------
# 3. Load travel-time matrix for this route
# --------------------------------------------------

print("\nReading travel-time matrix...")

route_times = None

with open(TRAVEL_TIMES, "rb") as f:
    for rid, value in ijson.kvitems(f, ""):
        if rid == route_id:
            route_times = value
            break

if route_times is None:
    raise RuntimeError("Travel-time data not found for route.")

print("Travel-time data loaded.")


# --------------------------------------------------
# 4. Build graph
# --------------------------------------------------

print("\nBuilding graph...")

graph = build_graph(
    stops,
    route_times,
    k_neighbors=15
)

edge_count = sum(len(edges) for edges in graph.values())

print("Graph nodes:", len(graph))
print("Graph edges:", edge_count)


# --------------------------------------------------
# 5. Run Dijkstra
# --------------------------------------------------

print("\nRunning Dijkstra...")

start_time = time.perf_counter()

path, total_time = dijkstra(
    graph,
    start,
    goal
)

runtime = time.perf_counter() - start_time


# --------------------------------------------------
# 6. Results
# --------------------------------------------------

print("\n" + "=" * 50)
print("DIJKSTRA RESULT")
print("=" * 50)

if path is None:
    print("No path found.")
else:
    print("Path found:", True)
    print("Number of nodes in path:", len(path))
    print("Travel time:", round(total_time, 2))
    print("Runtime:", round(runtime, 6), "seconds")

    print("\nFirst few nodes:")
    print(path[:10])

    print("\nLast few nodes:")
    print(path[-5:])
# --------------------------------------------------
# 7. Run A*
# --------------------------------------------------

print("\nRunning A*...")

start_time = time.perf_counter()

astar_path, astar_time = astar(
    graph,
    stops,
    start,
    goal,
)

astar_runtime = time.perf_counter() - start_time


# --------------------------------------------------
# 8. A* Results
# --------------------------------------------------

print("\n" + "=" * 50)
print("A* RESULT")
print("=" * 50)

if astar_path is None:
    print("No path found.")
else:
    print("Path found:", True)
    print("Number of nodes in path:", len(astar_path))
    print("Travel time:", round(astar_time, 2))
    print("Runtime:", round(astar_runtime, 6), "seconds")

    print("\nFirst few nodes:")
    print(astar_path[:10])

    print("\nLast few nodes:")
    print(astar_path[-5:])


# --------------------------------------------------
# 9. Compare
# --------------------------------------------------

print("\n" + "=" * 50)
print("DIJKSTRA vs A*")
print("=" * 50)

if path is not None and astar_path is not None:

    print(
        "Same path:",
        path == astar_path
    )

    print(
        "Same travel time:",
        abs(total_time - astar_time) < 1e-9
    )

    print(
        "Dijkstra runtime:",
        round(runtime, 6),
        "seconds"
    )

    print(
        "A* runtime:",
        round(astar_runtime, 6),
        "seconds"
    )