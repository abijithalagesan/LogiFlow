import json
import time
import ijson
import statistics

from src.optimization.dijkstra_router import (
    build_graph,
    dijkstra,
    astar,
)


BASE = "dataset/almrrc2021-data-training/model_build_inputs"

ROUTE_DATA = f"{BASE}/route_data.json"
SEQUENCES = f"{BASE}/actual_sequences.json"
TRAVEL_TIMES = f"{BASE}/travel_times.json"

NUMBER_OF_ROUTES = 50
K_NEIGHBORS = 15


print("Loading route data...")

with open(ROUTE_DATA, "r") as f:
    routes = json.load(f)

with open(SEQUENCES, "r") as f:
    sequences = json.load(f)


route_ids = list(routes.keys())[:NUMBER_OF_ROUTES]

print("Routes selected:", len(route_ids))


# --------------------------------------------------
# Load selected travel-time matrices
# --------------------------------------------------

print("\nLoading travel-time data...")

selected_times = {}

with open(TRAVEL_TIMES, "rb") as f:

    for route_id, route_times in ijson.kvitems(f, ""):

        if route_id in route_ids:
            selected_times[route_id] = route_times

        if len(selected_times) == len(route_ids):
            break


print("Travel-time matrices loaded:", len(selected_times))


# --------------------------------------------------
# Run benchmark
# --------------------------------------------------

dijkstra_runtimes = []
astar_runtimes = []

dijkstra_costs = []
astar_costs = []

same_paths = 0
successful_routes = 0


print("\nRunning benchmark...\n")


for index, route_id in enumerate(route_ids, start=1):

    route = routes[route_id]

    stops = route["stops"]

    actual = sequences[route_id]["actual"]

    ordered_stops = [
        stop_id
        for stop_id, _ in sorted(
            actual.items(),
            key=lambda x: x[1]
        )
        if stop_id in stops
    ]

    if len(ordered_stops) < 2:
        continue

    start = ordered_stops[0]
    goal = ordered_stops[-1]

    route_times = selected_times.get(route_id)

    if route_times is None:
        continue


    # --------------------------------------------------
    # Build graph
    # --------------------------------------------------

    graph = build_graph(
        stops,
        route_times,
        k_neighbors=K_NEIGHBORS
    )


    # --------------------------------------------------
    # Dijkstra
    # --------------------------------------------------

    start_time = time.perf_counter()

    dijkstra_path, dijkstra_cost, dijkstra_nodes = dijkstra(
    graph,
    start,
    goal,
    return_stats=True
)

    dijkstra_runtime = (
        time.perf_counter() - start_time
    )


    # --------------------------------------------------
    # A*
    # --------------------------------------------------

    start_time = time.perf_counter()

    astar_path, astar_cost, astar_nodes = astar(
    graph,
    stops,
    start,
    goal,
    return_stats=True
)

    astar_runtime = (
        time.perf_counter() - start_time
    )


    if (
        dijkstra_path is None
        or astar_path is None
    ):
        continue


    successful_routes += 1
    dijkstra_nodes_explored.append(dijkstra_nodes)
    astar_nodes_explored.append(astar_nodes)

    dijkstra_runtimes.append(
        dijkstra_runtime
    )

    astar_runtimes.append(
        astar_runtime
    )

    dijkstra_costs.append(
        dijkstra_cost
    )

    astar_costs.append(
        astar_cost
    )
    dijkstra_nodes_explored = []
    astar_nodes_explored = []


    if (
        dijkstra_path == astar_path
        and abs(dijkstra_cost - astar_cost) < 1e-9
    ):
        same_paths += 1


    if index % 10 == 0:
        print(
            f"Completed {index}/{len(route_ids)} routes"
        )


# --------------------------------------------------
# Results
# --------------------------------------------------

print("\n" + "=" * 60)
print("DIJKSTRA vs A* BENCHMARK")
print("=" * 60)

print("\nSuccessful routes:", successful_routes)

if successful_routes == 0:
    print("No successful routes.")
    raise SystemExit


avg_dijkstra_runtime = statistics.mean(
    dijkstra_runtimes
)

avg_astar_runtime = statistics.mean(
    astar_runtimes
)

median_dijkstra_runtime = statistics.median(
    dijkstra_runtimes
)

median_astar_runtime = statistics.median(
    astar_runtimes
)

avg_dijkstra_cost = statistics.mean(
    dijkstra_costs
)

avg_astar_cost = statistics.mean(
    astar_costs
)
avg_dijkstra_nodes = statistics.mean(
    dijkstra_nodes_explored
)

avg_astar_nodes = statistics.mean(
    astar_nodes_explored
)

median_dijkstra_nodes = statistics.median(
    dijkstra_nodes_explored
)

median_astar_nodes = statistics.median(
    astar_nodes_explored
)

print(
    "\nAverage Dijkstra runtime:",
    round(avg_dijkstra_runtime, 6),
    "seconds"
)

print(
    "Average A* runtime:",
    round(avg_astar_runtime, 6),
    "seconds"
)

print(
    "\nMedian Dijkstra runtime:",
    round(median_dijkstra_runtime, 6),
    "seconds"
)

print(
    "Median A* runtime:",
    round(median_astar_runtime, 6),
    "seconds"
)

print(
    "\nAverage Dijkstra path cost:",
    round(avg_dijkstra_cost, 2)
)

print(
    "Average A* path cost:",
    round(avg_astar_cost, 2)
)

print(
    "\nSame optimal result:",
    f"{same_paths}/{successful_routes}"
)
print(
    "\nAverage Dijkstra nodes explored:",
    round(avg_dijkstra_nodes, 2)
)

print(
    "Average A* nodes explored:",
    round(avg_astar_nodes, 2)
)

print(
    "\nMedian Dijkstra nodes explored:",
    round(median_dijkstra_nodes, 2)
)

print(
    "Median A* nodes explored:",
    round(median_astar_nodes, 2)
)

if avg_astar_runtime > 0:

    speed_ratio = (
        avg_dijkstra_runtime
        / avg_astar_runtime
    )

    print(
        "\nDijkstra/A* runtime ratio:",
        round(speed_ratio, 2)
    )