import json
import importlib.util
import time

from src.optimization.travel_time_loader import load_route_travel_times


ROUTE_ID = "RouteID_00143bdd-0a6b-49ec-bb35-36593d303e77"

with open(
    "dataset/almrrc2021-data-training/model_build_inputs/actual_sequences.json"
) as f:
    sequences = json.load(f)

actual_route = [
    stop_id
    for stop_id, _ in sorted(
        sequences[ROUTE_ID]["actual"].items(),
        key=lambda item: item[1]
    )
]

travel_times = load_route_travel_times(ROUTE_ID)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


old = load_module(
    "old_optimizer",
    "src/optimization/travel_time_optimizer_baseline.py"
)

new = load_module(
    "new_optimizer",
    "src/optimization/travel_time_optimizer.py"
)


start = actual_route[0]

starting_route = old.nearest_neighbor_by_travel_time(
    start,
    actual_route,
    travel_times
)


t1 = time.perf_counter()

old_route = old.two_opt_by_travel_time(
    starting_route,
    travel_times
)

old_runtime = time.perf_counter() - t1

old_cost = old.calculate_route_travel_time(
    old_route,
    travel_times
)


t2 = time.perf_counter()

new_route = new.two_opt_by_travel_time(
    starting_route,
    travel_times
)

new_runtime = time.perf_counter() - t2

new_cost = new.calculate_route_travel_time(
    new_route,
    travel_times
)


print("===== OLD vs NEW 2-OPT COMPARISON =====")
print("Stops:", len(starting_route))
print()
print("Baseline cost:", round(old_cost, 2))
print("New cost:", round(new_cost, 2))
print()
print("Baseline runtime:", round(old_runtime, 3), "s")
print("New runtime:", round(new_runtime, 3), "s")
print()
print("Cost difference:", round(new_cost - old_cost, 2))
print("Routes identical:", old_route == new_route)