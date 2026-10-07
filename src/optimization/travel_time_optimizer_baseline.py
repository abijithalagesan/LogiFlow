from typing import Dict, List


TravelTimes = Dict[str, Dict[str, float]]


def calculate_route_travel_time(
    route: List[str],
    travel_times: TravelTimes
) -> float:
    """
    Calculate total travel time for an ordered route.

    The travel-time matrix is directed, so:
    A -> B can differ from B -> A.
    """

    if len(route) < 2:
        return 0.0

    total = 0.0

    for i in range(len(route) - 1):
        current_stop = route[i]
        next_stop = route[i + 1]

        total += float(travel_times[current_stop][next_stop])

    return total


def nearest_neighbor_by_travel_time(
    start: str,
    stops: List[str],
    travel_times: TravelTimes
) -> List[str]:
    """
    Build an initial route using directed travel time.
    The starting stop remains fixed.
    """

    if not stops:
        return []

    unvisited = set(stops)
    unvisited.discard(start)

    route = [start]
    current = start

    while unvisited:
        next_stop = min(
            unvisited,
            key=lambda stop: travel_times[current][stop]
        )

        route.append(next_stop)
        unvisited.remove(next_stop)
        current = next_stop

    return route


def two_opt_by_travel_time(
    route: List[str],
    travel_times: TravelTimes
) -> List[str]:
    """
    Improve a route using 2-opt while respecting
    the directed travel-time matrix.

    For directed costs, reversing a segment changes
    all internal edge directions, so the complete
    candidate route cost is evaluated.
    """

    if len(route) < 4:
        return route[:]

    best_route = route[:]
    best_cost = calculate_route_travel_time(
        best_route,
        travel_times
    )

    improved = True

    while improved:
        improved = False

        for i in range(1, len(best_route) - 2):

            for j in range(i + 1, len(best_route)):

                candidate = (
                    best_route[:i]
                    + best_route[i:j + 1][::-1]
                    + best_route[j + 1:]
                )

                candidate_cost = (
                    calculate_route_travel_time(
                        candidate,
                        travel_times
                    )
                )

                if candidate_cost < best_cost:

                    best_route = candidate
                    best_cost = candidate_cost

                    improved = True
                    break

            if improved:
                break

    return best_route


def optimize_route_by_travel_time(
    start: str,
    stops: List[str],
    travel_times: TravelTimes
) -> dict:
    """
    Complete travel-time-based route optimization.
    """

    original_route = [
        start
    ] + [
        stop
        for stop in stops
        if stop != start
    ]

    initial_route = nearest_neighbor_by_travel_time(
        start,
        stops,
        travel_times
    )

    optimized_route = two_opt_by_travel_time(
        initial_route,
        travel_times
    )

    original_time = calculate_route_travel_time(
        original_route,
        travel_times
    )

    optimized_time = calculate_route_travel_time(
        optimized_route,
        travel_times
    )

    time_saved = original_time - optimized_time

    improvement = (
        time_saved / original_time * 100
        if original_time > 0
        else 0.0
    )

    return {
        "original_route": original_route,
        "optimized_route": optimized_route,
        "original_travel_time": round(
            original_time,
            2
        ),
        "optimized_travel_time": round(
            optimized_time,
            2
        ),
        "travel_time_saved": round(
            time_saved,
            2
        ),
        "improvement_percent": round(
            improvement,
            2
        )
    }