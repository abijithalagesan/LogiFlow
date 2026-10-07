from typing import Dict, List


TravelTimes = Dict[str, Dict[str, float]]


def calculate_route_travel_time(
    route: List[str],
    travel_times: TravelTimes
) -> float:
    """Calculate total directed travel time."""

    if len(route) < 2:
        return 0.0

    total = 0.0

    for i in range(len(route) - 1):
        total += float(
            travel_times[route[i]][route[i + 1]]
        )

    return total


def nearest_neighbor_by_travel_time(
    start: str,
    stops: List[str],
    travel_times: TravelTimes
) -> List[str]:
    """Create initial route using directed travel time."""

    if not stops:
        return []

    unvisited = set(stops)
    unvisited.discard(start)

    route = [start]
    current = start

    while unvisited:
        next_stop = min(
            unvisited,
            key=lambda stop: (
                float(travel_times[current][stop]),
                stop
            )
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
    Directed 2-opt using first-improvement search.

    The search order matches the original baseline
    implementation, while candidate costs are evaluated
    using O(1) prefix-sum calculations.
    """

    n = len(route)

    if n < 4:
        return route[:]

    current = route[:]

    while True:

        # Current forward edge costs.
        forward_edges = [
            float(
                travel_times[
                    current[i]
                ][
                    current[i + 1]
                ]
            )
            for i in range(n - 1)
        ]

        # Costs of the same edges in reverse direction.
        reverse_edges = [
            float(
                travel_times[
                    current[i + 1]
                ][
                    current[i]
                ]
            )
            for i in range(n - 1)
        ]

        # Prefix sums allow internal reversed-segment costs
        # to be evaluated in O(1).
        forward_prefix = [0.0]

        reverse_prefix = [0.0]

        for i in range(n - 1):
            forward_prefix.append(
                forward_prefix[-1]
                + forward_edges[i]
            )

            reverse_prefix.append(
                reverse_prefix[-1]
                + reverse_edges[i]
            )

        improvement_found = False

        # IMPORTANT:
        # Same i/j traversal order as the baseline.
        for i in range(1, n - 2):

            for j in range(i + 1, n):

                # Original cost of affected edges.
                old_cost = forward_edges[i - 1]

                # New direction of the boundary edge.
                new_cost = float(
                    travel_times[
                        current[i - 1]
                    ][
                        current[j]
                    ]
                )

                # Internal edges inside the reversed segment.
                old_internal = (
                    forward_prefix[j]
                    - forward_prefix[i]
                )

                new_internal = (
                    reverse_prefix[j]
                    - reverse_prefix[i]
                )

                old_cost += old_internal
                new_cost += new_internal

                # Second boundary edge exists when j is not
                # the final stop.
                if j < n - 1:

                    old_cost += forward_edges[j]

                    new_cost += float(
                        travel_times[
                            current[i]
                        ][
                            current[j + 1]
                        ]
                    )

                delta = new_cost - old_cost

                # Same first-improvement behavior as baseline.
                if delta < -1e-9:

                    current[
                        i:j + 1
                    ] = reversed(
                        current[i:j + 1]
                    )

                    improvement_found = True
                    break

            if improvement_found:
                break

        if not improvement_found:
            break

    return current


def optimize_route_by_travel_time(
    start: str,
    stops: List[str],
    travel_times: TravelTimes
) -> dict:
    """Complete travel-time optimization pipeline."""

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

    time_saved = (
        original_time - optimized_time
    )

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
