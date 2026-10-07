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
            travel_times[
                route[i]
            ][
                route[i + 1]
            ]
        )

    return total


def nearest_neighbor_by_travel_time(
    start: str,
    stops: List[str],
    travel_times: TravelTimes
) -> List[str]:
    """
    Create a deterministic initial route using
    directed travel time.
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
            key=lambda stop: (
                float(
                    travel_times[
                        current
                    ][
                        stop
                    ]
                ),
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

    Candidate swap costs are evaluated using prefix sums,
    avoiding full-route recalculation for every candidate.

    A small tolerance prevents floating-point zero-cost
    swaps from being treated as genuine improvements.
    """

    n = len(route)

    if n < 4:
        return route[:]

    current = route[:]

    while True:

        # ----------------------------------------------
        # Directed edge costs in current route
        # ----------------------------------------------

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


        # ----------------------------------------------
        # Prefix sums
        # ----------------------------------------------

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


        # ----------------------------------------------
        # First-improvement 2-opt search
        # ----------------------------------------------

        for i in range(1, n - 2):

            for j in range(i + 1, n):

                # Old boundary edge.
                old_cost = forward_edges[i - 1]

                # Internal edges before reversal.
                old_cost += (
                    forward_prefix[j]
                    - forward_prefix[i]
                )


                # New boundary edge.
                new_cost = float(
                    travel_times[
                        current[i - 1]
                    ][
                        current[j]
                    ]
                )

                # Internal edges after reversal.
                new_cost += (
                    reverse_prefix[j]
                    - reverse_prefix[i]
                )


                # Second boundary edge.
                if j < n - 1:

                    old_cost += forward_edges[j]

                    new_cost += float(
                        travel_times[
                            current[i]
                        ][
                            current[j + 1]
                        ]
                    )


                delta = (
                    new_cost
                    - old_cost
                )


                # Ignore numerical noise.
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
    """
    Hybrid LogiFlow optimizer.

    Candidates:
      1. Historical route
      2. Historical route + directed 2-opt
      3. Nearest Neighbor + directed 2-opt

    The route with the lowest travel-time cost is selected.
    """

    # --------------------------------------------------
    # Historical route
    # --------------------------------------------------

    historical_route = [
        start
    ] + [
        stop
        for stop in stops
        if stop != start
    ]


    # --------------------------------------------------
    # Historical route cost
    # --------------------------------------------------

    historical_time = (
        calculate_route_travel_time(
            historical_route,
            travel_times
        )
    )


    # --------------------------------------------------
    # Candidate 1:
    # Historical route + directed 2-opt
    # --------------------------------------------------

    historical_optimized_route = (
        two_opt_by_travel_time(
            historical_route,
            travel_times
        )
    )

    historical_optimized_time = (
        calculate_route_travel_time(
            historical_optimized_route,
            travel_times
        )
    )


    # --------------------------------------------------
    # Candidate 2:
    # Nearest Neighbor + directed 2-opt
    # --------------------------------------------------

    nn_route = (
        nearest_neighbor_by_travel_time(
            start,
            historical_route,
            travel_times
        )
    )

    nn_optimized_route = (
        two_opt_by_travel_time(
            nn_route,
            travel_times
        )
    )

    nn_optimized_time = (
        calculate_route_travel_time(
            nn_optimized_route,
            travel_times
        )
    )


    # --------------------------------------------------
    # Choose the best candidate
    # --------------------------------------------------

    candidates = [
        (
            historical_route,
            historical_time,
            "historical"
        ),
        (
            historical_optimized_route,
            historical_optimized_time,
            "historical_2opt"
        ),
        (
            nn_optimized_route,
            nn_optimized_time,
            "nn_2opt"
        ),
    ]


    optimized_route, optimized_time, source = min(
        candidates,
        key=lambda item: item[1]
    )


    # --------------------------------------------------
    # Improvement relative to historical route
    # --------------------------------------------------

    time_saved = (
        historical_time
        - optimized_time
    )


    improvement = (
        time_saved
        / historical_time
        * 100
        if historical_time > 0
        else 0.0
    )


    # --------------------------------------------------
    # Final result
    # --------------------------------------------------

    return {
        "original_route":
            historical_route,

        "optimized_route":
            optimized_route,

        "original_travel_time":
            round(
                historical_time,
                2
            ),

        "optimized_travel_time":
            round(
                optimized_time,
                2
            ),

        "travel_time_saved":
            round(
                time_saved,
                2
            ),

        "improvement_percent":
            round(
                improvement,
                2
            ),

        "optimization_source":
            source,
    }