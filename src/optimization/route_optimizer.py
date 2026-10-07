import math


def haversine_distance(point1, point2):
    """
    Calculate the great-circle distance between two GPS coordinates.

    Parameters
    ----------
    point1 : tuple
        (latitude, longitude)
    point2 : tuple
        (latitude, longitude)

    Returns
    -------
    float
        Distance in kilometers.
    """
    lat1, lon1 = point1
    lat2, lon2 = point2

    radius = 6371.0

    lat1, lon1, lat2, lon2 = map(
        math.radians,
        [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    return 2 * radius * math.asin(math.sqrt(a))


def calculate_route_distance(route, locations):
    """
    Calculate the total geographic distance of a route.

    Parameters
    ----------
    route : list
        Ordered list of stop IDs.
    locations : dict
        {stop_id: (latitude, longitude)}

    Returns
    -------
    float
        Total distance in kilometers.
    """
    if len(route) < 2:
        return 0.0

    total_distance = 0.0

    for i in range(len(route) - 1):
        total_distance += haversine_distance(
            locations[route[i]],
            locations[route[i + 1]]
        )

    return total_distance


def nearest_neighbor(start, stops, locations):
    """
    Construct an initial route using the Nearest Neighbor heuristic.

    The first stop remains fixed as the starting point.
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
            key=lambda stop: haversine_distance(
                locations[current],
                locations[stop]
            )
        )

        route.append(next_stop)
        unvisited.remove(next_stop)
        current = next_stop

    return route


def two_opt(route, locations):
    """
    Improve a route using the 2-opt local search algorithm.

    Pairwise distances are precomputed so that every candidate
    swap does not require recalculating the entire route distance.
    """
    n = len(route)

    if n < 4:
        return route[:]

    # Precompute pairwise distances.
    distance = {}

    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_distance(
                locations[route[i]],
                locations[route[j]]
            )

            distance[(route[i], route[j])] = d
            distance[(route[j], route[i])] = d

    def get_distance(a, b):
        return distance[(a, b)]

    best_route = route[:]

    while True:
        improvement_found = False

        for i in range(1, n - 2):
            for j in range(i + 1, n):
                a = best_route[i - 1]
                b = best_route[i]
                c = best_route[j]

                if j == n - 1:
                    old_cost = get_distance(a, b)
                    new_cost = get_distance(a, c)
                else:
                    d = best_route[j + 1]

                    old_cost = (
                        get_distance(a, b)
                        + get_distance(c, d)
                    )

                    new_cost = (
                        get_distance(a, c)
                        + get_distance(b, d)
                    )

                if new_cost < old_cost:
                    best_route[i:j + 1] = reversed(
                        best_route[i:j + 1]
                    )

                    improvement_found = True
                    break

            if improvement_found:
                break

        if not improvement_found:
            break

    return best_route


def optimize_route(start, stops, locations):
    """
    Complete route optimization pipeline.

    Steps:
    1. Preserve the starting stop.
    2. Build a Nearest Neighbor route.
    3. Improve it using 2-opt.
    4. Compare the original and optimized routes.
    """
    if not stops:
        return {
            "original_route": [start],
            "optimized_route": [start],
            "original_distance_km": 0.0,
            "optimized_distance_km": 0.0,
            "distance_reduction_percent": 0.0
        }

    original_route = [start] + [
        stop for stop in stops
        if stop != start
    ]

    initial_route = nearest_neighbor(
        start,
        stops,
        locations
    )

    optimized_route = two_opt(
        initial_route,
        locations
    )

    original_distance = calculate_route_distance(
        original_route,
        locations
    )

    optimized_distance = calculate_route_distance(
        optimized_route,
        locations
    )

    if original_distance > 0:
        improvement = (
            (original_distance - optimized_distance)
            / original_distance
            * 100
        )
    else:
        improvement = 0.0

    return {
        "original_route": original_route,
        "optimized_route": optimized_route,
        "original_distance_km": round(
            original_distance,
            2
        ),
        "optimized_distance_km": round(
            optimized_distance,
            2
        ),
        "distance_reduction_percent": round(
            improvement,
            2
        )
    }
