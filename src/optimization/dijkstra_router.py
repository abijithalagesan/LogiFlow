import heapq
import math


# ==========================================================
# Utility
# ==========================================================

def haversine_km(lat1, lon1, lat2, lon2):
    """
    Calculate straight-line distance between two GPS points
    using the Haversine formula.
    """

    radius = 6371.0

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    return 2 * radius * math.asin(math.sqrt(a))


# ==========================================================
# Graph Construction
# ==========================================================

def build_graph(stops, travel_times, k_neighbors=15):
    """
    Build a sparse directed graph.

    Nodes:
        Delivery stops

    Edges:
        Each stop connects to its k nearest geographic neighbors.

    Edge weight:
        ALMRRC travel time from source stop to destination stop.

    Important:
        The graph is directed because A -> B travel time can differ
        from B -> A travel time.
    """

    graph = {
        stop_id: {}
        for stop_id in stops
    }

    stop_ids = list(stops.keys())

    for current_id in stop_ids:

        current = stops[current_id]

        current_lat = current.get("lat")
        current_lng = current.get("lng")

        if current_lat is None or current_lng is None:
            continue

        candidate_neighbors = []

        for other_id in stop_ids:

            if other_id == current_id:
                continue

            other = stops[other_id]

            other_lat = other.get("lat")
            other_lng = other.get("lng")

            if other_lat is None or other_lng is None:
                continue

            distance = haversine_km(
                current_lat,
                current_lng,
                other_lat,
                other_lng,
            )

            candidate_neighbors.append(
                (distance, other_id)
            )

        # Nearest geographic neighbors
        candidate_neighbors.sort(
            key=lambda item: (item[0], item[1])
        )

        for _, neighbor_id in candidate_neighbors[:k_neighbors]:

            travel_time = (
                travel_times
                .get(current_id, {})
                .get(neighbor_id)
            )

            if travel_time is None:
                continue

            travel_time = float(travel_time)

            # Ignore invalid negative travel times.
            if travel_time < 0:
                continue

            graph[current_id][neighbor_id] = travel_time

    return graph


# ==========================================================
# Dijkstra
# ==========================================================

def dijkstra(graph, start, goal, return_stats=False):
    """
    Find the minimum-cost path from start to goal using Dijkstra.

    Edge weights represent travel time.

    Returns:
        return_stats=False:
            (path, cost)

        return_stats=True:
            (path, cost, nodes_explored)
    """

    if start not in graph:
        raise ValueError(
            f"Start node not found: {start}"
        )

    if goal not in graph:
        raise ValueError(
            f"Goal node not found: {goal}"
        )

    # Best known distance from start
    distances = {
        node: float("inf")
        for node in graph
    }

    # Previous node used to reconstruct path
    previous = {
        node: None
        for node in graph
    }

    distances[start] = 0.0

    # (cost, node)
    priority_queue = [
        (0.0, start)
    ]

    visited = set()

    while priority_queue:

        current_distance, current_node = (
            heapq.heappop(priority_queue)
        )

        # Skip stale queue entries / already visited nodes
        if current_node in visited:
            continue

        visited.add(current_node)

        # Goal reached
        if current_node == goal:
            break

        for neighbor, edge_cost in graph[current_node].items():

            new_distance = (
                current_distance + edge_cost
            )

            if new_distance < distances[neighbor]:

                distances[neighbor] = new_distance
                previous[neighbor] = current_node

                heapq.heappush(
                    priority_queue,
                    (
                        new_distance,
                        neighbor,
                    ),
                )

    # ------------------------------------------------------
    # No path
    # ------------------------------------------------------

    if distances[goal] == float("inf"):

        if return_stats:
            return (
                None,
                float("inf"),
                len(visited),
            )

        return None, float("inf")

    # ------------------------------------------------------
    # Reconstruct path
    # ------------------------------------------------------

    path = []

    current = goal

    while current is not None:

        path.append(current)
        current = previous[current]

    path.reverse()

    # ------------------------------------------------------
    # Return
    # ------------------------------------------------------

    if return_stats:
        return (
            path,
            distances[goal],
            len(visited),
        )

    return path, distances[goal]


# ==========================================================
# A*
# ==========================================================

def astar(
    graph,
    stops,
    start,
    goal,
    return_stats=False,
):
    """
    Find the minimum-travel-time path using A*.

    g(n):
        Actual travel time from start to current node.

    h(n):
        Estimated minimum travel time from current node
        to the goal using straight-line distance.

    f(n):
        g(n) + h(n)

    Returns:
        return_stats=False:
            (path, cost)

        return_stats=True:
            (path, cost, nodes_explored)
    """

    if start not in graph:
        raise ValueError(
            f"Start node not found: {start}"
        )

    if goal not in graph:
        raise ValueError(
            f"Goal node not found: {goal}"
        )

    # ------------------------------------------------------
    # Build an admissible travel-time heuristic
    # ------------------------------------------------------

    # Find the largest observed graph speed.
    # This provides a lower bound on travel time.
    max_speed = 0.0

    for source, neighbors in graph.items():

        source_data = stops.get(source, {})

        source_lat = source_data.get("lat")
        source_lng = source_data.get("lng")

        if (
            source_lat is None
            or source_lng is None
        ):
            continue

        for target, travel_time in neighbors.items():

            target_data = stops.get(target, {})

            target_lat = target_data.get("lat")
            target_lng = target_data.get("lng")

            if (
                target_lat is None
                or target_lng is None
            ):
                continue

            if travel_time <= 0:
                continue

            distance = haversine_km(
                source_lat,
                source_lng,
                target_lat,
                target_lng,
            )

            if distance <= 0:
                continue

            speed = distance / travel_time

            if speed > max_speed:
                max_speed = speed

    # Safety fallback
    if max_speed <= 0:
        max_speed = 1.0

    # ------------------------------------------------------
    # Heuristic function
    # ------------------------------------------------------

    def heuristic(node):

        node_data = stops[node]
        goal_data = stops[goal]

        node_lat = node_data.get("lat")
        node_lng = node_data.get("lng")

        goal_lat = goal_data.get("lat")
        goal_lng = goal_data.get("lng")

        if (
            node_lat is None
            or node_lng is None
            or goal_lat is None
            or goal_lng is None
        ):
            return 0.0

        straight_distance = haversine_km(
            node_lat,
            node_lng,
            goal_lat,
            goal_lng,
        )

        return straight_distance / max_speed

    # ------------------------------------------------------
    # A* initialization
    # ------------------------------------------------------

    g_score = {
        node: float("inf")
        for node in graph
    }

    previous = {
        node: None
        for node in graph
    }

    g_score[start] = 0.0

    # (f_score, g_score, node)
    priority_queue = [
        (
            heuristic(start),
            0.0,
            start,
        )
    ]

    visited = set()

    # ------------------------------------------------------
    # A* search
    # ------------------------------------------------------

    while priority_queue:

        _, current_cost, current_node = (
            heapq.heappop(priority_queue)
        )

        if current_node in visited:
            continue

        visited.add(current_node)

        # Goal reached
        if current_node == goal:
            break

        for neighbor, edge_cost in graph[current_node].items():

            new_cost = (
                current_cost + edge_cost
            )

            if new_cost < g_score[neighbor]:

                g_score[neighbor] = new_cost
                previous[neighbor] = current_node

                f_score = (
                    new_cost
                    + heuristic(neighbor)
                )

                heapq.heappush(
                    priority_queue,
                    (
                        f_score,
                        new_cost,
                        neighbor,
                    ),
                )

    # ------------------------------------------------------
    # No path
    # ------------------------------------------------------

    if g_score[goal] == float("inf"):

        if return_stats:
            return (
                None,
                float("inf"),
                len(visited),
            )

        return None, float("inf")

    # ------------------------------------------------------
    # Reconstruct path
    # ------------------------------------------------------

    path = []

    current = goal

    while current is not None:

        path.append(current)
        current = previous[current]

    path.reverse()

    # ------------------------------------------------------
    # Return
    # ------------------------------------------------------

    if return_stats:
        return (
            path,
            g_score[goal],
            len(visited),
        )

    return path, g_score[goal]