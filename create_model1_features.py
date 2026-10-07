import csv
import json
import math
import statistics

BASE = "dataset/almrrc2021-data-training/model_build_inputs"
OUTPUT = "dataset/model1_route_sequence_features.csv"


def is_valid_number(value):
    return isinstance(value, (int, float)) and not (
        isinstance(value, float) and math.isnan(value)
    )


def haversine_km(lat1, lon1, lat2, lon2):
    """Distance between two GPS coordinates in kilometers."""
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


# --------------------------------------------------
# Load dataset files
# --------------------------------------------------

print("Loading dataset...")

with open(f"{BASE}/route_data.json") as f:
    routes = json.load(f)

with open(f"{BASE}/travel_times.json") as f:
    travel_times = json.load(f)

with open(f"{BASE}/actual_sequences.json") as f:
    sequences = json.load(f)

with open(f"{BASE}/package_data.json") as f:
    package_data = json.load(f)

print("Routes loaded:", len(routes))


rows = []


# --------------------------------------------------
# Process every route
# --------------------------------------------------

for route_id, route in routes.items():

    stops = route.get("stops", {})
    route_times = travel_times.get(route_id, {})
    actual = sequences.get(route_id, {}).get("actual", {})
    packages = package_data.get(route_id, {})

    # ----------------------------------------------
    # Ordered stop sequence
    # ----------------------------------------------

    ordered_stops = sorted(
        actual.items(),
        key=lambda x: x[1]
    )

    stop_ids = [stop_id for stop_id, _ in ordered_stops]

    # Keep only stops that exist in route data
    stop_ids = [
        stop_id for stop_id in stop_ids
        if stop_id in stops
    ]

    # ----------------------------------------------
    # Basic route statistics
    # ----------------------------------------------

    num_stops = len(stop_ids)

    dropoff_stops = [
        stop_id
        for stop_id in stop_ids
        if stops[stop_id].get("type") == "Dropoff"
    ]

    num_dropoffs = len(dropoff_stops)

    zones = [
        stops[stop_id].get("zone_id")
        for stop_id in stop_ids
        if stops[stop_id].get("zone_id")
    ]

    unique_zones = set(zones)

    # ----------------------------------------------
    # Zone sequence characteristics
    # ----------------------------------------------

    zone_transitions = 0

    for i in range(1, len(zones)):
        if zones[i] != zones[i - 1]:
            zone_transitions += 1

    zone_revisits = 0
    seen_zones = set()

    for zone in zones:
        if zone in seen_zones:
            zone_revisits += 1
        else:
            seen_zones.add(zone)

    # ----------------------------------------------
    # Travel-time sequence features
    # ----------------------------------------------

    leg_times = []

    for i in range(len(stop_ids) - 1):

        current_stop = stop_ids[i]
        next_stop = stop_ids[i + 1]

        time_value = route_times.get(current_stop, {}).get(
            next_stop, 0.0
        )

        if is_valid_number(time_value):
            leg_times.append(float(time_value))

    total_travel_time = sum(leg_times)

    avg_leg_time = (
        statistics.mean(leg_times)
        if leg_times else 0.0
    )

    median_leg_time = (
        statistics.median(leg_times)
        if leg_times else 0.0
    )

    max_leg_time = (
        max(leg_times)
        if leg_times else 0.0
    )

    std_leg_time = (
        statistics.pstdev(leg_times)
        if len(leg_times) > 1 else 0.0
    )

    if leg_times:
        sorted_times = sorted(leg_times)
        index_90 = min(
            len(sorted_times) - 1,
            math.ceil(0.90 * len(sorted_times)) - 1
        )
        p90_leg_time = sorted_times[index_90]
    else:
        p90_leg_time = 0.0

    # ----------------------------------------------
    # Geographic route features
    # ----------------------------------------------

    geographic_leg_distances = []

    for i in range(len(stop_ids) - 1):

        current = stops[stop_ids[i]]
        nxt = stops[stop_ids[i + 1]]

        if (
            is_valid_number(current.get("lat"))
            and is_valid_number(current.get("lng"))
            and is_valid_number(nxt.get("lat"))
            and is_valid_number(nxt.get("lng"))
        ):
            distance = haversine_km(
                current["lat"],
                current["lng"],
                nxt["lat"],
                nxt["lng"]
            )

            geographic_leg_distances.append(distance)

    geographic_route_distance = sum(geographic_leg_distances)

    avg_geo_leg_distance = (
        statistics.mean(geographic_leg_distances)
        if geographic_leg_distances else 0.0
    )

    max_geo_leg_distance = (
        max(geographic_leg_distances)
        if geographic_leg_distances else 0.0
    )

    # ----------------------------------------------
    # Package features
    # ----------------------------------------------

    package_count = 0
    time_window_packages = 0
    total_service_time = 0.0
    total_package_volume = 0.0

    for stop_packages in packages.values():

        for package in stop_packages.values():

            package_count += 1

            window = package.get("time_window", {})

            start = window.get("start_time_utc")
            end = window.get("end_time_utc")

            if (
                isinstance(start, str) and start.strip()
            ) or (
                isinstance(end, str) and end.strip()
            ):
                time_window_packages += 1

            service_time = package.get(
                "planned_service_time_seconds"
            )

            if is_valid_number(service_time):
                total_service_time += float(service_time)

            dimensions = package.get("dimensions", {})

            depth = dimensions.get("depth_cm")
            height = dimensions.get("height_cm")
            width = dimensions.get("width_cm")

            if all(
                is_valid_number(v)
                for v in [depth, height, width]
            ):
                total_package_volume += (
                    float(depth)
                    * float(height)
                    * float(width)
                )

    avg_service_time = (
        total_service_time / package_count
        if package_count else 0.0
    )

    vehicle_capacity = route.get(
        "executor_capacity_cm3", 0.0
    )

    capacity_utilization = (
        total_package_volume / vehicle_capacity
        if vehicle_capacity else 0.0
    )

    # ----------------------------------------------
    # Departure time
    # ----------------------------------------------

    departure = route.get(
        "departure_time_utc", ""
    )

    departure_hour = None

    if departure:
        try:
            departure_hour = int(
                departure.split(":")[0]
            )
        except (ValueError, IndexError):
            departure_hour = None

    # ----------------------------------------------
    # Route target
    # ----------------------------------------------

    route_score = route.get("route_score")

    # ----------------------------------------------
    # Store row
    # ----------------------------------------------

    rows.append({
        "route_id": route_id,
        "station_code": route.get("station_code"),
        "num_stops": num_stops,
        "num_dropoffs": num_dropoffs,
        "num_zones": len(unique_zones),
        "zone_transitions": zone_transitions,
        "zone_revisits": zone_revisits,
        "total_travel_time_seconds": round(
            total_travel_time, 2
        ),
        "avg_leg_time_seconds": round(
            avg_leg_time, 2
        ),
        "median_leg_time_seconds": round(
            median_leg_time, 2
        ),
        "max_leg_time_seconds": round(
            max_leg_time, 2
        ),
        "p90_leg_time_seconds": round(
            p90_leg_time, 2
        ),
        "std_leg_time_seconds": round(
            std_leg_time, 2
        ),
        "geographic_route_distance_km": round(
            geographic_route_distance, 3
        ),
        "avg_geo_leg_distance_km": round(
            avg_geo_leg_distance, 3
        ),
        "max_geo_leg_distance_km": round(
            max_geo_leg_distance, 3
        ),
        "package_count": package_count,
        "time_window_packages": time_window_packages,
        "vehicle_capacity_cm3": vehicle_capacity,
        "total_package_volume_cm3": round(
            total_package_volume, 2
        ),
        "capacity_utilization": round(
            capacity_utilization, 4
        ),
        "avg_service_time_seconds": round(
            avg_service_time, 2
        ),
        "departure_hour": departure_hour,
        "route_score": route_score
    })


# --------------------------------------------------
# Save CSV
# --------------------------------------------------

fields = [
    "route_id",
    "station_code",
    "num_stops",
    "num_dropoffs",
    "num_zones",
    "zone_transitions",
    "zone_revisits",
    "total_travel_time_seconds",
    "avg_leg_time_seconds",
    "median_leg_time_seconds",
    "max_leg_time_seconds",
    "p90_leg_time_seconds",
    "std_leg_time_seconds",
    "geographic_route_distance_km",
    "avg_geo_leg_distance_km",
    "max_geo_leg_distance_km",
    "package_count",
    "time_window_packages",
    "vehicle_capacity_cm3",
    "total_package_volume_cm3",
    "capacity_utilization",
    "avg_service_time_seconds",
    "departure_hour",
    "route_score"
]

with open(OUTPUT, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=fields
    )
    writer.writeheader()
    writer.writerows(rows)


# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\nCreated:", OUTPUT)
print("Total routes:", len(rows))

scores = {}

for row in rows:
    score = row["route_score"]
    scores[score] = scores.get(score, 0) + 1

print("\nRoute score distribution:")

for score in ["High", "Medium", "Low"]:
    count = scores.get(score, 0)
    percentage = (
        count / len(rows) * 100
        if rows else 0
    )

    print(
        f"{score}: {count} "
        f"({percentage:.2f}%)"
    )