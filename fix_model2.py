import json
import csv

BASE = "almrrc2021-data-training/model_build_inputs"

# Load package data
with open(f"{BASE}/package_data.json") as f:
    package_data = json.load(f)

# Load existing route-duration features
with open("route_duration_features.csv") as f:
    rows = list(csv.DictReader(f))

# Correct the time-window count
for row in rows:
    route_id = row["route_id"]
    packages_by_stop = package_data.get(route_id, {})

    time_window_packages = 0

    for packages in packages_by_stop.values():
        for package in packages.values():

            window = package.get("time_window", {})

            start = window.get("start_time_utc")
            end = window.get("end_time_utc")

            # Count only real timestamps
            if isinstance(start, str) and start.strip():
                time_window_packages += 1
            elif isinstance(end, str) and end.strip():
                time_window_packages += 1

    row["time_window_packages"] = time_window_packages


# Keep only valid Model 2 features
fields = [
    "route_id",
    "station_code",
    "num_stops",
    "num_zones",
    "package_count",
    "time_window_packages",
    "vehicle_capacity_cm3",
    "total_package_volume_cm3",
    "avg_service_time_seconds",
    "departure_hour",
    "route_duration_hours"
]

# Write clean dataset
with open("model2_route_duration_clean.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    clean_rows = [
        {field: row[field] for field in fields}
        for row in rows
    ]

    writer.writerows(clean_rows)

print("Created: model2_route_duration_clean.csv")
print("Rows:", len(rows))

print("\nFirst 5 routes:")
for row in rows[:5]:
    print(
        row["route_id"],
        "| packages =", row["package_count"],
        "| time_windows =", row["time_window_packages"]
    )