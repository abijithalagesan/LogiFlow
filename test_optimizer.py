from src.optimization.route_optimizer import optimize_route

locations = {
    "Depot": (13.0827, 80.2707),
    "A": (13.0878, 80.2785),
    "B": (13.0674, 80.2376),
    "C": (13.0604, 80.2496),
    "D": (13.1007, 80.2605),
    "E": (13.0750, 80.2900),
}

start = "Depot"
stops = ["A", "B", "C", "D", "E"]

result = optimize_route(start, stops, locations)

print("\n===== LOGIFLOW ROUTE OPTIMIZATION =====")

print("Original Route:", result["original_route"])
print("Optimized Route:", result["optimized_route"])

print("\nOriginal Distance:",
      result["original_distance_km"], "km")

print("Optimized Distance:",
      result["optimized_distance_km"], "km")

print("Distance Reduction:",
      result["distance_reduction_percent"], "%")
