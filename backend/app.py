import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    ROOT
    / "dataset"
    / "almrrc2021-data-training"
    / "model_build_inputs"
)

ROUTE_DATA_FILE = DATASET_DIR / "route_data.json"
ACTUAL_SEQUENCE_FILE = DATASET_DIR / "actual_sequences.json"

FEATURE_FILE = (
    ROOT
    / "dataset"
    / "route_features.csv"
)

MODEL_FILE = (
    ROOT
    / "models"
    / "route_quality_model.joblib"
)


# ============================================================
# PROJECT IMPORTS
# ============================================================

sys.path.insert(0, str(ROOT))

from src.optimization.travel_time_loader import (
    load_route_travel_times,
)

from src.optimization.travel_time_optimizer import (
    optimize_route_by_travel_time,
)

from src.optimization.road_router import (
    get_road_route,
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="LogiFlow API",
    description=(
        "AI-assisted last-mile route analysis "
        "and travel-time optimization."
    ),
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# LOAD DATA ON STARTUP
# ============================================================

print("==============================================")
print("              LOGIFLOW BACKEND")
print("==============================================")

print("Loading ALMRRC route data...")

with open(ROUTE_DATA_FILE, "r") as file:
    routes = json.load(file)

print(f"Loaded {len(routes)} routes.")


print("Loading actual route sequences...")

with open(ACTUAL_SEQUENCE_FILE, "r") as file:
    sequences = json.load(file)

print(f"Loaded {len(sequences)} route sequences.")


print("Loading route features...")

route_features = pd.read_csv(
    FEATURE_FILE
)

print(
    f"Loaded {len(route_features)} feature rows."
)


print("Loading route quality model...")

route_quality_model = joblib.load(
    MODEL_FILE
)

print("Route quality model loaded.")

print("==============================================")


# ============================================================
# MODEL FEATURES
# ============================================================

MODEL_FEATURES = [
    "station_code",
    "executor_capacity_cm3",
    "num_stops",
    "num_dropoffs",
    "num_zones",
    "avg_lat",
    "avg_lng",
    "day_of_week",
    "departure_hour",
]


# ============================================================
# ML ROUTE QUALITY PREDICTION
# ============================================================

def predict_route_quality(route_id: str) -> dict:
    """
    Predict route quality using the trained route-quality model.
    """

    rows = route_features[
        route_features["route_id"] == route_id
    ].copy()

    if rows.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                "Route features not found "
                f"for {route_id}"
            ),
        )

    row = rows.iloc[[0]].copy()

    # Date/time feature preparation.
    row["date"] = pd.to_datetime(
        row["date"]
    )

    row["departure_time_utc"] = pd.to_datetime(
        row["departure_time_utc"]
    )

    row["day_of_week"] = (
        row["date"].dt.dayofweek
    )

    row["departure_hour"] = (
        row["departure_time_utc"].dt.hour
    )

    X = row[MODEL_FEATURES]

    # Prediction.
    prediction = route_quality_model.predict(X)[0]

    probabilities = {}

    if hasattr(
        route_quality_model,
        "predict_proba"
    ):
        probability_values = (
            route_quality_model.predict_proba(X)[0]
        )

        for label, probability in zip(
            route_quality_model.classes_,
            probability_values,
        ):
            probabilities[str(label)] = round(
                float(probability),
                4,
            )

    confidence = None

    if probabilities:
        confidence = round(
            max(probabilities.values()),
            4,
        )

    return {
        "prediction": str(prediction),
        "confidence": confidence,
        "probabilities": probabilities,
    }


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():
    return {
        "project": "LogiFlow",
        "status": "running",
        "routes_loaded": len(routes),
        "model_loaded": True,
    }


# ============================================================
# ROUTE LIST ENDPOINT
# ============================================================

@app.get("/api/routes")
def get_routes():
    """
    Return route summaries used by the frontend route selector.
    """

    route_list = []

    for route_id, route in routes.items():

        route_list.append(
            {
                "route_id": route_id,
                "station": route["station_code"],
                "date": route["date_YYYY_MM_DD"],
                "route_score": route["route_score"],
                "number_of_stops": len(
                    route["stops"]
                ),
            }
        )

    return {
        "total_routes": len(route_list),
        "routes": route_list,
    }


# ============================================================
# ROUTE OPTIMIZATION ENDPOINT
# ============================================================

@app.get(
    "/api/routes/{route_id}/optimize"
)
def optimize_route(route_id: str):
    """
    Complete LogiFlow route-analysis pipeline.

    1. Load actual delivery sequence.
    2. Load ALMRRC travel-time matrix.
    3. Optimize using travel time.
    4. Route both sequences through OSRM.
    5. Generate road geometry.
    6. Predict route quality using ML.
    7. Return results to React.
    """

    # --------------------------------------------------------
    # Validate route
    # --------------------------------------------------------

    if route_id not in routes:
        raise HTTPException(
            status_code=404,
            detail=f"Route not found: {route_id}",
        )

    if route_id not in sequences:
        raise HTTPException(
            status_code=404,
            detail=(
                "Actual sequence not found "
                f"for route: {route_id}"
            ),
        )


    route_data = routes[route_id]

    sequence_data = sequences[
        route_id
    ].get("actual")

    if not sequence_data:
        raise HTTPException(
            status_code=404,
            detail=(
                "Actual route sequence is "
                "missing or empty."
            ),
        )


    # --------------------------------------------------------
    # GPS LOCATIONS
    # --------------------------------------------------------

    locations = {
        stop_id: (
            stop["lat"],
            stop["lng"],
        )
        for stop_id, stop
        in route_data["stops"].items()
    }


    # --------------------------------------------------------
    # ACTUAL DELIVERY SEQUENCE
    # --------------------------------------------------------

    actual_route = [
        stop_id
        for stop_id, _ in sorted(
            sequence_data.items(),
            key=lambda item: item[1],
        )
    ]


    # Remove any sequence IDs that do not have
    # corresponding GPS coordinates.
    actual_route = [
        stop_id
        for stop_id in actual_route
        if stop_id in locations
    ]


    if len(actual_route) < 2:
        raise HTTPException(
            status_code=400,
            detail=(
                "Route does not contain enough "
                "valid stops for optimization."
            ),
        )


    start_stop = actual_route[0]


    # --------------------------------------------------------
    # LOAD TRAVEL-TIME MATRIX
    # --------------------------------------------------------

    print(
        f"[{route_id}] Loading travel-time matrix..."
    )

    try:
        travel_times = (
            load_route_travel_times(
                route_id
            )
        )

    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc


    # --------------------------------------------------------
    # TRAVEL-TIME OPTIMIZATION
    # --------------------------------------------------------

    print(
        f"[{route_id}] Optimizing route..."
    )

    try:
        optimization_result = (
            optimize_route_by_travel_time(
                start_stop,
                actual_route,
                travel_times,
            )
        )

    except KeyError as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Travel-time matrix is missing "
                f"an expected stop pair: {exc}"
            ),
        ) from exc


    optimized_route = (
        optimization_result[
            "optimized_route"
        ]
    )


    original_travel_time = (
        optimization_result[
            "original_travel_time"
        ]
    )

    optimized_travel_time = (
        optimization_result[
            "optimized_travel_time"
        ]
    )

    travel_time_saved = (
        optimization_result[
            "travel_time_saved"
        ]
    )

    travel_time_improvement = (
        optimization_result[
            "improvement_percent"
        ]
    )


    # --------------------------------------------------------
    # OSRM ROAD ROUTING
    # --------------------------------------------------------

    print(
        f"[{route_id}] Routing actual sequence..."
    )

    try:
        actual_road = get_road_route(
            actual_route,
            locations,
        )

    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "Road routing failed for "
                f"actual route: {exc}"
            ),
        ) from exc


    print(
        f"[{route_id}] Routing optimized sequence..."
    )

    try:
        optimized_road = get_road_route(
            optimized_route,
            locations,
        )

    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "Road routing failed for "
                f"optimized route: {exc}"
            ),
        ) from exc


    # --------------------------------------------------------
    # ROAD COMPARISON
    # --------------------------------------------------------

    actual_road_distance = (
        actual_road["distance_km"]
    )

    optimized_road_distance = (
        optimized_road["distance_km"]
    )

    road_distance_saved = round(
        actual_road_distance
        - optimized_road_distance,
        2,
    )


    if actual_road_distance > 0:
        road_improvement = round(
            (
                road_distance_saved
                / actual_road_distance
                * 100
            ),
            2,
        )
    else:
        road_improvement = 0.0


    actual_road_duration = (
        actual_road["duration_min"]
    )

    optimized_road_duration = (
        optimized_road["duration_min"]
    )

    road_duration_saved = round(
        actual_road_duration
        - optimized_road_duration,
        2,
    )


    if actual_road_duration > 0:
        road_duration_improvement = round(
            (
                road_duration_saved
                / actual_road_duration
                * 100
            ),
            2,
        )
    else:
        road_duration_improvement = 0.0


    # --------------------------------------------------------
    # STOP COORDINATES FOR FRONTEND
    # --------------------------------------------------------

    route_coordinates = {
        stop_id: {
            "lat": route_data["stops"][
                stop_id
            ]["lat"],
            "lng": route_data["stops"][
                stop_id
            ]["lng"],
        }
        for stop_id in actual_route
    }


    # --------------------------------------------------------
    # ML ROUTE-QUALITY PREDICTION
    # --------------------------------------------------------

    ml_result = predict_route_quality(
        route_id
    )


    # --------------------------------------------------------
    # FINAL API RESPONSE
    # --------------------------------------------------------

    return {

        # Basic route information.
        "route_id": route_id,

        "station":
            route_data["station_code"],

        "date":
            route_data["date_YYYY_MM_DD"],

        "number_of_stops":
            len(actual_route),

        "start_stop":
            start_stop,


        # Dataset route label.
        "route_score":
            route_data["route_score"],


        # ML prediction.
        "ml_prediction":
            ml_result["prediction"],

        "ml_confidence":
            ml_result["confidence"],

        "ml_probabilities":
            ml_result["probabilities"],


        # Stop sequences.
        "actual_route":
            actual_route,

        "optimized_route":
            optimized_route,


        # ----------------------------------------------------
        # ALMRRC TRAVEL-TIME OPTIMIZATION
        # ----------------------------------------------------

        "original_travel_time":
            original_travel_time,

        "optimized_travel_time":
            optimized_travel_time,

        "travel_time_saved":
            travel_time_saved,

        "travel_time_improvement_percent":
            travel_time_improvement,


        # ----------------------------------------------------
        # OSRM ROAD METRICS
        # ----------------------------------------------------

        "actual_road_distance_km":
            actual_road_distance,

        "optimized_road_distance_km":
            optimized_road_distance,

        "road_distance_saved_km":
            road_distance_saved,

        "road_improvement_percent":
            road_improvement,


        "actual_road_duration_min":
            actual_road_duration,

        "optimized_road_duration_min":
            optimized_road_duration,

        "road_duration_saved_min":
            road_duration_saved,

        "road_duration_improvement_percent":
            road_duration_improvement,


        # ----------------------------------------------------
        # MAP DATA
        # ----------------------------------------------------

        "route_coordinates":
            route_coordinates,

        "actual_road_geometry":
            actual_road["geometry"],

        "optimized_road_geometry":
            optimized_road["geometry"],


        # ----------------------------------------------------
        # ALGORITHM INFORMATION
        # ----------------------------------------------------

        "algorithm":
            "Travel-Time Nearest Neighbor + Directed 2-opt",
        "optimization_source":
            optimization_result["optimization_source"],

        "optimization_objective":
            "Minimize ALMRRC travel time",

        "road_router":
            "OSRM",

    }