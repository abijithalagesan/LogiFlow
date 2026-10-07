import { useEffect, useState } from "react";
import RouteMap from "./RouteMap";

const API_BASE = "http://127.0.0.1:8000";

function App() {
  const [routes, setRoutes] = useState([]);
  const [selectedRouteId, setSelectedRouteId] = useState("");
  const [routeData, setRouteData] = useState(null);

  const [loadingRoutes, setLoadingRoutes] = useState(true);
  const [loadingOptimization, setLoadingOptimization] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    const loadRoutes = async () => {
      try {
        setLoadingRoutes(true);
        setError("");

        const response = await fetch(`${API_BASE}/api/routes`);

        if (!response.ok) {
          throw new Error("Failed to load routes");
        }

        const data = await response.json();

        setRoutes(data.routes);

        if (data.routes.length > 0) {
          setSelectedRouteId(data.routes[0].route_id);
        }
      } catch (err) {
        setError(err.message);
      } finally {
        setLoadingRoutes(false);
      }
    };

    loadRoutes();
  }, []);

  const selectedRoute =
    routes.find(
      (route) => route.route_id === selectedRouteId
    ) || null;

  const optimizeRoute = async () => {
    if (!selectedRouteId) {
      return;
    }

    try {
      setLoadingOptimization(true);
      setError("");

      const response = await fetch(
        `${API_BASE}/api/routes/${selectedRouteId}/optimize`
      );

      if (!response.ok) {
        const message = await response.text();
        throw new Error(
          message || "Failed to optimize route"
        );
      }

      const data = await response.json();

      setRouteData(data);
    } catch (err) {
      setError(err.message);
      setRouteData(null);
    } finally {
      setLoadingOptimization(false);
    }
  };

  const handleRouteChange = (event) => {
    setSelectedRouteId(event.target.value);
    setRouteData(null);
    setError("");
  };

  // --------------------------------------------------
  // Main optimization metrics
  // --------------------------------------------------

  const originalTravelTime =
    routeData?.original_travel_time ?? null;

  const optimizedTravelTime =
    routeData?.optimized_travel_time ?? null;

  const travelTimeSaved =
    routeData?.travel_time_saved ?? null;

  const travelTimeImprovement =
    routeData?.travel_time_improvement_percent ?? null;

  // --------------------------------------------------
  // OSRM validation metrics
  // --------------------------------------------------

  const actualRoadDistance =
    routeData?.actual_road_distance_km ?? null;

  const optimizedRoadDistance =
    routeData?.optimized_road_distance_km ?? null;

  const roadDistanceSaved =
    routeData?.road_distance_saved_km ?? null;

  const actualRoadDuration =
    routeData?.actual_road_duration_min ?? null;

  const optimizedRoadDuration =
    routeData?.optimized_road_duration_min ?? null;

  // --------------------------------------------------
  // Route information
  // --------------------------------------------------

  const numberOfStops =
    routeData?.number_of_stops ??
    selectedRoute?.number_of_stops ??
    0;

  const station =
    routeData?.station ??
    selectedRoute?.station ??
    "—";

  const datasetScore =
    routeData?.route_score ??
    selectedRoute?.route_score ??
    "—";

  const mlPrediction =
    routeData?.ml_prediction ??
    "—";

  const mlConfidence =
    routeData?.ml_confidence ?? null;

  const startStop =
    routeData?.start_stop ?? "—";

  return (
    <div className="app">

      {/* ==================================================
          HEADER
      ================================================== */}

      <header className="topbar">

        <div>
          <div className="brand">
            LOGIFLOW
          </div>

          <div className="subtitle">
            Intelligent Last-Mile Route Optimization
          </div>
        </div>

        <div className="status">
          <span className="status-dot"></span>
          Historical ALMRRC Dataset
        </div>

      </header>


      <main className="dashboard">

        {/* ==================================================
            HERO
        ================================================== */}

        <section className="hero">

          <div>

            <p className="eyebrow">
              ROUTE INTELLIGENCE
            </p>

            <h1>
              Optimize every delivery route.
            </h1>

            <p className="hero-text">
              Analyze historical delivery routes and
              discover more efficient delivery sequences
              using machine learning and travel-time
              optimization.
            </p>

          </div>


          {/* ROUTE SELECTOR */}

          <div className="route-selector">

            <label htmlFor="route-select">
              Select Route
            </label>

            {loadingRoutes ? (

              <div className="route-value">
                Loading routes...
              </div>

            ) : (

              <select
                id="route-select"
                className="route-select"
                value={selectedRouteId}
                onChange={handleRouteChange}
              >

                {routes.map((route) => (

                  <option
                    key={route.route_id}
                    value={route.route_id}
                  >
                    {route.route_id} · {route.station}
                  </option>

                ))}

              </select>

            )}


            {selectedRoute && (

              <div className="route-meta">

                <span>
                  {station}
                </span>

                <span>
                  {numberOfStops} stops
                </span>

                <span>
                  {selectedRoute.date}
                </span>

              </div>

            )}

          </div>

        </section>


        {/* ==================================================
            ERROR
        ================================================== */}

        {error && (

          <div className="error-banner">
            {error}
          </div>

        )}


        {/* ==================================================
            MAIN METRICS
        ================================================== */}

        <section className="metrics">

          {/* ML */}

          <div className="metric-card">

            <span>
              ML ROUTE PREDICTION
            </span>

            <strong className="high">
              {mlPrediction}
            </strong>

            <small>

              {mlConfidence !== null
                ? `${(
                    mlConfidence * 100
                  ).toFixed(1)}% confidence`
                : "Model prediction"}

            </small>

          </div>


          {/* ORIGINAL TRAVEL TIME */}

          <div className="metric-card">

            <span>
              ACTUAL TRAVEL TIME
            </span>

            <strong>

              {originalTravelTime !== null
                ? `${(
                    originalTravelTime / 60
                  ).toFixed(1)} min`
                : "—"}

            </strong>

            <small>
              ALMRRC travel-time cost
            </small>

          </div>


          {/* OPTIMIZED TRAVEL TIME */}

          <div className="metric-card">

            <span>
              OPTIMIZED TRAVEL TIME
            </span>

            <strong>

              {optimizedTravelTime !== null
                ? `${(
                    optimizedTravelTime / 60
                  ).toFixed(1)} min`
                : "—"}

            </strong>

            <small>
              Travel-time optimization
            </small>

          </div>


          {/* IMPROVEMENT */}

          <div className="metric-card">

            <span>
              TIME IMPROVEMENT
            </span>

            <strong className="improvement">

              {travelTimeImprovement !== null
                ? `${travelTimeImprovement}%`
                : "—"}

            </strong>

            <small>

              {travelTimeSaved !== null
                ? `${(
                    travelTimeSaved / 60
                  ).toFixed(1)} min saved`
                : "Travel-time reduction"}

            </small>

          </div>

        </section>


        {/* ==================================================
            WORKSPACE
        ================================================== */}

        <section className="workspace">


          {/* ==================================================
              MAP
          ================================================== */}

          <div className="map-panel">

            <div className="panel-heading">

              <div>

                <p className="eyebrow">
                  ROUTE VISUALIZATION
                </p>

                <h2>
                  Delivery Network
                </h2>

              </div>

              <span className="map-badge">
                {numberOfStops} STOPS
              </span>

            </div>


            <div className="map-placeholder">

              <RouteMap
                routeData={routeData}
              />

            </div>

          </div>


          {/* ==================================================
              ANALYSIS
          ================================================== */}

          <div className="analysis-panel">

            <p className="eyebrow">
              OPTIMIZATION ENGINE
            </p>

            <h2>
              Route Analysis
            </h2>


            <div className="analysis-row">

              <span>
                Station
              </span>

              <strong>
                {station}
              </strong>

            </div>


            <div className="analysis-row">

              <span>
                Starting point
              </span>

              <strong>
                {startStop}
              </strong>

            </div>


            <div className="analysis-row">

              <span>
                Stops analyzed
              </span>

              <strong>
                {numberOfStops}
              </strong>

            </div>


            <div className="analysis-row">

              <span>
                Dataset label
              </span>

              <strong>
                {datasetScore}
              </strong>

            </div>


            <div className="analysis-row">

              <span>
                ML prediction
              </span>

              <strong>
                {mlPrediction}
              </strong>

            </div>


            <div className="analysis-row">

              <span>
                Algorithm
              </span>

              <strong>
                Travel-Time NN + Directed 2-opt
              </strong>

            </div>


            <div className="analysis-row">

              <span>
                Objective
              </span>

              <strong>
                Minimize travel time
              </strong>

            </div>


            <button
              className="optimize-button"
              onClick={optimizeRoute}
              disabled={
                loadingRoutes ||
                loadingOptimization ||
                !selectedRouteId
              }
            >

              {loadingOptimization
                ? "Optimizing..."
                : routeData
                  ? "Route Optimized ✓"
                  : "Optimize Route →"}

            </button>


            {/* ==================================================
                RESULT
            ================================================== */}

            {routeData && (

              <div className="result-box">

                <div className="result-title">
                  Optimization complete
                </div>


                <div className="result-value">

                  {travelTimeSaved !== null
                    ? `${(
                        travelTimeSaved / 60
                      ).toFixed(1)} min saved`
                    : "Optimization complete"}

                </div>


                <div className="result-text">

                  ALMRRC travel time reduced from{" "}

                  {originalTravelTime !== null
                    ? `${(
                        originalTravelTime / 60
                      ).toFixed(1)} min`
                    : "—"}

                  {" "}to{" "}

                  {optimizedTravelTime !== null
                    ? `${(
                        optimizedTravelTime / 60
                      ).toFixed(1)} min`
                    : "—"}.

                </div>


                <div className="result-text">

                  Road validation:{" "}

                  {actualRoadDistance !== null
                    ? `${actualRoadDistance} km`
                    : "—"}

                  {" "}→{" "}

                  {optimizedRoadDistance !== null
                    ? `${optimizedRoadDistance} km`
                    : "—"}

                  {" "}road distance.

                </div>


                <div className="result-text">

                  OSRM duration:{" "}

                  {actualRoadDuration !== null
                    ? `${actualRoadDuration} min`
                    : "—"}

                  {" "}→{" "}

                  {optimizedRoadDuration !== null
                    ? `${optimizedRoadDuration} min`
                    : "—"}.

                </div>


                {roadDistanceSaved !== null && (

                  <div className="result-text">

                    Road distance change:{" "}

                    {roadDistanceSaved > 0
                      ? `-${roadDistanceSaved} km`
                      : `+${Math.abs(
                          roadDistanceSaved
                        )} km`}

                  </div>

                )}

              </div>

            )}

          </div>

        </section>

      </main>

    </div>
  );
}

export default App;