import {
  MapContainer,
  TileLayer,
  Polyline,
  CircleMarker,
  Popup,
  useMap,
} from "react-leaflet";

import { useEffect } from "react";
import "leaflet/dist/leaflet.css";


function FitRouteBounds({ points }) {
  const map = useMap();

  useEffect(() => {
    if (!points || points.length === 0) {
      return;
    }

    map.fitBounds(points, {
      padding: [30, 30],
    });
  }, [map, points]);

  return null;
}


function RouteMap({ routeData }) {

  if (!routeData) {
    return (
      <div className="map-empty">
        Click "Optimize Route" to load the road route.
      </div>
    );
  }


  const coordinates =
    routeData.route_coordinates || {};


  /*
   * REAL ROAD GEOMETRY FROM OSRM
   */

  const actualRoad =
    routeData.actual_road_geometry || [];

  const optimizedRoad =
    routeData.optimized_road_geometry || [];


  /*
   * Fallback to GPS coordinates if road
   * geometry is unavailable.
   */

  const actualPoints =
    actualRoad.length > 0
      ? actualRoad
      : routeData.actual_route
          .map((stop) => {
            const point = coordinates[stop];

            if (!point) {
              return null;
            }

            return [
              point.lat,
              point.lng,
            ];
          })
          .filter(Boolean);


  const optimizedPoints =
    optimizedRoad.length > 0
      ? optimizedRoad
      : routeData.optimized_route
          .map((stop) => {
            const point = coordinates[stop];

            if (!point) {
              return null;
            }

            return [
              point.lat,
              point.lng,
            ];
          })
          .filter(Boolean);


  /*
   * Combine both routes so the map
   * fits everything.
   */

  const boundsPoints = [
    ...actualPoints,
    ...optimizedPoints,
  ];


  return (
    <div className="route-map-wrapper">

      <MapContainer
        className="real-map"
        center={
          actualPoints[0] ||
          [34.09, -118.28]
        }
        zoom={13}
        scrollWheelZoom={true}
      >

        <TileLayer
          attribution="&copy; OpenStreetMap contributors"
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />


        <FitRouteBounds
          points={boundsPoints}
        />


        {/* ACTUAL ROAD ROUTE */}

        <Polyline
          positions={actualPoints}
          pathOptions={{
            color: "#ff6b6b",
            weight: 4,
            opacity: 0.8,
          }}
        />


        {/* OPTIMIZED ROAD ROUTE */}

        <Polyline
          positions={optimizedPoints}
          pathOptions={{
            color: "#6f8cff",
            weight: 5,
            opacity: 0.9,
          }}
        />


        {/* DELIVERY STOP MARKERS */}

        {routeData.actual_route.map(
          (stop, index) => {

            const point =
              coordinates[stop];

            if (!point) {
              return null;
            }

            return (
              <CircleMarker
                key={stop}
                center={[
                  point.lat,
                  point.lng,
                ]}
                radius={3}
                pathOptions={{
                  color: "#ffffff",
                  fillColor: "#ffffff",
                  fillOpacity: 0.9,
                  weight: 1,
                }}
              >

                <Popup>

                  <strong>
                    Stop {index + 1}
                  </strong>

                  <br />

                  Stop ID: {stop}

                  <br />

                  Latitude: {point.lat}

                  <br />

                  Longitude: {point.lng}

                </Popup>

              </CircleMarker>
            );
          }
        )}


        {/* START MARKER */}

        {coordinates[
          routeData.start_stop
        ] && (

          <CircleMarker
            center={[
              coordinates[
                routeData.start_stop
              ].lat,

              coordinates[
                routeData.start_stop
              ].lng,
            ]}
            radius={8}
            pathOptions={{
              color: "#ffffff",
              fillColor: "#42d392",
              fillOpacity: 1,
              weight: 2,
            }}
          >

            <Popup>

              <strong>
                START
              </strong>

              <br />

              Stop ID:{" "}
              {routeData.start_stop}

            </Popup>

          </CircleMarker>

        )}

      </MapContainer>


      {/* MAP LEGEND */}

      <div className="map-legend">

        <div className="legend-item">

          <span className="legend-line actual-line"></span>

          Actual road route

        </div>

        <div className="legend-item">

          <span className="legend-line optimized-line"></span>

          Optimized road route

        </div>

        <div className="legend-item">

          <span className="legend-dot"></span>

          Delivery stop

        </div>

      </div>

    </div>
  );
}


export default RouteMap;