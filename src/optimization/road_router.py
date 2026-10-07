import subprocess
import os


OSRM_BASE_URL = os.getenv(
    "OSRM_BASE_URL",
    "https://router.project-osrm.org"
)

# Keep requests comfortably small.
MAX_WAYPOINTS = 50


def _request_osrm(coordinates, timeout=60):
    """
    Send one GET request to the OSRM Route service.

    coordinates:
        [(longitude, latitude), ...]
    """

    coordinate_string = ";".join(
        f"{lng},{lat}"
        for lng, lat in coordinates
    )

    url = (
        OSRM_BASE_URL.rstrip("/")
        + "/route/v1/driving/"
        + coordinate_string
        + "?overview=full"
        + "&geometries=geojson"
        + "&steps=false"
    )

    process = subprocess.run(
        [
            "curl",
            "--silent",
            "--show-error",
            "--fail-with-body",
            "--max-time",
            str(timeout),
            url,
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    if process.returncode != 0:
        message = (
            process.stderr.strip()
            or process.stdout.strip()
            or "Unknown curl error"
        )

        raise RuntimeError(
            f"OSRM request failed: {message}"
        )

    try:
        import json

        result = json.loads(process.stdout)

    except Exception as exc:
        raise RuntimeError(
            "OSRM returned invalid JSON."
        ) from exc

    if result.get("code") != "Ok":
        raise RuntimeError(
            "OSRM routing failed: "
            + str(
                result.get(
                    "message",
                    result.get("code")
                )
            )
        )

    return result


def _route_chunk(route, locations, timeout=60):
    """
    Route one ordered chunk of stops.
    """

    coordinates = [
        (
            locations[stop_id][1],
            locations[stop_id][0]
        )
        for stop_id in route
    ]

    result = _request_osrm(
        coordinates,
        timeout=timeout
    )

    road_route = result["routes"][0]

    geometry = [
        [lat, lng]
        for lng, lat in (
            road_route["geometry"]["coordinates"]
        )
    ]

    return {
        "distance_km": (
            road_route["distance"] / 1000
        ),
        "duration_min": (
            road_route["duration"] / 60
        ),
        "geometry": geometry,
    }


def get_road_route(route, locations, timeout=60):
    """
    Convert an ordered delivery sequence into a
    road-network route.

    Large routes are split into smaller consecutive
    requests. One stop overlaps between chunks so
    the road geometry remains continuous.
    """

    if len(route) < 2:
        return {
            "distance_km": 0.0,
            "duration_min": 0.0,
            "geometry": []
        }

    total_distance = 0.0
    total_duration = 0.0
    combined_geometry = []

    start = 0

    while start < len(route) - 1:

        end = min(
            start + MAX_WAYPOINTS,
            len(route)
        )

        chunk = route[start:end]

        print(
            f"Routing stops "
            f"{start + 1}-{end} "
            f"of {len(route)}..."
        )

        result = _route_chunk(
            chunk,
            locations,
            timeout=timeout
        )

        total_distance += result["distance_km"]
        total_duration += result["duration_min"]

        if not combined_geometry:
            combined_geometry.extend(
                result["geometry"]
            )
        else:
            combined_geometry.extend(
                result["geometry"][1:]
            )

        if end >= len(route):
            break

        # Overlap the final stop of this chunk
        # with the first stop of the next chunk.
        start = end - 1

    return {
        "distance_km": round(
            total_distance,
            2
        ),
        "duration_min": round(
            total_duration,
            2
        ),
        "geometry": combined_geometry
    }