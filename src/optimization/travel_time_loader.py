import json
from pathlib import Path

import ijson


BASE_DIR = Path(__file__).resolve().parents[2]

TRAVEL_TIME_FILE = (
    BASE_DIR
    / "dataset"
    / "almrrc2021-data-training"
    / "model_build_inputs"
    / "travel_times.json"
)


def load_route_travel_times(route_id):
    """
    Stream travel_times.json until the requested route is found.

    Returns:
        {
            "from_stop": {
                "to_stop": travel_time
            }
        }
    """

    with open(
        TRAVEL_TIME_FILE,
        "rb"
    ) as file:

        for key, value in ijson.kvitems(
            file,
            ""
        ):

            if key == route_id:
                return value

    raise KeyError(
        f"Travel-time data not found for route: {route_id}"
    )


if __name__ == "__main__":

    route_id = (
        "RouteID_00143bdd-0a6b-49ec-bb35-36593d303e77"
    )

    print("===== TRAVEL TIME LOADER TEST =====")
    print("Route:", route_id)

    travel_times = load_route_travel_times(
        route_id
    )

    print(
        "Number of origin stops:",
        len(travel_times)
    )

    first_origin = next(
        iter(travel_times)
    )

    print(
        "First origin:",
        first_origin
    )

    print(
        "Sample travel times:"
    )

    for destination, value in list(
        travel_times[first_origin].items()
    )[:10]:

        print(
            f"  {first_origin} -> "
            f"{destination}: {value}"
        )