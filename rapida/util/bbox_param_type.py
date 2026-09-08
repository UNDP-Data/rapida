import logging
import math
import reverse_geocoder as rg
import click

logger = logging.getLogger(__name__)


def buffer_bbox(bbox: tuple[float, float, float, float], meters: float) -> tuple[float, float, float, float]:
    """
    Enlarge a geographic (WGS84 lon/lat) bbox by a distance in meters on all sides.

    Meters are converted to degrees using an approximation at the bbox center latitude.
    Useful for coarse rasters (e.g. NTL ~500m pixels) where a small AOI would otherwise
    contain too few pixels.

    Latitude is clamped to the valid range so a bbox that already touches a pole is not
    pushed past it. Longitude is left alone because a bbox crossing the antimeridian is
    a legitimate AOI.

    :param bbox: (min_lon, min_lat, max_lon, max_lat) in EPSG:4326
    :param meters: buffer distance in meters; 0/None returns the bbox unchanged
    :return: the enlarged bbox
    """
    if not meters:
        return bbox
    minlon, minlat, maxlon, maxlat = bbox
    center_lat = (minlat + maxlat) / 2.0
    dlat = meters / 111320.0
    dlon = meters / (111320.0 * max(math.cos(math.radians(center_lat)), 1e-6))
    return (minlon - dlon, max(-90.0, minlat - dlat), maxlon + dlon, min(90.0, maxlat + dlat))


class BboxParamType(click.ParamType):
    name = "bbox"
    def convert(self, value, param, ctx):
        try:
            bbox = tuple([float(x.strip()) for x in value.split(",")])
            fail = False
        except ValueError:  # ValueError raised when passing non-numbers to float()
            fail = True

        if fail or len(bbox) != 4:
            self.fail(
                f"bbox must be 4 floating point numbers separated by commas. Got '{value}'"
            )

        # nan compares False against everything, so this has to come before the range checks
        if not all(math.isfinite(x) for x in bbox):
            self.fail(f"bbox must contain finite numbers. Got '{value}'")

        minlon, minlat, maxlon, maxlat = bbox

        if not (-180 <= minlon <= 180 and -180 <= maxlon <= 180):
            self.fail(f"bbox longitude must be between -180 and 180. Got '{value}'")

        if not (-90 <= minlat <= 90 and -90 <= maxlat <= 90):
            self.fail(f"bbox latitude must be between -90 and 90. Got '{value}'")

        if minlon >= maxlon:
            self.fail(f"bbox min longitude must be smaller than max longitude. Got '{value}'")

        if minlat >= maxlat:
            self.fail(f"bbox min latitude must be smaller than max latitude. Got '{value}'")

        return bbox


def get_bbox_label(bbox: tuple[float, float, float, float])->dict:
    minlon, minlat, maxlon, maxlat = bbox

    lon = (minlon + maxlon) * .5
    lat = (minlat + maxlat) * .5
    results = rg.search((lat,lon))
    if not results:
        raise click.BadParameter(f"no geocoding result for the center of bbox {bbox}")
    return results[0]


def get_best_semantic_label(bbox: tuple[float, float, float, float]):
    minlon, minlat, maxlon, maxlat = bbox

    lon_center = (minlon + maxlon) * .5
    lat_center = (minlat + maxlat) * .5

    # 2. Get the offline geocode result
    # rg.search expects a list/tuple of tuples
    results = rg.search((lat_center, lon_center))
    if not results:
        raise click.BadParameter(f"no geocoding result for the center of bbox {bbox}")
    result = results[0]

    country = result.get('cc', '')
    admin1 = result.get('admin1', '').strip()
    admin2 = result.get('admin2', '').strip()
    city_name = result.get('name', '').strip()

    # 3. Apply the hierarchy logic
    # We want to build a clean string: Country -> Region -> Local
    label_parts = [country]

    if admin1:
        label_parts.append(admin1)

    # If admin2 exists and isn't just repeating admin1, it's often the most precise local boundary
    if admin2 and admin2 != admin1:
        label_parts.append(admin2)

    # If the specific city/town name is unique and not already covered by admin2
    if city_name and city_name != admin2 and city_name != admin1:
        label_parts.append(city_name)

    # 4. Join with a standardized delimiter and remove spaces
    universal_label = "_".join(label_parts).replace(" ", "_")
    return universal_label