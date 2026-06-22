import logging
import reverse_geocoder as rg
import click

logger = logging.getLogger(__name__)


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

        return bbox


def get_bbox_label(bbox: tuple[float, float, float, float])->str:
    minlon, minlat, maxlon, maxlat = bbox

    lon = (minlon + maxlon) * .5
    lat = (minlat + maxlat) * .5
    result = rg.search((lat,lon))[0]
    return result


def get_best_semantic_label(bbox: tuple[float, float, float, float]):
    minlon, minlat, maxlon, maxlat = bbox

    lon_center = (minlon + maxlon) * .5
    lat_center = (minlat + maxlat) * .5

    # 2. Get the offline geocode result
    # rg.search expects a list/tuple of tuples
    result = rg.search((lat_center, lon_center))[0]

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