import math
import pytest
from typing import Dict, Tuple
import click
from rapida.util import bbox_param_type
from rapida.util.bbox_param_type import (
    BboxParamType,
    buffer_bbox,
    get_bbox_label,
    get_best_semantic_label,
)


@pytest.mark.parametrize(
    "value, expected",
    [
        # a plain bbox over Kampala
        ("32.5,0.2,32.7,0.4", (32.5, 0.2, 32.7, 0.4)),
        # whitespace around each ordinate is stripped
        (" 32.5 , 0.2 , 32.7 , 0.4 ", (32.5, 0.2, 32.7, 0.4)),
        # southern/western hemisphere, negative ordinates
        ("-58.5,-34.7,-58.3,-34.5", (-58.5, -34.7, -58.3, -34.5)),
        # a bbox anchored on the null island corner
        ("0,0,1,1", (0.0, 0.0, 1.0, 1.0)),
        # integers are promoted to float
        ("1,2,3,4", (1.0, 2.0, 3.0, 4.0)),
        # the antimeridian/pole corners are inside the valid domain
        ("-180,-90,180,90", (-180.0, -90.0, 180.0, 90.0)),
    ]
)
def test_convert_accepts_valid_bbox(value: str, expected: Tuple[float, float, float, float]) -> None:
    """
    Test BboxParamType returns a 4 float tuple for well formed input.
    Args:
        value (str): the raw --bbox string as typed on the command line.
        expected (Tuple[float, float, float, float]): the parsed bbox.
    """
    param_type = BboxParamType()
    assert param_type.convert(value, None, None) == expected


@pytest.mark.parametrize(
    "value",
    [
        # too few ordinates
        "1,2,3",
        # too many ordinates
        "1,2,3,4,5",
        # not numbers at all
        "a,b,c,d",
        # empty string
        "",
        # a lone separator
        ",,,",
        # one bad ordinate among three good ones
        "1,2,three,4",
        # semicolons instead of commas
        "1;2;3;4",
    ]
)
def test_convert_rejects_malformed_bbox(value: str) -> None:
    """
    Test BboxParamType fails with a click error when the string is not 4 numbers.
    Args:
        value (str): a malformed --bbox string.
    """
    param_type = BboxParamType()
    with pytest.raises(click.BadParameter):
        param_type.convert(value, None, None)


def test_nan_input_rejection() -> None:
    """
    Test a nan ordinate is refused. float() accepts it, so convert has to check.
    """
    with pytest.raises(click.BadParameter):
        BboxParamType().convert("nan,nan,nan,nan", None, None)


def test_inf_input_rejection() -> None:
    """
    Test a positive inf ordinate is refused.
    """
    with pytest.raises(click.BadParameter):
        BboxParamType().convert("1,2,inf,4", None, None)


def test_negative_inf_input_rejection() -> None:
    """
    Test a negative inf ordinate is refused.
    """
    with pytest.raises(click.BadParameter):
        BboxParamType().convert("-inf,2,3,4", None, None)


def test_longitude_above_maximum_rejection() -> None:
    """
    Test a longitude beyond 180 degrees is refused.
    """
    with pytest.raises(click.BadParameter):
        BboxParamType().convert("200,10,210,20", None, None)


def test_latitude_above_maximum_rejection() -> None:
    """
    Test a latitude beyond 90 degrees is refused.
    """
    with pytest.raises(click.BadParameter):
        BboxParamType().convert("10,100,20,110", None, None)


def test_ordinates_below_minimum_rejection() -> None:
    """
    Test ordinates below -180/-90 are refused.
    """
    with pytest.raises(click.BadParameter):
        BboxParamType().convert("-200,-100,-190,-95", None, None)


def test_inverted_longitude_rejection() -> None:
    """
    Test a bbox whose minimum longitude exceeds its maximum is refused.
    """
    with pytest.raises(click.BadParameter):
        BboxParamType().convert("10,0,5,1", None, None)


def test_inverted_latitude_rejection() -> None:
    """
    Test a bbox whose minimum latitude exceeds its maximum is refused.
    """
    with pytest.raises(click.BadParameter):
        BboxParamType().convert("0,10,1,5", None, None)


def test_zero_area_bbox_rejection() -> None:
    """
    Test a bbox with identical corners is refused before it reaches GDAL.
    """
    with pytest.raises(click.BadParameter):
        BboxParamType().convert("5,5,5,5", None, None)


@pytest.mark.parametrize("meters", [0, 0.0, None])
def test_buffer_bbox_returns_input_when_no_buffer(meters: float) -> None:
    """
    Test buffer_bbox is a no op for a falsy buffer distance.
    Args:
        meters (float): the buffer distance in meters.
    """
    bbox = (32.5, 0.2, 32.7, 0.4)
    assert buffer_bbox(bbox, meters) == bbox


def test_buffer_bbox_grows_on_all_sides() -> None:
    """
    Test buffer_bbox enlarges the bbox in every direction.
    """
    minlon, minlat, maxlon, maxlat = buffer_bbox((0.0, 0.0, 1.0, 1.0), 1000)
    assert minlon < 0.0
    assert minlat < 0.0
    assert maxlon > 1.0
    assert maxlat > 1.0


def test_buffer_bbox_latitude_buffer_is_constant() -> None:
    """
    Test the latitude buffer only depends on the distance, not on the latitude.
    """
    _, low_minlat, _, _ = buffer_bbox((0.0, 0.0, 1.0, 1.0), 1000)
    _, high_minlat, _, _ = buffer_bbox((0.0, 60.0, 1.0, 61.0), 1000)
    assert low_minlat == pytest.approx(0.0 - 1000 / 111320.0)
    assert high_minlat == pytest.approx(60.0 - 1000 / 111320.0)


def test_buffer_bbox_longitude_buffer_widens_towards_the_pole() -> None:
    """
    Test the longitude buffer grows with latitude, since a degree of longitude
    covers less ground away from the equator.
    """
    equator = buffer_bbox((0.0, 0.0, 1.0, 1.0), 1000)
    high_lat = buffer_bbox((0.0, 60.0, 1.0, 61.0), 1000)
    assert (1.0 - equator[0]) < (1.0 - high_lat[0])


def test_buffer_bbox_survives_a_polar_bbox() -> None:
    """
    Test a bbox touching the pole does not divide by zero. cos(90) is clamped,
    so the call returns instead of raising.
    """
    minlon, _, maxlon, _ = buffer_bbox((0.0, 89.9, 1.0, 90.0), 1000)
    assert math.isfinite(minlon)
    assert math.isfinite(maxlon)


def test_buffer_bbox_keeps_latitude_within_range() -> None:
    """
    Test buffering a polar bbox does not push latitude past 90 degrees.
    """
    _, _, _, maxlat = buffer_bbox((0.0, 89.9, 1.0, 90.0), 1000)
    assert maxlat <= 90.0


def test_get_bbox_label_uses_the_bbox_centre(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test get_bbox_label geocodes the centre of the bbox rather than a corner.
    Args:
        monkeypatch (pytest.MonkeyPatch): used to capture the geocoder call.
    """
    seen = {}

    def fake_search(coords):
        seen["coords"] = coords
        return [{"cc": "UG", "admin1": "Central Region", "admin2": "", "name": "Kampala"}]

    monkeypatch.setattr(bbox_param_type.rg, "search", fake_search)
    result = get_bbox_label((32.5, 0.2, 32.7, 0.4))
    assert seen["coords"] == (pytest.approx(0.3), pytest.approx(32.6))
    assert result["name"] == "Kampala"


@pytest.mark.parametrize(
    "record, expected",
    [
        # full hierarchy, every part contributes
        (
            {"cc": "UG", "admin1": "Central Region", "admin2": "Kampala District", "name": "Kampala"},
            "UG_Central_Region_Kampala_District_Kampala",
        ),
        # admin2 repeats admin1 and is dropped
        (
            {"cc": "KE", "admin1": "Nairobi", "admin2": "Nairobi", "name": "Westlands"},
            "KE_Nairobi_Westlands",
        ),
        # the settlement name repeats admin2 and is dropped
        (
            {"cc": "RW", "admin1": "Kigali", "admin2": "Nyarugenge", "name": "Nyarugenge"},
            "RW_Kigali_Nyarugenge",
        ),
        # empty administrative levels are skipped
        (
            {"cc": "ZM", "admin1": "", "admin2": "", "name": "Lusaka"},
            "ZM_Lusaka",
        ),
        # only the country code is known
        (
            {"cc": "NG", "admin1": "", "admin2": "", "name": ""},
            "NG",
        ),
    ]
)
def test_get_best_semantic_label_builds_the_hierarchy(
    record: Dict[str, str], expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Test get_best_semantic_label joins the administrative hierarchy without
    repeating a level and without leaving spaces in the label.
    Args:
        record (Dict[str, str]): a single reverse geocoder result.
        expected (str): the label the record should produce.
        monkeypatch (pytest.MonkeyPatch): used to stub the geocoder.
    """
    monkeypatch.setattr(bbox_param_type.rg, "search", lambda coords: [record])
    assert get_best_semantic_label((32.5, 0.2, 32.7, 0.4)) == expected


def test_get_bbox_label_empty_geocoder_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test get_bbox_label reports an empty geocoder result as a click error rather
    than indexing an empty list.
    Args:
        monkeypatch (pytest.MonkeyPatch): used to stub the geocoder.
    """
    monkeypatch.setattr(bbox_param_type.rg, "search", lambda coords: [])
    with pytest.raises(click.BadParameter):
        get_bbox_label((32.5, 0.2, 32.7, 0.4))


def test_get_best_semantic_label_empty_geocoder_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test get_best_semantic_label reports an empty geocoder result as a click
    error rather than indexing an empty list.
    Args:
        monkeypatch (pytest.MonkeyPatch): used to stub the geocoder.
    """
    monkeypatch.setattr(bbox_param_type.rg, "search", lambda coords: [])
    with pytest.raises(click.BadParameter):
        get_best_semantic_label((32.5, 0.2, 32.7, 0.4))
