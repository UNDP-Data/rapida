def is_int(val):
    """
    Check value/variable is integer
    :param val:
    :return:
    """
    if type(val) == int:
        return True
    else:
        if val.is_integer():
            return True
        else:
            return False

def bbox_to_geojson_polygon(west, south, east, north):
    """
    Converts a bounding box to a GeoJSON Polygon geometry.

    Parameters:
        west (float): Western longitude
        south (float): Southern latitude
        east (float): Eastern longitude
        north (float): Northern latitude

    Returns:
        dict: A GeoJSON Polygon geometry representing the bounding box.
    """
    # Define the coordinates of the bounding box as a polygon
    coordinates = [[
        [west, south],  # bottom-left corner
        [west, north],  # top-left corner
        [east, north],  # top-right corner
        [east, south],  # bottom-right corner
        [west, south]  # closing the polygon (back to bottom-left corner)
    ]]


    # Construct a GeoJSON Polygon representation of the bounding box
    geojson = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": coordinates
        }
    }

    return geojson

