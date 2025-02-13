import os
import ipywidgets as widgets
import leafmap
from IPython.core.display_functions import display
from ipyfilechooser import FileChooser
from ipyleaflet import DrawControl, GeoJSON

from cbsurge.admin import save
from cbsurge.admin.ocha import fetch_admin as fetch_ocha_admin
from cbsurge.admin.osm import fetch_admin as fetch_osm_admin

# Initialize global parameters
global_params = {
    "fc": FileChooser(os.getcwd()),
    "datasource": "ocha",
    "parameters": {
        "admin_level": 0,
        "bbox": None,
        "selected": None,
        "clip": False,
    },
    "data": None,
}


def mount_directory(directory):
    """
    Initialize a FileChooser instance for selecting a directory.
    """
    global_params["fc"] = FileChooser(directory)


def load_ui():
    """
    Initializes and displays the UI elements for selecting, loading, and saving administrative boundaries.
    """
    if global_params["fc"] is None:
        raise ValueError("FileChooser (fc) is not initialized. Call mount_directory(directory) first.")

    fc = global_params["fc"]
    parameters = global_params["parameters"]

    # UI Elements
    save_button = widgets.Button(description='Save', disabled=True, button_style='success', icon='arrow-down')
    load_button = widgets.Button(description='Load to Map', disabled=True, button_style='info', icon='plus')
    clip_select = widgets.Checkbox(value=False, description='Clip to BBOX', indent=False)
    datasource_select = widgets.Dropdown(options=['ocha', 'osm'], value='ocha', description='Data Source')
    admin_level_selector = widgets.Dropdown(options=[(f'ADM {i}', i) for i in range(3)], value=0,
                                            description="ADM Level")
    bbox_widget = widgets.Text(value="None", disabled=True, placeholder="Bounding box will appear here",
                               description="BBOX")

    # Map and drawing control
    m = leafmap.Map(center=[0, 34], zoom=7)
    draw_control = DrawControl()
    m.add(draw_control)

    # Event handlers
    def update_datasource(change):
        global_params["data"] = None
        global_params["datasource"] = change['new']
        print(f"Data source changed to {global_params['datasource']}")

    def on_file_select(b):
        global_params["data"] = None
        parameters['selected'] = fc.selected
        save_button.disabled = not (parameters['selected'] and parameters['bbox'])

    def handle_draw(target, action, geo_json):
        global_params["data"] = None
        if action == 'created' and geo_json['geometry']['type'] == 'Polygon':
            coords = geo_json['geometry']['coordinates'][0]
            bbox = (
            min(c[0] for c in coords), min(c[1] for c in coords), max(c[0] for c in coords), max(c[1] for c in coords))
            bbox_widget.value = str(bbox)
            parameters['bbox'] = bbox
            load_button.disabled = False

    def load_admin(b):
        if global_params["datasource"] == "ocha":
            global_params['data'] = fetch_ocha_admin(bbox=parameters['bbox'], admin_level=parameters['admin_level'], clip=parameters['clip'])
        else:
            global_params['data'] = fetch_osm_admin(bbox=parameters['bbox'], admin_level=parameters['admin_level'], clip=parameters['clip'], osm_level=None)
        m.add_layer(GeoJSON(data=global_params["data"], name="Admin Layer"))

    def save_admin(b):
        if not global_params["data"]:
            load_admin(b)
        save(geojson_dict=global_params["data"], dst_path=parameters['selected'])

    # Observers
    datasource_select.observe(update_datasource, names='value')
    fc.register_callback(on_file_select)
    draw_control.on_draw(handle_draw)
    load_button.on_click(load_admin)
    save_button.on_click(save_admin)
    clip_select.observe(lambda change: parameters.update({'clip': change['new']}), names='value')
    admin_level_selector.observe(lambda change: parameters.update({'admin_level': change['new']}), names='value')

    # UI Layout
    controls = widgets.HBox(
        [datasource_select, admin_level_selector, bbox_widget, fc, clip_select, load_button, save_button])
    layout = widgets.VBox([m, controls])
    display(layout)
