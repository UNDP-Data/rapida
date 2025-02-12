import os

import ipywidgets as widgets
from IPython.core.display_functions import display
from ipyfilechooser import FileChooser
from ipyleaflet import DrawControl, GeoJSON
import leafmap

from cbsurge.admin import save
from cbsurge.admin.ocha import fetch_admin

fc = FileChooser(os.getcwd())

def mount_directory(directory):
    """
    Initialize a FileChooser instance for selecting a directory.
    parameters:
        directory - The path to the directory where the FileChooser should be mounted.
    his function sets the global variable `fc` to an instance of `FileChooser`,
    allowing users to navigate and select files within the specified directory.
    """
    global fc
    fc = FileChooser(directory)



def load_ui():
    """
    Initializes and displays the UI elements for selecting, loading and saving the administrative boundaries.
    Raises:
        ValueError:
            If `fc` is None, indicating that the FileChooser has not been initialized.
    This function:
    - Creates an interactive map using leafmap.
    - Allows users to draw a bounding box for data selection.
    - Provides dropdowns for selecting administrative levels.
    - Enables loading and saving of administrative boundary data.

    Ideally this function should be used within a Jupyter Notebook.
    """
    if fc is None:
        raise ValueError("FileChooser (fc) is not initialized. Call mount_directory(directory) first.")
    parameters = {
        'admin_level': 0,
        'bbox': None,
        'selected': None
    }

    data = None
    selected = None

    save_button = widgets.Button(
        description='Save',
        disabled=True,
        button_style='success',
        tooltip='Save',
        icon='arrow-down',
        layout=widgets.Layout(left='6%', width="215px")
    )

    load_button = widgets.Button(
        description='Load to Map',
        disabled=True,
        button_style='info',
        tooltip='Load Selected',
        icon='plus',
        layout=widgets.Layout(left='6%', width="215px")
    )

    def on_file_select(b):
        nonlocal selected
        selected = fc.selected
        if selected and parameters['bbox']:
            save_button.disabled = False

    fc.title = '<b>Save As</b>'
    fc.register_callback(on_file_select)

    def handle_draw(target, action, geo_json):
        if action == 'created' and geo_json['geometry']['type'] == 'Polygon':
            coords = geo_json['geometry']['coordinates'][0]
            min_x = min([c[0] for c in coords])
            max_x = max([c[0] for c in coords])
            min_y = min([c[1] for c in coords])
            max_y = max([c[1] for c in coords])
            bbox = (min_x, min_y, max_x, max_y)
            bbox_widget.value = str(bbox)
            parameters['bbox'] = bbox
            load_button.disabled = False

    m = leafmap.Map(center=[0, 34], zoom=7)
    draw_control = next((c for c in m.controls if isinstance(c, DrawControl)), None)
    if draw_control is None:
        draw_control = DrawControl()
        m.add_control(draw_control)

    draw_control.on_draw(handle_draw)

    admin_level_selector = widgets.Dropdown(
        options=[(f'{i}', i) for i in range(3)],
        value=0,
        disabled=False,
        description="ADM Level"
    )

    bbox_widget = widgets.Text(
        value="None",
        disabled=True,
        placeholder="Bounding box will appear here",
        description="BBOX"
    )

    def load_admin(b):
        nonlocal data
        data = fetch_admin(bbox=parameters['bbox'],
                           admin_level=parameters['admin_level'],
                           clip=True)
        geojson_layer = GeoJSON(data=data, name="Admin Layer")
        m.add_layer(geojson_layer)

    def save_admin(b):
        nonlocal data
        if not data:
            data = fetch_admin(bbox=parameters['bbox'],
                               admin_level=parameters['admin_level'],
                               clip=True)
        save(geojson_dict=data, dst_path=selected)

    load_button.on_click(load_admin)
    save_button.on_click(save_admin)

    def update_params(change):
        parameters["admin_level"] = change["new"]
        print("Updated parameters:", parameters)

    admin_level_selector.observe(update_params, names='value')

    display(m, bbox_widget, admin_level_selector, load_button, fc, save_button)