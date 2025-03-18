from IPython.core.display_functions import display
import ipywidgets as widgets
import leafmap
import rasterio


def display_data(gdf=None, raster=None, col=None, cmap='viridis', classification_method='EqualInterval'):
    """
    Displays a GeoDataFrame or raster on a Leafmap map.
    """
    m = leafmap.Map()
    classifiers = ['EqualInterval', 'FisherJenks', 'NaturalBreaks', 'Quantiles']

    if gdf is not None:
        columns = [c for c in gdf.columns if c != 'geometry']
    else:
        columns = []

    visualization_params = {
        'layer_name': 'vector_layer',
        'column': col or (columns[0] if columns else None),
        'cmap': cmap,
        'scheme': classification_method or classifiers[0],
        'add_legend': True,
        'info_mode': None,
        'raster_min': None,
        'raster_max': None,
    }

    column_selector = widgets.Dropdown(options=columns, description='Column:', value=visualization_params['column'])
    colormap_selector = widgets.Dropdown(options=leafmap.list_palettes(), description='Colormap:', value=cmap)
    classification_selector = widgets.Dropdown(options=classifiers, description='Classification:',
                                               value=classification_method)

    def update_map():
        """ Updates the map visualization. """
        if gdf is not None:
            try:
                m.remove(m.legend_control)
            except:
                pass

            m.remove(m.find_layer('vector_layer'))
            m.add_data(gdf, **visualization_params)
        if raster:
            with rasterio.open(raster) as src:
                no_data = src.nodata
                m.remove(m.find_layer('raster_layer'))
                m.add_raster(
                    source=raster,
                    colormap=visualization_params['cmap'],
                    layer_name='raster_layer',
                    nodata=no_data,
                    vmax=visualization_params['raster_max'],
                    vmin=visualization_params['raster_min'],
                )

    def on_column_change(change):
        visualization_params['column'] = change['new']
        is_numeric = gdf[visualization_params['column']].apply(lambda x: isinstance(x, (int, float))).all()
        classification_selector.layout.visibility = 'visible' if is_numeric else 'hidden'
        if gdf is not None and len(gdf[visualization_params['column']].unique()) == 1:
            # just show the data as is without classification
            m.remove(m.legend_control)
            m.remove(m.find_layer('vector_layer'))
            m.add_data(data=gdf, column=visualization_params['column'], cmap=visualization_params['cmap'],
                       add_legend=True, info_mode=None, layer_name="vector_layer")
            return
        update_map()

    def on_classification_change(change):
        visualization_params['scheme'] = change['new']
        update_map()
        if gdf is not None and len(gdf[visualization_params['column']].unique()) == 1:
            # just show the data as is without classification
            m.remove(m.legend_control)
            m.remove(m.find_layer('vector_layer'))
            m.add_data(data=gdf, column=visualization_params['column'], cmap=visualization_params['cmap'],
                       add_legend=True, info_mode=None, layer_name="vector_layer")
            return

    def on_colormap_change(change):
        visualization_params['cmap'] = change['new']
        if gdf is not None and len(gdf[visualization_params['column']].unique()) == 1:
            # just show the data as is without classification
            m.remove(m.legend_control)
            m.remove(m.find_layer('vector_layer'))
            m.add_data(data=gdf, column=visualization_params['column'], cmap=visualization_params['cmap'],
                       add_legend=True, info_mode=None, layer_name="vector_layer")
            return
        update_map()

    column_selector.observe(on_column_change, names='value')
    classification_selector.observe(on_classification_change, names='value')
    colormap_selector.observe(on_colormap_change, names='value')

    if gdf is not None:
        controls = widgets.HBox([column_selector, colormap_selector, classification_selector])
    elif raster:
        controls = widgets.HBox([colormap_selector])
    else:
        controls = widgets.HBox([colormap_selector, classification_selector])

    if gdf is not None:
        centroid = gdf.to_crs(4326).dissolve().geometry.centroid.iloc[0]
        m.set_center(lat=centroid.y, lon=centroid.x, zoom=10)
        update_map()
        if not gdf[visualization_params['column']].apply(lambda x: isinstance(x, (int, float))).all():
            visualization_params['method'] = None
            # hide the classification method selector
            classification_selector.layout.visibility = 'hidden'


    if raster:
        with rasterio.open(raster) as src:
            no_data = src.nodata
            visualization_params['raster_min'] = src.read(1).min()
            visualization_params['raster_max'] = src.read(1).max()
            min_max_slider = widgets.FloatRangeSlider(
                min=visualization_params['raster_min'],
                max=visualization_params['raster_max'])
            min_max_label = widgets.Label(value='Min/Max:')
            def on_values_change(change):
                visualization_params['raster_min'] = change['new'][0]
                visualization_params['raster_max'] = change['new'][1]
                update_map()


            min_max_slider.observe(on_values_change, names='value')

            leafmap.cog_validate(raster, verbose=True)

            m.add_raster(source=raster,
                         colormap=visualization_params['cmap'],
                         layer_name='raster_layer',
                         zoom_to_layer=True,
                         nodata=no_data,
                         vmax=visualization_params['raster_max'],
                         vmin=visualization_params['raster_min'],
                         attribution="United Nations Development Programme (UNDP)"
                         )
            controls = widgets.HBox([min_max_label, min_max_slider, controls])

    display(controls, m)
    return m

