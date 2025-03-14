import leafmap
from IPython.core.display_functions import display
import ipywidgets as widgets



def display_data(gdf=None, raster=None, col=None, cmap='viridis', classification_method='EqualInterval', show_legend=True):
    """
    This function is going to be used to display a geodataframe on a leafmap map.
    :param classification_method:
    :param show_legend: show legend
    :param gdf: geodataframe
    :param raster: raster data
    :param col: column to be used for coloring
    :param cmap: colormap
    :return:
    """
    m = leafmap.Map()

    classifiers = [
        'EqualInterval',
        'FisherJenks',
        'NaturalBreaks',
        'Quantiles',
    ]

    columns = [
        col for col in gdf.columns if col not in ['geometry']
    ]

    visualization_params = {
        'layer_name': 'vector_layer',
        'column': columns[7],
        'cmap': cmap,
        'scheme': classifiers[0],
        'add_legend':True,
        'info_mode': None
    }


    column_selector = widgets.Dropdown(
        options=columns,
        description='Column:',
        disabled=False,
        value=columns[7],
    )
    colormap_selector = widgets.Dropdown(
        options=leafmap.list_palettes(),
        description='Colormap:',
        disabled=False,
        value='viridis'
    )

    classification_method_selector = widgets.Dropdown(
        options=classifiers,
        description='Classification method:',
        disabled=False,
        value='EqualInterval',
    )

    def __switch_column(b):
        try:
            visualization_params['column'] = b['new']
            if gdf[visualization_params['column']].apply(lambda x: isinstance(x, (int, float))).all():
                # all values in the column are numeric
                if classification_method_selector.layout.visibility != 'visible':
                    classification_method_selector.layout.visibility = 'visible'
            else:
                # hide the classification method selector
                classification_method_selector.layout.visibility = 'hidden'
            # if all the values are the same, don't apply classification
            if len(gdf[visualization_params['column']].unique()) == 1:
                # just show the data as is without classification
                m.remove(m.legend_control)
                m.remove(m.find_layer('vector_layer'))
                m.add_data(data=gdf, column=visualization_params['column'], cmap=visualization_params['cmap'], add_legend=True, info_mode=None, layer_name="vector_layer")
                return
            m.remove(m.legend_control)
            m.remove(m.find_layer('vector_layer'))
        except:
            pass
        finally:
            m.add_data(gdf, **visualization_params)


    def __switch_classification_method(b):
        try:
            visualization_params['scheme'] = b['new']
            m.remove(m.legend_control)
            m.remove(m.find_layer('vector_layer'))
        except:
            pass
        finally:
            m.add_data(gdf, **visualization_params)

    def __switch_colormap(b):
        try:
            visualization_params['cmap'] = b['new']
            m.remove(m.legend_control)
            m.remove(m.find_layer('vector_layer'))
        except:
            pass
        finally:
            m.add_data(gdf, **visualization_params)


    column_selector.observe(
        __switch_column,
        names='value'
    )
    colormap_selector.observe(
        __switch_colormap,
        names='value'
    )
    classification_method_selector.observe(
        __switch_classification_method,
        names='value'
    )


    controls = widgets.HBox(
        [column_selector, colormap_selector, classification_method_selector]
    )

    if gdf is not None:
        # all gdf center and zoom
        dissolved = gdf.to_crs(4326).dissolve()
        centroid = dissolved.geometry.centroid.iloc[0]
        lat, lon = centroid.y, centroid.x

        m.set_center(
            lat=lat,
            lon=lon,
            zoom=10
        )

        # don't apply classification if data in column is not numeric
        if not gdf[visualization_params['column']].apply(lambda x: isinstance(x, (int, float))).all():
            gdf[visualization_params['column']].apply(lambda x: print(isinstance(x, (int, float))))
            visualization_params['method'] = None
            # hide the classification method selector
            classification_method_selector.layout.visibility = 'hidden'

        m.add_data(gdf, **visualization_params)

        if raster is not None:
            m.add_raster(raster, colormap="viridis", layer_name='raster_layer', layer_control=True)
    display(controls, m)
    return m