import logging
import click
from cbsurge.stats.ZonalStats import ZonalStats


@click.group()
def stats():
    f"""Command line interface for {__package__} package"""
    pass
@stats.command(no_args_is_help=True)
@click.argument('input_file', type=click.Path())
@click.argument('output_file', type=click.Path())
@click.option('-r', '--raster',
              required=True,
              multiple=True,
              type = str,
              help='The list of input raster files'
              )
@click.option('-o', '--operation',
              required=True,
              multiple=True,
              type=click.Choice([
                  'cell_id', 'center_x', 'center_y', 'coefficient_of_variation', 'count',
                  'coverage', 'frac', 'majority', 'max', 'max_center_x', 'max_center_y',
                  'mean', 'median', 'min', 'min_center_x', 'min_center_y',
                  'minority', 'quantile', 'stdev', 'sum', 'unique',
                  'values', 'variance', 'variety', 'weighted_frac','weighted_mean',
                  'weighted_stdev', 'weighted_sum', 'weighted_variance', 'weights'
              ], case_sensitive=True),
              help=
              """
              Operations to perform (e.g., "sum", "mean", "count"). Can be specified multiple times. 
              See all available operations at https://isciences.github.io/exactextract/operations.html
              """
              )
@click.option('-c', '--column',
              required=False,
              multiple=True,
              help=
              """
              list of column names for each operation. The number of columns must match the number of operations multiplied by the number of rasters.
              If this option is not specified, raster file name is used as prefix of column names.
              For example, if aaa.tif is specified, column name will be 'aaa_sum'.
              """
              )
@click.option('-s','--target-crs',
              required=False,
              default="EPSG:3857",
              type=str,
              help='SRID for output vector file. Default is EPSG:3857',
              )
@click.option('--debug',
              is_flag=True,
              default=False,
              help="Set log level to debug"
              )
def compute(
        input_file=None,
        output_file=None,
        raster=None,
        target_crs="EPSG:3857",
        operation=None,
        column=None,
        debug=False):
    """
    This command compute zonal statistics with given raster files from a vector file, and save the result to output file.

    Arguments: \n
        The below arguments must be specified to execute the command apart from optional parameters.

        - input_file: input file must be vector file such as geojson, flatgeobuf.\n
        - output_file: output file format supports Shapefile (.shp), GeoJSON (.geojson), FlatGeobuf (.fgb) and GeoPackage (.gpkg)

    Usage: \n
        The below command provides how to use the command to compute zonal statistics.\n
        rapida stats compute --help

    Example: \n
        rapida stats compute ./cbsurge/stats/tests/assets/admin2.geojson ./cbsurge/stats/tests/assets/admin2_stats.fgb -r ./cbsurge/stats/tests/assets/rwa_m_5_2020_constrained_UNadj.tif -r ./cbsurge/stats/tests/assets/rwa_f_5_2020_constrained_UNadj.tif -o sum -c male_5_sum -c female_5_sum
    """
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)

    with ZonalStats(input_file, target_crs="ESRI:54009") as st:
        st.compute(raster, operations=operation, operation_cols=column)
        st.write(output_file, target_crs=target_crs)
