import h3.api.basic_int as h3
import logging

from osgeo import gdal, ogr
from rich.progress import Progress

from rapida.util.countries import COUNTRY_CODES

gdal.UseExceptions()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def fetch_admin(bbox=None, admin_level=None, clip=False, destination_path=None, dst_layer_name=None, keep_disputed_areas=False, h3id_precision=7,):
    """
    Fetch ADMIN from CGAZ
    Parameters
    ----------

    bbox: The bounding box to fetch administrative units for.
    admin_level: The administrative level to fetch. Should be an integer.
    clip: bool, False, if True, the admin boundaries are clipped to the bounding box.
    dst_layer_name: The layer name of the destination
    destination_path: The destination path of where to save the dataset
    keep_disputed_areas: Keep disputed areas in the dataset. Default is False.
    h3id_precision: The h3id precision to use. Default is 7.

    Returns
    -------

    """
    with Progress() as progress:
        task = progress.add_task("[cyan]Fetching admin boundaries...", total=2)

        translate_task = progress.add_task(description="Downloading ADMIN data", total=None)

        options = gdal.VectorTranslateOptions(
            layerName=dst_layer_name or f'admin{admin_level}',
            makeValid=True,
            spatFilter=bbox,
            clipSrc=bbox if clip else None,
            geometryType="MultiPolygon",
        )

        url = f"/vsicurl/https://undpngddlsgeohubdev01.blob.core.windows.net/admin/cgaz/geoBoundariesCGAZ_ADM{admin_level}.fgb"

        configs = {
            'GDAL_DISABLE_READDIR_ON_OPEN': 'TRUE',
            'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.fgb',
            'OGR2OGR_USE_ARROW_API': 'YES'
        }


        with gdal.config_options(options=configs):
            ds = gdal.VectorTranslate(destination_path, url, options=options)
            if ds is None:
                logger.error(f"GDAL VectorTranslate failed for URL: {url} with bbox: {bbox}")
                progress.update(task, completed=5, description="[red]GDAL VectorTranslate failed")
                return None

            ds = None
        progress.remove_task(translate_task)

        progress.update(task, advance=1, description="[green]Downloaded admin data", refresh=True)
        with gdal.OpenEx(destination_path, gdal.OF_VECTOR | gdal.OF_UPDATE) as ds:
            layer_name = dst_layer_name or f'admin{admin_level}'
            layer = ds.GetLayerByName(layer_name)
            defn = layer.GetLayerDefn()

            # 1. Dynamically find the source name field (admX_name OR adminX_name)
            potential_names = [f"adm{admin_level}_name", f"admin{admin_level}_name"]
            source_name_field = next((f for f in potential_names if defn.GetFieldIndex(f) >= 0), None)

            if not source_name_field:
                raise Exception(f"Could not find a name field for level {admin_level}")

            # 2. Prepare destination fields
            if defn.GetFieldIndex("h3id") < 0:
                layer.CreateField(ogr.FieldDefn("h3id", ogr.OFTInteger64))
            if defn.GetFieldIndex("name") < 0:
                layer.CreateField(ogr.FieldDefn("name", ogr.OFTString))

            # 3. Single Pass Processing
            fids_to_delete = []
            layer.ResetReading()

            for feature in layer:
                fid = feature.GetFID()

                # Check Disputed Areas
                iso3 = feature.GetField("iso3")
                if not keep_disputed_areas and (iso3 not in COUNTRY_CODES):
                    fids_to_delete.append(fid)
                    feature = None  # Release pointer immediately
                    continue

                # Update Name column
                feature.SetField("name", feature.GetField(source_name_field))

                # Update H3ID
                geom = feature.GetGeometryRef()
                if geom and not geom.IsEmpty():
                    centroid = geom.Centroid()
                    # Note: Centroid() returns a GEOMETRY object, use GetX/Y
                    h3_val = h3.latlng_to_cell(centroid.GetY(), centroid.GetX(), h3id_precision)
                    feature.SetField("h3id", h3_val)

                # Save and release
                layer.SetFeature(feature)
                feature = None  # <--- THIS PREVENTS THE SEGFAULT

            # 4. Finalize deletions after the loop is finished
            for fid in fids_to_delete:
                layer.DeleteFeature(fid)

            ds.FlushCache()
        progress.update(task, advance=1, description="[green]Download Completed", refresh=True)
        return None

