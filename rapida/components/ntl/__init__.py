from rapida.core.component import Component
from rapida.core.variable import Variable
from rapida.project.project import Project
from rapida.session import Session
import asyncio
import datetime
import glob
import logging
import os
import re
import click

logger = logging.getLogger('rapida')

class NtlComponent(Component):
    """
    Nighttime Lights (NTL) power-outage detection component for `rapida assess -c ntl`.

    Detects power outages by comparing a target night's VIIRS nighttime-lights radiance
    against a recent monthly baseline, then summarizes the result per admin polygon in a
    ``stats.ntl`` layer of the project geopackage.

    Variables (data streams), pass with ``-v``:
      - ``noaa_outage``     : NOAA real-time VIIRS data. Usually the most reliable choice.
      - ``nasa_nrt_outage`` : NASA Black Marble operational / near-real-time (LANCE). Last ~7 days only.
      - ``nasa_outage``     : NASA Black Marble archived (LAADS). Bottom-of-atmosphere; frequently
                              flagged as cloudy, so results can be sparse.
    If ``-v`` is omitted, all three variables are assessed.

    Options (defined on the assess command):
      - ``--outage-date YYYY-MM-DD`` : REQUIRED. The target night to assess.
      - ``--popvar NAME``            : Population variable(s) whose affected population is summed over
                                       outages. If omitted, the population count-variables already
                                       present in the project are auto-selected.
      - ``-ot / --percentage-drop N``: Radiance-drop threshold (%) to flag a pixel as an outage. Default 50.
      - ``-cm / --cmask``            : Enable cloud masking. Off by default to avoid over-masking.
      - ``--bbox-buffer METERS``     : Enlarge the project bbox before detection (NTL pixels are ~500m,
                                       which helps small areas of interest). Default 0 (no buffer).

    Credentials (read from the environment / .env):
      - NASA streams require ``EARTHDATA_TOKEN``.
      - NOAA stream requires ``SPACETRACK_USER`` and ``SPACETRACK_PASSWORD``.

    Outputs:
      - ``<project>/data/ntl/<DELIVERABLE>_<area>.tif`` : multi-band outage raster.
      - ``stats.ntl`` layer in the project geopackage with, per admin polygon:
          ``<name>_outage_pixels`` (outage cell count), ``<name>_outage_pct`` (% of area in outage),
          and ``<name>_<popvar>_affected`` (population inside outage areas), reusing the population
          rasters already in the project.

    Examples:
      rapida assess -c ntl -v noaa_outage --outage-date 2026-07-20
          Detect outages for one night using NOAA data.

      rapida assess -c ntl -v noaa_outage --outage-date 2026-07-20 -ot 40 --bbox-buffer 2000
          Loosen the threshold and widen the search window for a small area of interest.

      rapida assess -c ntl -v nasa_nrt_outage --outage-date 2026-07-20 --popvar total
          Also compute the affected total population per admin polygon.

    Tip: to inspect input data quality for the same bbox and date, use `rapida ntl search`.
    """

    def __call__(self, variables: list[str],  **kwargs):
        if not variables:
            variables = self.variables
        else:
            for var_name in variables:
                if var_name not in self.variables:
                    logger.error(f'variable "{var_name}" is invalid. Valid options are "{", ".join(self.variables)}"')
                    return

        with Session() as session:
            variable_data = session.get_component(self.component_name)

            for var_name in variables:
                var_data = variable_data[var_name]

                v = NTLVariable(
                    name=var_name,
                    component=self.component_name,

                    **var_data
                )
                v(**kwargs)


class NTLVariable(Variable):


    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        project = Project(path=os.getcwd())
        geopackage_path = project.geopackage_file_path
        output_filename = f"{self.name}.tif"
        self.local_path = os.path.join(os.path.dirname(geopackage_path), self.component, output_filename)

    def download(self, **kwargs):
        pass

    def resolve(self, **kwargs):
        pass

    def _check_credentials(self, deliverable):
        """
        Fail fast with a clear message if the credentials required by the target
        data stream are not set, instead of raising deep inside the pipeline.
        """
        missing = []
        if 'NASA' in deliverable and not os.environ.get('EARTHDATA_TOKEN'):
            missing.append('EARTHDATA_TOKEN')
        if 'NOAA' in deliverable:
            for var_name in ('SPACETRACK_USER', 'SPACETRACK_PASSWORD'):
                if not os.environ.get(var_name):
                    missing.append(var_name)
        if missing:
            raise click.UsageError(
                f"Missing credentials for {deliverable}: {', '.join(missing)}. "
                f"Set them in your .env / environment."
            )

    def _run_async(self, coro):
        """
        Run an async coroutine from the synchronous assess flow. Mirrors the
        AsyncCommand pattern in rapida/cli/aclick.py: nest_asyncio is applied
        globally at rapida.cli import time, so run_until_complete tolerates being
        called from an already-async context.
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)

    def compute(self, force=False, outage_date=None, progress=None, year=None, pop_vars=None,
                percentage_drop=50, mask_clouds=False, bbox_buffer=0.0, **kwargs):
        # Imported here (not at module top) to avoid a circular import:
        # rapida.ntl.outage imports rapida.cli.assess.
        from rapida.ntl.outage import detect_outage
        from rapida.util.bbox_param_type import buffer_bbox

        if outage_date is None:
            raise click.UsageError("The ntl component requires --outage-date (YYYY-MM-DD).")

        deliverable = self.name.upper()
        self._check_credentials(deliverable)

        project = Project(path=os.getcwd())
        bbox = project.geobounds
        if bbox_buffer:
            bbox = buffer_bbox(bbox, bbox_buffer)

        dst_dir = os.path.dirname(self.local_path)
        os.makedirs(dst_dir, exist_ok=True)

        # Save the outage analysis visualization (same 6-panel plot as
        # `ntl detect --display`) into the project's ntl folder.
        display_path = os.path.join(dst_dir, f"{self.name}_outage.png")

        # NOTE: pop_vars is intentionally NOT passed to detect_outage. detect_outage's own
        # pop_vars path builds a throwaway Project from the outage polygons and re-downloads
        # population into a separate gpkg. In the assess flow we instead reuse the population
        # data already in the project and write a stats.ntl layer (see self.evaluate).
        coro = detect_outage(
            bbox=bbox,
            nominal_date=outage_date,
            deliverable=deliverable,
            dst_dir=dst_dir,
            mask_clouds=mask_clouds,
            percentage_drop=percentage_drop,
            pop_vars=None,
            display=False,
            display_path=display_path,
            progress=progress,
            year=year or datetime.datetime.now().year,
        )
        outage_tif = self._run_async(coro)

        if outage_tif and os.path.exists(outage_tif):
            self.evaluate(outage_tif=outage_tif, pop_vars=pop_vars or None, year=year, progress=progress)
        else:
            logger.info(f'No outage output produced for {deliverable}; nothing to evaluate.')
        return outage_tif

    def evaluate(self, outage_tif=None, pop_vars=None, year=None, progress=None, **kwargs):
        """
        Compute per-admin-polygon NTL statistics from the outage raster and write them
        into the project geopackage as a stats.<component> (stats.ntl) layer.

        Statistics per polygon:
          - <name>_outage_pixels : count of outage cells
          - <name>_outage_pct    : percentage of polygon area flagged as outage
          - <name>_<popvar>_affected : population inside outage areas (population raster x outage),
                                       reusing the population rasters already in the project.
        """
        import geopandas as gpd
        from rapida.util import geo
        from rapida.stats.raster_zonal_stats import zst

        project = Project(path=os.getcwd())
        ntl_dir = os.path.dirname(self.local_path)

        # When no pop_vars were requested, auto-select the population count-variables that
        # already have data in this project, so affected population is computed out of the box.
        if not pop_vars:
            pop_vars = self._discover_population_vars(project, year)
            if pop_vars:
                logger.info(
                    f'Auto-selected population variables for affected-population stats: {", ".join(pop_vars)}'
                )
            else:
                logger.info(
                    'No population variables found in the project; writing outage extent only. '
                    'Run `rapida assess -c population` first to include affected population.'
                )

        # 1. Collapse the OUTAGE band(s) into a single 0/1 mask in the source CRS (EPSG:4326).
        outage_4326 = os.path.join(ntl_dir, f'{self.name}_outage_mask_4326.tif')
        self._extract_outage_mask(outage_tif, outage_4326)

        # 2. Reproject/crop/align the outage mask to the project CRS for zonal stats.
        outage_proj = os.path.join(ntl_dir, f'{self.name}_outage_mask.tif')
        geo.import_raster(
            source=outage_4326, dst=outage_proj, target_srs=project.target_srs,
            crop_ds=project.geopackage_file_path, crop_layer_name=project.polygons_layer_name,
            progress=progress,
        )

        # 3. Outage extent per polygon (sum = cell count, mean = fraction -> percentage).
        src_rasters = [outage_proj, outage_proj]
        vars_ops = [(f'{self.name}_outage_pixels', 'sum'), (f'{self.name}_outage_fraction', 'mean')]

        # 4. Affected population per requested pop_var, reusing project population rasters.
        for pv in (pop_vars or ()):
            pop_raster = self._find_population_raster(project, pv, year)
            if not pop_raster:
                logger.warning(
                    f'Population raster for "{pv}" not found under '
                    f'{os.path.join(project.data_folder, "population", pv)}. '
                    f'Run `rapida assess -c population -v {pv}` first. '
                    f'Skipping affected-population for "{pv}".'
                )
                continue
            affected = self._compute_affected_population(pop_raster, outage_4326, ntl_dir, pv)
            src_rasters.append(affected)
            vars_ops.append((f'{self.name}_{pv}_affected', 'sum'))

        # 5. Zonal stats against the project polygons; merge into stats.ntl if it already exists.
        dst_layer = f'stats.{self.component}'
        lnames = gpd.list_layers(project.geopackage_file_path).name.tolist()
        polygons_layer = dst_layer if dst_layer in lnames else project.polygons_layer_name
        gdf = zst(
            src_rasters=src_rasters, polygon_ds=project.geopackage_file_path,
            polygon_layer=polygons_layer, vars_ops=vars_ops, progress=progress,
        )

        frac_col = f'{self.name}_outage_fraction'
        if frac_col in gdf.columns:
            gdf[f'{self.name}_outage_pct'] = gdf[frac_col] * 100.0
            gdf.drop(columns=[frac_col], inplace=True)

        self._write_stats_layer(project, gdf, dst_layer)
        logger.info(f'Wrote NTL statistics to {project.geopackage_file_path}:{dst_layer}')

    def _extract_outage_mask(self, outage_tif, dst_4326):
        """Collapse every band whose description contains 'OUTAGE' into a single 0/1 GeoTIFF."""
        import numpy as np
        from osgeo import gdal

        ds = gdal.Open(outage_tif)
        try:
            gt = ds.GetGeoTransform()
            srs = ds.GetSpatialRef()
            xsize, ysize = ds.RasterXSize, ds.RasterYSize
            combined = None
            for i in range(1, ds.RasterCount + 1):
                band = ds.GetRasterBand(i)
                desc = band.GetDescription() or ''
                if 'OUTAGE' in desc.upper():
                    arr = band.ReadAsArray()
                    mask = np.nan_to_num(arr, nan=0.0) > 0
                    combined = mask if combined is None else (combined | mask)
        finally:
            ds = None

        if combined is None:
            raise click.UsageError(f'No OUTAGE band found in {outage_tif}')

        driver = gdal.GetDriverByName('GTiff')
        dst = driver.Create(dst_4326, xsize, ysize, 1, gdal.GDT_Byte)
        dst.SetGeoTransform(gt)
        if srs is not None:
            dst.SetSpatialRef(srs)
        dst.GetRasterBand(1).WriteArray(combined.astype('uint8'))
        dst.FlushCache()
        dst = None
        return dst_4326

    def _discover_population_vars(self, project, year):
        """
        Auto-discover population count-variables (operator 'sum') that already have a raster
        in this project. Ratio variables (e.g. dependency) are excluded because summing
        population x outage is only meaningful for additive counts.
        """
        discovered = []
        try:
            with Session() as session:
                pop_defs = session.get_component('population')
        except Exception as e:
            logger.debug(f'Could not read population component definitions: {e}')
            return discovered
        for name, var_data in pop_defs.items():
            if var_data.get('operator') != 'sum':
                continue
            if self._find_population_raster(project, name, year):
                discovered.append(name)
        return discovered

    def _find_population_raster(self, project, pop_var, year):
        """Locate the population raster already produced in the project for pop_var."""
        base = os.path.join(project.data_folder, 'population', pop_var)
        if not os.path.isdir(base):
            return None
        if year:
            candidate = os.path.join(base, f'{pop_var}_{year}.tif')
            if os.path.exists(candidate):
                return candidate
        matches = [
            p for p in sorted(glob.glob(os.path.join(base, f'{pop_var}_*.tif')))
            if re.fullmatch(rf'{re.escape(pop_var)}_\d+\.tif', os.path.basename(p))
        ]
        if matches:
            return matches[-1]
        fallback = os.path.join(base, f'{pop_var}.tif')
        return fallback if os.path.exists(fallback) else None

    def _compute_affected_population(self, pop_raster, outage_4326, ntl_dir, pop_var):
        """population x outage, aligned to the population grid, so a zonal sum yields affected people."""
        from osgeo import gdal
        from osgeo_utils.gdal_calc import Calc

        pds = gdal.Open(pop_raster)
        try:
            gt = pds.GetGeoTransform()
            width, height = pds.RasterXSize, pds.RasterYSize
            psrs = pds.GetSpatialRef()
        finally:
            pds = None

        minx, maxy = gt[0], gt[3]
        maxx = minx + gt[1] * width
        miny = maxy + gt[5] * height

        outage_aligned = os.path.join(ntl_dir, f'{self.name}_{pop_var}_outage_on_popgrid.tif')
        gdal.Warp(
            outage_aligned, outage_4326, format='GTiff', dstSRS=psrs,
            outputBounds=(minx, miny, maxx, maxy), width=width, height=height,
            resampleAlg='near', dstNodata=0,
        )

        affected = os.path.join(ntl_dir, f'{self.name}_{pop_var}_affected.tif')
        ds = Calc(
            calc='a*b', a=pop_raster, b=outage_aligned, outfile=affected,
            projectionCheck=True, format='GTiff', quiet=True, overwrite=True,
        )
        ds = None
        return affected

    def _write_stats_layer(self, project, gdf, dst_layer):
        """Overwrite (or create) the stats layer in the project geopackage from a GeoDataFrame."""
        import io
        from osgeo import gdal
        from pyogrio import write_dataframe

        with io.BytesIO() as bio:
            fpath = f'/vsimem/{dst_layer}.fgb'
            write_dataframe(df=gdf, path=bio, layer=dst_layer, driver='FlatGeobuf')
            gdal.FileFromMemBuffer(fpath, bio.getbuffer())
            bio.seek(0)
            with gdal.OpenEx(fpath) as src:
                options = gdal.VectorTranslateOptions(
                    format='GPKG', accessMode='overwrite', layerName=dst_layer, makeValid=True,
                )
                gdal.VectorTranslate(destNameOrDestDS=project.geopackage_file_path, srcDS=src, options=options)
            gdal.Unlink(fpath)

    def __call__(self, *args, **kwargs):
        return self.compute(**kwargs)
