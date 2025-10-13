import datetime
import json

import rasterio
from affine import Affine
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.transform import array_bounds
from rasterio.windows import from_bounds
from shapely import unary_union, box
from shapely.geometry import shape, mapping

from rapida.components.landuse.search_utils.search import fetch_s2_tiles
from rapida.components.landuse.search_utils.tiles import Candidate, download_item, s3_to_https
from rapida.components.landuse.search_utils.zones import _parse_mgrs_100k, utm_bounds
from typing import List
import geopandas as gpd
import os


class Sentinel2Item:
    """
    A sophisticated Sentinel 2 utility image that aligns perfectly with MGRS 100K grid tiles.
    While Sentinel 2 imagery is organized into tiles, these only followS roughly the UTM MGRS 100K grid.
    Additionally, Sentinel 2 images can come incomplete in a variety of data coverages. For this reason the
    Sentinel2Item class takes several S2 images and combines spatially disjoint tiles into one ideal MGRS 100K tile
    perfectly aligned with 100K MGRS grid
    """

    def __init__(self, mgrs_grid:str=None, s2_tiles:List[Candidate] = None):
        self.zone, self.band, self.letters = _parse_mgrs_100k(grid_id=mgrs_grid)
        self.mgrs_grid = mgrs_grid
        self.s2_tiles = s2_tiles
        self.mgrs_polygon, self.mgrs_poly_crs = utm_bounds(self.mgrs_grid)
        self._select_tiles_()

    def _select_tiles_(self):
        scand = sorted(self.s2_tiles, key=lambda c: -c.quality_score)
        mgrs_geom = self.mgrs_polygon

        best_cand = scand[0]
        best_geom = shape(best_cand.tile_data_geometry).intersection(mgrs_geom)
        href = s3_to_https(best_cand.assets['blue']['href'])
        mgrs_bounds = mgrs_geom.bounds

        with rasterio.open(href) as ref_src:
            res_x = ref_src.transform.a
            res_y = -ref_src.transform.e
            out_meta = ref_src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": int((mgrs_bounds[3] - mgrs_bounds[1]) / res_y),
                "width": int((mgrs_bounds[2] - mgrs_bounds[0]) / res_x),
                "transform": Affine(
                    res_x, 0, mgrs_bounds[0],
                    0, -res_y, mgrs_bounds[3]
                ),
                "count": 1,
                "dtype": ref_src.dtypes[0],
                "crs": ref_src.crs
            })

        with rasterio.open("merged_blue_band.tif", "w", **out_meta) as dest:
            with rasterio.open(href) as src:
                out_img, out_transform = mask(
                    src, [mapping(best_geom)], crop=True
                )
                masked_bounds = array_bounds(
                    out_img.shape[1], out_img.shape[2], out_transform
                )
                window = from_bounds(
                    *masked_bounds, transform=dest.transform, height=dest.height, width=dest.width
                )
                dest.write(out_img[0], 1, window=window)

            current_geom = best_geom

            for cand in scand[1:]:
                cand_geom = cand.tile_data_geometry.intersection(mgrs_geom)

                if cand_geom.is_empty:
                    continue
                gdf = gpd.GeoDataFrame(geometry=[cand_geom])
                gdf.set_crs(self.mgrs_poly_crs, inplace=True)
                gdf.to_file(f'/tmp/{cand.id}_cand.fgb', driver='FlatGeobuf')

                geom_diff = cand_geom.difference(current_geom)
                if geom_diff.is_empty:
                    continue


                gdf = gpd.GeoDataFrame(geometry=[geom_diff])
                gdf.set_crs(self.mgrs_poly_crs, inplace=True)
                gdf.to_file(f'/tmp/{cand.id}_geom_diff.fgb', driver='FlatGeobuf')

                href = s3_to_https(cand.assets['blue']['href'])
                with rasterio.open(href) as src:
                    cand_img, cand_transform = mask(
                        src, [mapping(geom_diff)], crop=True
                    )

                    cand_bounds = array_bounds(
                        cand_img.shape[1], cand_img.shape[2], cand_transform
                    )
                    # cand_bounds = (
                    #     max(cand_bounds[0], mgrs_bounds[0]),
                    #     max(cand_bounds[1], mgrs_bounds[1]),
                    #     min(cand_bounds[2], mgrs_bounds[2]),
                    #     min(cand_bounds[3], mgrs_bounds[3])
                    # )

                    win = from_bounds(*cand_bounds, transform=dest.transform)
                    dest.write(cand_img[0], 1, window=win)

                current_geom = current_geom.union(cand_geom)