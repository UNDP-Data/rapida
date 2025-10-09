import datetime
import json

from rapida.components.landuse.search_utils.tiles import Candidate
from rapida.components.landuse.search_utils.zones import _parse_mgrs_100k
from typing import List


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
        self._select_tiles_()

    def _select_tiles_(self):
        scand = sorted(self.s2_tiles, key=lambda c:-c.quality_score)
        for cand in scand:
            #print(cand.id, cand.quality_score, cand.cloud_cover, cand.data_coverage, datetime.datetime.fromtimestamp(cand.ref_ts).strftime('%d%m%Y'))
            print(f'{cand.id} quality score: {cand.quality_score} cloud-cover: {cand.cloud_cover} data-coverage?: {cand.data_coverage} ref_ts: {datetime.datetime.fromtimestamp(cand.ref_ts).strftime('%d-%m-%Y')}')
            # for k, v in cand.assets.items():
            #     print(k, json.dumps(v, indent=4))