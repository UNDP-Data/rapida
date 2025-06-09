import time
import numpy as np
import pystac
from s2cloudless import S2PixelCloudDetector, download_bands_and_valid_data_mask
from rich.progress import Progress
from rapida.components.landuse.prediction.base import PredictionBase
from rapida.util.setup_logger import setup_logger


logger = setup_logger('rapida')


class CloudDetection(PredictionBase):


    def __init__(self,item: pystac.Item, tile_size: int = 512, tile_buffer: int = 64):
        super().__init__(item,
                         component_name="cloud_mask",
                         bands=["B01", "B02", "B04", "B05", "B08", "B8A", "B09", "B10", "B11", "B12"],
                         output_nodata_value=255,
                         tile_size=tile_size,
                         tile_buffer=tile_buffer)

    def run_model(self, data: np.ndarray) -> np.ndarray:
        """
        Run s2cloudless cloud detector model.
        :param data: Input data array, typically preprocessed and harmonized, with shape [height, width, bands].
        :return: Output prediction array, typically a 2D array [height, width] or a probability map, depending on the model's purpose.
        """
        # convert DI number to reflectance
        # see https://github.com/sentinel-hub/sentinel2-cloud-detector?tab=readme-ov-file#input-sentinel-2-scenes
        reflectance_conversion_factor = self.item.properties.get("s2:reflectance_conversion_factor", 1.0)
        reflectance_factor = 10000
        harmonized_data = (data.astype(np.float32) * reflectance_conversion_factor) / reflectance_factor
        harmonized_data = np.clip(harmonized_data, 0, None)

        cloud_detector = S2PixelCloudDetector(threshold=0.4, average_over=8, dilation_size=2, all_bands=False)
        cloud_prob = cloud_detector.get_cloud_probability_maps(harmonized_data[np.newaxis, ...])
        logger.debug(f"Cloud detection probability: {cloud_prob}")
        cloud_mask = cloud_detector.get_mask_from_prob(cloud_prob)
        return cloud_mask[0]


if __name__ == "__main__":
    img_paths = ['/data/sentinel_tests/S2A_35MRT_20240819_0_L1C/B01.jp2', '/data/sentinel_tests/S2A_35MRT_20240819_0_L1C/B02.jp2',
                 '/data/sentinel_tests/S2A_35MRT_20240819_0_L1C/B04.jp2', '/data/sentinel_tests/S2A_35MRT_20240819_0_L1C/B05.jp2',
                 '/data/sentinel_tests/S2A_35MRT_20240819_0_L1C/B08.jp2', '/data/sentinel_tests/S2A_35MRT_20240819_0_L1C/B8A.jp2',
                 '/data/sentinel_tests/S2A_35MRT_20240819_0_L1C/B09.jp2', '/data/sentinel_tests/S2A_35MRT_20240819_0_L1C/B10.jp2',
                 '/data/sentinel_tests/S2A_35MRT_20240819_0_L1C/B11.jp2', '/data/sentinel_tests/S2A_35MRT_20240819_0_L1C/B12.jp2']
    output_file_path = "/data/sentinel_tests/S2A_35MRT_20240819_0_L1C/cloud.tif"
    item_url = '/data/sentinel_tests/S2A_35MRT_20240819_0_L1C/item.json'
    mask_file = "/data/kigali_small/data/kigali_small.gpkg"
    mask_layer = "polygons"

    item = pystac.Item.from_file(item_url)

    with Progress() as progress:
        t1 = time.time()
        cloud_detection = CloudDetection(item=item)

        cloud_detection.predict(img_paths=img_paths,
                                output_file_path=output_file_path,
                                progress=progress,
                                mask_file=mask_file,
                                mask_layer=mask_layer,)
        t2 = time.time()

        logger.info(f"prediction time: {t2 - t1}")


