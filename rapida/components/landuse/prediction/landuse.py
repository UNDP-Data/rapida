import os
import json
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pystac
from rich.progress import Progress
from rapida.components.landuse.constants import DYNAMIC_WORLD_COLORMAP
from rapida.components.landuse.prediction.base import PredictionBase
from rapida.util.setup_logger import setup_logger


logger = setup_logger('rapida')

try:
    import tensorflow as tf
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception as ie:
    tf = None


class LandusePrediction(PredictionBase):

    required_bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B11', 'B12']
    def __init__(self,item: pystac.Item, tile_size: int = 512, tile_buffer: int = 64):
        super().__init__(item,
                         component_name="landuse",
                         bands=self.required_bands,
                         output_nodata_value=255,
                         input_nodata_value=0,
                         tile_size=tile_size,
                         tile_buffer=tile_buffer)

    def _normalize(self, image):
        """
        Normalize the image prior to run land use prediction model.
        """
        norm_percentiles = np.array([
            [1.7417268007636313, 2.023298706048351],
            [1.7261204997060209, 2.038905204308012],
            [1.6798346251414997, 2.179592821212937],
            [1.7734969472909623, 2.2890068333026603],
            [2.289154079164943, 2.6171674549378166],
            [2.382939712192371, 2.773418590375327],
            [2.3828939530384052, 2.7578332604178284],
            [2.1952484264967844, 2.789092484314204],
            [1.554812948247501, 2.4140534947492487]
        ])

        image = np.log(image * 0.005 + 1)
        image = (image - norm_percentiles[:, 0]) / norm_percentiles[:, 1]
        image = np.exp(image * 5 - 1)
        image = image / (image + 1)
        return image

    def _load_model(self):
        """
        Load the model from the saved path
        This is the model from Dynamic World
        """
        folder = os.path.dirname(__file__)

        model_path = os.path.join(folder, 'model', 'forward')

        # model_path = "./model/forward"
        model = tf.saved_model.load(model_path)
        return model

    def run_model(self, data: np.ndarray) -> np.ndarray:
        """
        Run land use prediction model.
        :param data: Input data array, typically preprocessed and harmonized, with shape [height, width, bands].
        :return: Output prediction array, typically a 2D array [height, width] or a probability map, depending on the model's purpose.
        """
        normalized_data = self._normalize(data)
        nhwc_image = tf.expand_dims(tf.cast(normalized_data, dtype=tf.float32), axis=0)
        logger.debug("Running model inference")
        forward_model = self._load_model()
        lulc_logits = forward_model(nhwc_image)
        lulc_prob = tf.nn.softmax(lulc_logits)
        lulc_prob = np.array(lulc_prob[0])
        lulc_prediction = np.argmax(lulc_prob, axis=-1)
        return lulc_prediction

    def _hex_to_rgb(self, hex_color: str):
        """
        Convert hex color to rgb color.
        """
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

    def write_metadata(self, dst):
        """
        Write additional metadata to the output raster file including:

        - colormap information of dynamic world colors
        - STATISTICS_UNIQUE_VALUES variable for GeoHub

        :param dst: An open rasterio dataset writer object, representing the output file to which metadata can be added.
        :return: The same dataset writer object, optionally updated with additional metadata.
        """
        landuse_colormap = {k: self._hex_to_rgb(v['color']) for k, v in DYNAMIC_WORLD_COLORMAP.items()}
        statistics_unique_values = {str(k): v['label'] for k, v in DYNAMIC_WORLD_COLORMAP.items()}
        unique_values_json = json.dumps(statistics_unique_values, ensure_ascii=False)

        # write colormap
        dst.write_colormap(1, landuse_colormap)
        # write STATISTICS_UNIQUE_VALUES for GeoHub
        dst.update_tags(1, STATISTICS_UNIQUE_VALUES=unique_values_json)

        statistics_unique_values = {
            str(k): v['label'] for k, v in DYNAMIC_WORLD_COLORMAP.items()
        }
        unique_values_json = json.dumps(statistics_unique_values, ensure_ascii=False)
        dst.update_tags(1, STATISTICS_UNIQUE_VALUES=unique_values_json)

        return dst


if __name__ == "__main__":
    bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B11', 'B12'],
    img_paths = ['/data/sentinel_tests/S2A_35MRT_20240819_0_L1C/B02.jp2', '/data/sentinel_tests/S2A_35MRT_20240819_0_L1C/B03.jp2',
                 '/data/sentinel_tests/S2A_35MRT_20240819_0_L1C/B04.jp2', '/data/sentinel_tests/S2A_35MRT_20240819_0_L1C/B05.jp2',
                 '/data/sentinel_tests/S2A_35MRT_20240819_0_L1C/B06.jp2', '/data/sentinel_tests/S2A_35MRT_20240819_0_L1C/B07.jp2',
                 '/data/sentinel_tests/S2A_35MRT_20240819_0_L1C/B08.jp2', '/data/sentinel_tests/S2A_35MRT_20240819_0_L1C/B11.jp2',
                 '/data/sentinel_tests/S2A_35MRT_20240819_0_L1C/B12.jp2']
    output_file_path = "/data/sentinel_tests/S2A_35MRT_20240819_0_L1C/landuse.tif"
    item_url = '/data/sentinel_tests/S2A_35MRT_20240819_0_L1C/item.json'
    mask_file = "/data/kigali_small/data/kigali_small.gpkg"
    mask_layer = "polygons"

    item = pystac.Item.from_file(item_url)

    with Progress() as progress:
        t1 = time.time()
        prediction = LandusePrediction(item=item)

        prediction.predict(img_paths=img_paths,
                           output_file_path=output_file_path,
                           progress=progress,
                           mask_file=mask_file,
                           mask_layer=mask_layer,)
        t2 = time.time()

        logger.info(f"prediction time: {t2 - t1}")