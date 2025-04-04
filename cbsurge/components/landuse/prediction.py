import logging
import os
from concurrent.futures import ProcessPoolExecutor
from typing import List

import numpy as np
import rasterio
from rasterio.windows import Window
import tensorflow as tf


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
tf.get_logger().setLevel('ERROR')

def normalize(image):
    """
    :param image:
    :return:
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





def load_model():
    """
    Load the model from the saved path
    This is the model from Dynamic World
    """
    folder = os.path.dirname(__file__)

    model_path = os.path.join(folder, 'model', 'forward')

    # model_path = "./model/forward"
    model = tf.saved_model.load(model_path)
    return model

def create_multi_band_image(input_files, output_file):
    """
    Create a multi-band image from the input files
    :param input_files:
    :param output_file:
    :return:
    """
    raw_data = np.empty(shape=(len(input_files),), dtype="u2")
    with rasterio.open(input_files[0]) as src:
        meta = src.meta.copy()
        meta.update(count=len(input_files))

    with rasterio.open(output_file, "w", **meta) as dst:
        for i, file in enumerate(input_files):
            with rasterio.open(file) as src:
                raw_data[:, :, i] = src.read(1)
    print(f"Multi-band image saved as {output_file}")
    return output_file

def preinference_check(img_paths: List):
    assert len(img_paths) > 0 and len(img_paths) == 9, "9 bands are required for prediction with this model"
    col_sizes = []
    row_sizes = []
    for img_path in img_paths:
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"File {img_path} does not exist")
        with rasterio.open(img_path) as src:
            if src.count != 1:
                raise ValueError("Each of the image must have one band")
            col_sizes.append(src.width)
            row_sizes.append(src.height)
    if len(set(col_sizes)) != 1:
        raise ValueError("All images must have the same dimensions")
    if len(set(row_sizes)) != 1:
        raise ValueError("All images must have the same dimensions")
    return col_sizes[0], row_sizes[0]


def process_tile(row, col, img_paths, buffer, max_window_size, row_size, col_size):
    logging.info(f"Processing window at row {row}, col {col}")
    window_width = min(256 + buffer * 2, max_window_size, col_size - col)
    window_height = min(256 + buffer * 2, max_window_size, row_size - row)

    window = Window(col, row, window_width, window_height)
    raw_data = np.empty((window_height, window_width, 9), dtype="u2")

    for i, file_path in enumerate(img_paths):
        with rasterio.open(file_path) as src:
            raw_data[:, :, i] = src.read(1, window=window)

    normalized_data = normalize(raw_data)
    nhwc_image = tf.expand_dims(tf.cast(normalized_data, dtype=tf.float32), axis=0)
    logging.info("Running model inference")
    forward_model = load_model()
    lulc_logits = forward_model(nhwc_image)
    lulc_prob = tf.nn.softmax(lulc_logits)
    lulc_prob = np.array(lulc_prob[0])
    lulc_prediction = np.argmax(lulc_prob, axis=-1)

    start_row_in_prediction = buffer if row > 0 else 0
    end_row_in_prediction = window_height - buffer if row + 256 < row_size else window_height

    start_col_in_prediction = buffer if col > 0 else 0
    end_col_in_prediction = window_width - buffer if col + 256 < col_size else window_width

    original_tile = lulc_prediction[start_row_in_prediction:end_row_in_prediction,
                    start_col_in_prediction:end_col_in_prediction]
    return (row, col, original_tile)


def predict(img_paths: List[str], output_file_path: str):
    logging.info("Starting land use prediction")
    col_size, row_size = preinference_check(img_paths)
    buffer = 64
    max_window_size = 512

    logging.info("Opening first image to get metadata")
    with rasterio.open(img_paths[0]) as src:
        crs = src.crs
        dst_transform = src.transform

    logging.info(f"Creating output file: {output_file_path}")
    with rasterio.open(output_file_path, mode='w', driver="GTiff", dtype='uint8',
                       width=col_size, height=row_size, crs=crs, transform=dst_transform, count=1) as dst:
        tasks = []
        with ProcessPoolExecutor() as executor:
            for row in range(0, row_size, 256):
                for col in range(0, col_size, 256):
                    tasks.append(
                        executor.submit(process_tile, row, col, img_paths, buffer, max_window_size, row_size, col_size))

            for future in tasks:
                row, col, original_tile = future.result()
                logging.info(f"Writing prediction tile at row {row}, col {col}")
                dst.write(original_tile.astype(np.uint8), 1,
                          window=Window(col, row, min(256, col_size - col), min(256, row_size - row)))
    logging.info("Prediction process completed successfully")
    return output_file_path



if __name__ == "__main__":
    pass