from pathlib import Path

import skimage

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 50)

DATA_PATH = "datasets/"
ICMC_PATH = "PessoasICMC/"
ORIGNAL_PATH = "OrlFaces20/"


def read_data(name, size=None):
    if size is None:
        print("The amount of images and their sizes must be specified")
        raise ValueError("size can not be None")

    data = np.ndarray(size)
    path = Path(name)

    for subdir in path.iterdir():
        class_num = int(subdir.name[1:]) - 1  # skip the "p/s" at the beggining
        for idx, img_file in enumerate(subdir.iterdir()):
            data[class_num, idx] = skimage.io.imread(img_file)

    # Convert to numpy array
    data = np.array(data)
    amount_data = size[0]*size[1]
    data = data.reshape(amount_data, size[-2], size[-1])

    # Create the target vector
    targets = np.array([[x]*size[1] for x in range(1, 21)])
    targets = targets.reshape(-1)

    return data, targets


if __name__ == "__main__":
    icmc_x, icmc_y = read_data(DATA_PATH + ICMC_PATH, size=(20, 1, 300, 200))
    original_x, original_y = read_data(DATA_PATH + ORIGNAL_PATH,
                                size=(20, 10, 112,  92))
