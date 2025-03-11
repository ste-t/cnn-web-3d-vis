#!/usr/bin/env python3

import tensorflow as tf
import keras

import pathlib
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt


def main():
    model: keras.Model = \
        keras.models.load_model("out/model.keras")  # type: ignore
    model.summary()

    for img_path in pathlib.Path("test/").glob("*.png"):

        img = Image.open(img_path)
        img_tensor = np.array(img).reshape(1, 28, 28, 1) / 255.0

        pred = model.predict(img_tensor)[0]
        pred = int(tf.math.argmax(pred))

        print(
            f"\nPrediction: {pred}\n"
        )

        plt.imshow(img)
        plt.show()

    # (test_images, test_labels) = keras.datasets.mnist.load_data()[1]
    # test_images = test_images / 255.0

    # test_loss, test_acc = model.evaluate(test_images, test_labels, verbose="2")
    # print(test_acc, test_loss)


if __name__ == "__main__":
    main()
