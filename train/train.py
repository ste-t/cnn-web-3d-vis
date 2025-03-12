#!/usr/bin/env python3

import keras
import tensorflowjs as tfjs

import matplotlib.pyplot as plt

"""
    Train tensor:
    (60000, 28, 28, 1)

    Test tensor:
    (10000, 28, 28, 1)
"""


def main():
    # Train and test data
    (train_images, train_labels), (test_images, test_labels) \
        = keras.datasets.mnist.load_data()

    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Architecture
    l = keras.layers
    model = keras.models.Sequential([

        l.Conv2D(14, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        l.MaxPooling2D((2, 2)),
        l.Conv2D(28, (3, 3), activation="relu"),

        l.Flatten(),
        l.Dropout(0.2),

        l.Dense(70, activation="relu"),
        # l.Dropout(0.2),
        l.Dense(10, activation="softmax")

    ])

    model.summary()

    model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"]
    )

    # Train
    hist = model.fit(train_images, train_labels, epochs=4,
                     validation_data=(test_images, test_labels))

    # Save as python keras and javascript model
    model.save('out/model.keras')
    tfjs.converters.save_keras_model(model, "out/tfjs/")

    # Plot training steps
    plt.plot(hist.history["accuracy"], label="accuracy")
    plt.plot(hist.history["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim([0.5, 1])
    plt.legend(loc="lower right")

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose="2")
    print(test_acc, test_loss)

    plt.show()


if __name__ == "__main__":
    main()
