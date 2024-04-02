import keras.utils
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

model = keras.Sequential([
    Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2), strides=2),
    Conv2D(64, (3, 3), padding="same", activation="relu"),
    MaxPooling2D((2, 2), strides=2),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(10, activation="softmax")
])

x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)
print(x_train.shape)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

his = model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2)

model.evaluate(x_test, y_test_cat)

print(model.summary())


def testImage():
    index = np.random.randint(0, len(x_test))
    test_image = x_test[index]
    test_label = y_test[index]

    prediction = model.predict(np.expand_dims(test_image, axis=0))

    plt.imshow(test_image.squeeze(), cmap="gray")
    plt.title(f"Actual: {test_label}, Predicted: {np.argmax(prediction)}, Number: {index}")
    plt.axis('off')
    plt.show()


while(True):
    index = np.random.randint(0, len(x_test))
    test_image = x_test[index]
    test_label = y_test[index]

    prediction = model.predict(np.expand_dims(test_image, axis=0))

    if test_label == np.argmax(prediction):
        continue

    plt.show()
    plt.imshow(test_image.squeeze())
    plt.title(f"Actual: {test_label}, Predicted: {np.argmax(prediction)}, Number: {index}")
    plt.axis('off')
