import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.optimizers import Adam
from keras.layers import Reshape 
from keras.layers import Input




# Завантаження та підготовка даних
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1)  # Зміна форми зображень на (28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
x_train = x_train.astype('float32') / 255.0  # Нормалізація
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)  # Одноразове кодування міток класів
y_test = to_categorical(y_test, 10)

# Побудова моделі нейронної мережі
model = Sequential([
    Input(shape=(28, 28)),
    Reshape((28, 28, 1)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Компіляція моделі
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Навчання моделі
history = model.fit(x_train, y_train, epochs=3, batch_size=32, validation_split=0.2)

# Виведення структури моделі на консоль
model.summary()

# Результати точності навчання
print("Точність навчання:", history.history['accuracy'])

# Розпізнавання тестового зображення
index = 0
predicted = model.predict(np.array([x_test[index]]))
print("Розпізнана цифра:", np.argmax(predicted))
plt.imshow(x_test[index].reshape(28, 28), cmap='gray')
plt.show()

# Пошук неправильно розпізнаних зразків
incorrect_indices = np.nonzero(np.argmax(model.predict(x_test), axis=1) != np.argmax(y_test, axis=1))[0]
print("Кількість неправильно розпізнаних зразків:", len(incorrect_indices))
# Виведення 5 неправильно розпізнаних зразків
for i in range(5):
    plt.imshow(x_test[incorrect_indices[i]].reshape(28, 28), cmap='gray')
    plt.show()


# Додана частина коду для завдання 3

# Визначення кількості зразків
num_samples = 30

# Вибір підмножини тренувальних даних
x_train_subset = x_train[:num_samples]
y_train_subset = y_train[:num_samples]





