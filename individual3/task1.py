import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.io import savemat

# 1. Подготовка данных для обучения сети нейронной связи
X_train = np.array([[0.1], [0.3], [0.5]])  # Значения X для обучения
Y_train = np.array([[0.2], [0.4], [0.6]])  # Приближенные значения Y

# 2. Создание нейронной сети с помощью библиотеки TensorFlow
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])

# 3. Обучение сети на подготовленных данных
model.compile(optimizer='sgd', loss='mean_squared_error')
model.fit(X_train, Y_train, epochs=100)

# 4. Тестирование обученной сети
X_test = np.array([[0.2], [0.4], [0.7], [0.9]])  # Тестовые значения X
Y_pred = model.predict(X_test)  # Предсказания соответствующих значений Y

print("Предсказания:", Y_pred)
plt.plot(X_test, Y_pred)
plt.show()

# 5. Сохранение обученной модели в формате .mat
weights = {'weights': model.get_weights()}
savemat('trained_model.mat', weights)