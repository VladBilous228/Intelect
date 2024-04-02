import numpy as np
from sklearn.model_selection import train_test_split

# Дані про харчову продукцію
# Приклад: текстури та кольори

# Розмір датасету
num_samples = 1000

# Дані про текстуру (приклад)
texture = np.random.randint(0, 3, size=num_samples)  # Припустимо, у нас є 3 типи текстур

# Дані про кольори (приклад)
color = np.random.randint(0, 3, size=num_samples)  # Припустимо, у нас є 3 типи кольорів

# Визначення типу харчової продукції (клас)
# Використовуємо випадкове значення з трьох можливих: овочі, фрукти, м'ясо
food_type = np.random.randint(0, 3, size=num_samples)

# Перевірка перших 5 записів
print("Перші 5 записів:")
print("Текстура | Колір | Тип продукту")
for i in range(5):
    print(f"{texture[i]}         | {color[i]}      | {food_type[i]}")

# Нормалізація числових ознак (необхідно тільки для текстур та кольорів)
def normalize_feature(feature):
    return (feature - np.min(feature)) / (np.max(feature) - np.min(feature))

texture = normalize_feature(texture)
color = normalize_feature(color)

# Перевірка перших 5 записів після нормалізації
print("\nПерші 5 записів після нормалізації:")
print("Текстура | Колір | Тип продукту")
for i in range(5):
    print(f"{texture[i]}         | {color[i]}      | {food_type[i]}")

# Нейронна мережа для сегментації харчової продукції

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Ініціалізація ваг та зміщень
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))

    # Сигмоїдна функція активації
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Пряме поширення
    def forward(self, X):
        self.hidden_sum = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.activated_hidden = self.sigmoid(self.hidden_sum)

        self.output_sum = np.dot(self.activated_hidden, self.weights_hidden_output) + self.bias_output
        self.activated_output = self.sigmoid(self.output_sum)

        return self.activated_output

    # Зворотне поширення
    def backward(self, X, y, output, learning_rate=0.01):
        self.output_error = y - output
        self.output_delta = self.output_error * (output * (1 - output))

        self.hidden_error = np.dot(self.output_delta, self.weights_hidden_output.T)
        self.hidden_delta = self.hidden_error * (self.activated_hidden * (1 - self.activated_hidden))

        self.weights_input_hidden += learning_rate * np.dot(X.T, self.hidden_delta)
        self.weights_hidden_output += learning_rate * np.dot(self.activated_hidden.T, self.output_delta)

# Розбиття даних на тренувальний та тестовий набори
X = np.concatenate((texture.reshape(-1, 1), color.reshape(-1, 1)), axis=1)
y = food_type
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Побудова та навчання моделі
input_size = 2  # кількість ознак (текстура, колір)
hidden_size = 5  # кількість прихованих нейронів
output_size = 3  # кількість класів харчової продукції

model = NeuralNetwork(input_size, hidden_size, output_size)
epochs = 1000

for epoch in range(epochs):
    # Пряме поширення та зворотне поширення для кожного навчального прикладу
    for i in range(len(X_train)):
        X = X_train[i].reshape(1, -1)  # вхідний приклад
        y = np.zeros((1, output_size))  # очікуваний вихід
        y[0, y_train[i]] = 1
        output = model.forward(X_train)  # пряме поширення
        model.backward(X_train, y, output)  # зворотне поширення

    # Оцінка точності моделі після кожної епохи
    if (epoch + 1) % 100 == 0:
        predictions = np.argmax(model.forward(X_train), axis=1)
        accuracy = np.mean(predictions == y_train)
        print(f"Епоха {epoch + 1}/{epochs}, Точність: {accuracy:.4f}")

# Тестування моделі
predictions_test = np.argmax(model.forward(X_test), axis=1)
accuracy_test = np.mean(predictions_test == y_test)
print(f"Точність на тестовому наборі: {accuracy_test:.4f}")

# Обчислення середньоквадратичної функції втрат
loss = np.mean(np.square(y_test - predictions_test))
print(f"Середньоквадратична функція втрат: {loss:.4f}")

# Виведення вагових коефіцієнтів
print("\nВагові коефіцієнти:")
print("Ваги від входу до прихованого шару:")
print(model.weights_input_hidden)
print("\nВаги від прихованого до вихідного шару:")
print(model.weights_hidden_output)
