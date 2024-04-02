import numpy as np

# Згенеруємо випадкові дані про мобільні телефони
np.random.seed(0)
num_samples = 1000

# Розмір екрану (від 4 до 7 дюймів)
screen_size = np.random.uniform(4, 7, size=num_samples)

# Камера (від 8 до 64 Мп)
camera = np.random.uniform(8, 64, size=num_samples)

# Об'єм пам'яті (від 32 до 512 Гб)
memory = np.random.choice([32, 64, 128, 256, 512], size=num_samples)

# Вартість (від 200 до 2000)
price = np.random.randint(200, 2001, size=num_samples)

# Перевірка перших 5 записів
print("Перші 5 записів:")
print("Розмір екрану | Камера (Мп) | Об'єм пам'яті | Вартість")
for i in range(5):
    print(f"{screen_size[i]:.1f} дюймів     | {camera[i]:.1f} Мп         | {memory[i]} Гб         | ${price[i]}")

# Нормалізація числових ознак
def normalize_feature(feature):
    return (feature - np.min(feature)) / (np.max(feature) - np.min(feature))

# Нормалізація ознак
screen_size = normalize_feature(screen_size)
camera = normalize_feature(camera)
memory = normalize_feature(memory)
price = normalize_feature(price)

# Перевірка перших 5 записів після нормалізації
print("\nПерші 5 записів після нормалізації:")
print("Розмір екрану | Камера (Мп) | Об'єм пам'яті | Вартість")
for i in range(5):
    print(f"{screen_size[i]:.3f}         | {camera[i]:.3f}         | {memory[i]}         | ${price[i]}")

# Використання попереднього класу нейронної мережі для класифікації мобільних телефонів
# та їх вартості

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

# Перевірка роботи моделі
input_size = 4  # кількість параметрів мобільного телефону (розмір екрану, камера, об'єм пам'яті, вартість)
hidden_size = 5  # кількість прихованих нейронів
output_size = 1  # кількість класів (вартість мобільного телефону)

# Навчання моделі
epochs = 1000
model = NeuralNetwork(input_size, hidden_size, output_size)
output = model.forward(np.array([screen_size, camera, memory, price]).T)
for epoch in range(epochs):
    # Пряме поширення та зворотне поширення для кожного навчального прикладу
    for i in range(len(screen_size)):
        X = np.array([screen_size[i], camera[i], memory[i], price[i]]).reshape(1, -1)  # вхідний приклад
        y = np.array([price[i]]).reshape(1, -1)  # очікуваний вихід
        output = model.forward(np.array([screen_size, camera, memory, price]).T)  # пряме поширення
        model.backward(np.array([screen_size, camera, memory, price]).T, y, output)  # зворотне поширення

    # Оцінка точності моделі після кожної епохи
    if (epoch + 1) % 100 == 0:
        predictions = model.forward(np.array([screen_size, camera, memory, price]).T)
        loss = np.mean(np.square(price - predictions))
        print(f"Епоха {epoch + 1}/{epochs}, Втрата: {loss:.4f}")

# Тестування моделі
test_input = np.array([[0.6, 0.8, 0.3, 0.5],   # розмір екрану: 6 дюймів, камера: 48 Мп, об'єм пам'яті: 128 Гб, вартість: $800
                       [0.3, 0.5, 0.6, 0.8],   # розмір екрану: 5 дюймів, камера: 32 Мп, об'єм пам'яті: 256 Гб, вартість: $1500
                       [0.8, 0.6, 0.4, 0.7]])  # розмір екрану: 7 дюймів, камера: 40 Мп, об'єм пам'яті: 64 Гб, вартість: $500
predictions = model.forward(test_input)
print("\nПрогнози для тестових даних:")
for i in range(len(test_input)):
    print(f"Розмір екрану: {test_input[i, 0] * 3} дюйми, Камера: {test_input[i, 1] * 56 + 8} Мп, Об'єм пам'яті: {int(test_input[i, 2] * 480) + 32} Гб, Вартість: ${int(test_input[i, 3] * 1800) + 200}")
