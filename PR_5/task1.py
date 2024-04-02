import numpy as np

# Логічний вираз X1 AND (X2 AND X3) OR X4
def logical_expression(x1, x2, x3, x4):
    return x1 and (x2 and x3) or x4

# Створення таблиці істинності
input_combinations = np.array([[x1, x2, x3, x4] for x1 in [0, 1] for x2 in [0, 1] for x3 in [0, 1] for x4 in [0, 1]])
output_values = np.array([logical_expression(*x) for x in input_combinations])

# Створення нейронної мережі
class NeuralNetwork:
    def __init__(self):
        self.input_size = 4
        self.hidden_size = 2
        self.output_size = 1

        # Ініціалізація ваг та зміщень
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))

    # Сигмоїдна функція активації
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Похідна сигмоїдної функції
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # Пряме поширення
    def forward(self, X):
        self.hidden_sum = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.activated_hidden = self.sigmoid(self.hidden_sum)

        self.output_sum = np.dot(self.activated_hidden, self.weights_hidden_output) + self.bias_output
        self.activated_output = self.sigmoid(self.output_sum)

        return self.activated_output

    # Зворотне поширення
    def backward(self, X, y, output):
        self.output_error = y - output
        self.output_delta = self.output_error * self.sigmoid_derivative(output)

        self.hidden_error = np.dot(self.output_delta, self.weights_hidden_output.T)
        self.hidden_delta = self.hidden_error * self.sigmoid_derivative(self.activated_hidden)

        self.weights_input_hidden += np.dot(X.T, self.hidden_delta)
        self.weights_hidden_output += np.dot(self.activated_hidden.T, self.output_delta)

    # Навчання
    def train(self, X, y, epochs):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)

# Навчання нейронної мережі
nn = NeuralNetwork()
nn.train(input_combinations, output_values.reshape(-1, 1), epochs=1000)

# Тестування нейронної мережі
test_input = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [1, 0, 1, 0]])
predictions = nn.forward(test_input)
print("\nTest Predictions:")
for i in range(len(test_input)):
    print(f"Input: {test_input[i]}, Output: {predictions[i]}")

# Аналіз результатів
def accuracy(predictions, targets):
    predictions = np.round(predictions)
    correct = np.sum(predictions == targets)
    total = len(targets)
    return correct / total

print("\nModel Performance:")
print(f"Accuracy: {accuracy(predictions, output_values)}")

