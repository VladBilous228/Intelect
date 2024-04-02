from numpy import exp, array, random, dot

from numpy import *
import numpy as np


def calcOutput(inputs, weight):
    return 1 / (1 + exp(-(dot(inputs, weight))))


def calcWeight(inputs, outputs, output):
    return dot(inputs.T, (outputs - output) * output * (1 - output))


def find_answer(arr):
    training_set_inputs = array([[0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 1]])
    training_set_outputs = array([[1, 1, 1, 1, 1, 1]]).T

    random.seed(1)

    weights = 2 * random.random((3, 1)) - 1

    for iteration in range(100):
        output = 1 / (1 + exp(-(dot(training_set_inputs, weights))))
        weights += dot(training_set_inputs.T, (training_set_outputs - output) * output * (1 - output))

    print(1 / (1 + exp(-dot(arr, weights))))


def find_OR_answer():
    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    random.seed(1)

    synaptic_weight = 2 * random.random((3, 1)) - 1

    for iteration in range(10000):
        output = calcOutput(training_set_inputs, synaptic_weight)
        synaptic_weight += calcWeight(training_set_inputs, training_set_outputs, output)

    print("Логічна функція АБО число 6 в двійковому форматі [1, 1, 0] ->  ?: ")
    print(1 / (1 + exp(-dot(array([0, 0, 1]), synaptic_weight))))

# Функція активації (ступінчаста функція)
def activation_function(x):
    return 1 if x >= 0 else 0

# Функція персептрона
def perceptron(X, w, b):
    return activation_function(np.dot(X, w) + b)

# Вхідні дані (X1, X2, X3) та вагові коефіцієнти
X = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
               [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
w = np.array([1, 1, 1])  # Вагові коефіцієнти
b = -1  # Зсув

# Вивід результатів для кожної комбінації
for i in range(len(X)):
    result = perceptron(X[i], w, b)
    print(f"Вхід: {X[i]}, Вихід: {result}")

def test1():
    arr = array([1, 1, 0])
    arr1 = array([0, 0, 1])
    arr2 = array([0, 1, 1])
    arr3 = array([0, 0, 0])
    arr4 = array([0, 1, 0])

    find_answer(arr)
    find_answer(arr1)
    find_answer(arr2)
    find_answer(arr3)
    find_answer(arr4)

find_OR_answer()

