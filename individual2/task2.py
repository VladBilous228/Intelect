import numpy as np
import matplotlib.pyplot as plt

# Задаємо параметри функцій
a_tri = 2.1
b_tri = 4.5
c_tri = 7.8

a_trap = 2.3
b_trap = 3.1
c_trap = 5.3
d_trap = 7.9

c_gauss = 5
s_gauss = 0.25

# Функція для трикутної функції належності
def triangular(x, a, b, c):
    return np.maximum(0, np.minimum((x - a) / (b - a), (c - x) / (c - b)))

# Функція для трапецієподібної функції належності
def trapezoidal(x, a, b, c, d):
    return np.maximum(0, np.minimum(np.minimum((x - a) / (b - a), 1), (d - x) / (d - c)))

# Функція для гаусової функції належності
def gaussian(x, c, s):
    return np.exp(-0.5 * ((x - c) / s) ** 2)

# Задаємо діапазон для побудови графіків
x = np.linspace(2, 8, 1000)

# Обчислюємо значення функцій на вказаному діапазоні
y_tri = triangular(x, a_tri, b_tri, c_tri)
y_trap = trapezoidal(x, a_trap, b_trap, c_trap, d_trap)
y_gauss = gaussian(x, c_gauss, s_gauss)

# Побудова графіків
plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(x, y_tri, linewidth=1.5)
plt.title('Трикутна функція належності')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(x, y_trap, linewidth=1.5)
plt.title('Трапецієподібна функція належності')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(x, y_gauss, linewidth=1.5)
plt.title('Гаусова функція належності')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)

plt.tight_layout()
plt.show()
