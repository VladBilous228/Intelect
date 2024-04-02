import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Задані коефіцієнти полінома
coefficients = {'a': 0, 'b': 1, 'c': 0, 'd': 2, 'e': 0, 'f': -3}

# Функція для генерації даних на інтервалі [-1, 1]
def generate_data(n_samples):
    x1 = np.random.uniform(-1, 1, n_samples)
    x2 = np.random.uniform(-1, 1, n_samples)
    y = (coefficients['a'] * x1 ** 2) + (coefficients['b'] * x2 ** 2) + (coefficients['c'] * x1 * x2) + (coefficients['d'] * x1) + (coefficients['e'] * x2) + coefficients['f']
    return x1, x2, y

# Побудова нейронної мережі
class PolynomialModel(nn.Module):
    def __init__(self):
        super(PolynomialModel, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Параметри навчання
lr = 0.01
n_epochs = 1000
batch_size = 32
n_samples = 1000

# Генерація даних
x1, x2, y = generate_data(n_samples)
X = np.column_stack((x1, x2))
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# Побудова моделі та втрат
model = PolynomialModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Навчання моделі
losses = []
for epoch in range(n_epochs):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

# Побудова графіка помилки мережі
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()


# Збереження навченої моделі
torch.save(model.state_dict(), 'polynomial_model.pth')


# Генерація даних для другого завдання
def generate_data_nonlinear(n_samples, t_range=(0, 10)):
    t = np.random.uniform(t_range[0], t_range[1], n_samples)
    f_t = np.cos(1.4 ** t)
    return t, f_t

# Побудова навчальних даних для другого завдання
n_samples_nonlinear = 1000
t, f_t = generate_data_nonlinear(n_samples_nonlinear)
t_tensor = torch.tensor(t, dtype=torch.float32).reshape(-1, 1)
f_t_tensor = torch.tensor(f_t, dtype=torch.float32).reshape(-1, 1)

# Побудова нейронної мережі для другого завдання
class NonlinearModel(nn.Module):
    def __init__(self):
        super(NonlinearModel, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Параметри навчання для другого завдання
lr_nonlinear = 0.01
n_epochs_nonlinear = 1000

# Побудова та навчання моделі для другого завдання
model_nonlinear = NonlinearModel()
criterion_nonlinear = nn.MSELoss()
optimizer_nonlinear = optim.Adam(model_nonlinear.parameters(), lr=lr_nonlinear)

losses_nonlinear = []
for epoch in range(n_epochs_nonlinear):
    optimizer_nonlinear.zero_grad()
    outputs_nonlinear = model_nonlinear(t_tensor)
    loss_nonlinear = criterion_nonlinear(outputs_nonlinear, f_t_tensor)
    loss_nonlinear.backward()
    optimizer_nonlinear.step()
    losses_nonlinear.append(loss_nonlinear.item())

# Збереження навченої моделі для другого завдання
torch.save(model_nonlinear.state_dict(), 'nonlinear_model.pth')



# Завантаження навчених моделей
model_loaded = PolynomialModel()
model_loaded.load_state_dict(torch.load('polynomial_model.pth'))

model_nonlinear_loaded = NonlinearModel()
model_nonlinear_loaded.load_state_dict(torch.load('nonlinear_model.pth'))

# Отримання прогнозів з моделей
with torch.no_grad():
    y_pred = model_loaded(X_tensor).numpy()
    f_t_pred = model_nonlinear_loaded(t_tensor).numpy()

# Побудова графіку
plt.figure(figsize=(10, 5))

# Графік для першого завдання
plt.subplot(1, 2, 1)
plt.plot(y, y_pred, 'bo', alpha=0.5)
plt.plot([-10, 10], [-10, 10], 'r--')
plt.xlabel('Desired')
plt.ylabel('Predicted')
plt.title('Polynomial Model')

# Графік для другого завдання
plt.subplot(1, 2, 2)
plt.plot(t, f_t, 'bo', alpha=0.5, label='Desired')
plt.plot(t, f_t_pred, 'r-', linewidth=2, label='Predicted')
plt.xlabel('t')
plt.ylabel('f(t)')
plt.title('Nonlinear Model')
plt.legend()

plt.tight_layout()
plt.show()