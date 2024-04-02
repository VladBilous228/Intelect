import numpy as np
import skfuzzy as fuzzy
from skfuzzy import control as ctrl

# Вхідні та вихідні змінні
distance = ctrl.Antecedent(np.arange(0, 101, 1), 'distance')
obstacle_position = ctrl.Antecedent(np.arange(-50, 51, 1), 'obstacle_position')
steering = ctrl.Consequent(np.arange(-90, 91, 1), 'steering')

# Функції належності
distance['close'] = fuzzy.trimf(distance.universe, [0, 0, 40])
distance['medium'] = fuzzy.trimf(distance.universe, [20, 50, 80])
distance['far'] = fuzzy.trimf(distance.universe, [60, 100, 100])

obstacle_position['left'] = fuzzy.trimf(obstacle_position.universe, [-50, -50, 0])
obstacle_position['center'] = fuzzy.trimf(obstacle_position.universe, [-25, 0, 25])
obstacle_position['right'] = fuzzy.trimf(obstacle_position.universe, [0, 50, 50])

steering['hard_left'] = fuzzy.trimf(steering.universe, [-90, -90, -45])
steering['left'] = fuzzy.trimf(steering.universe, [-75, -45, -15])
steering['straight'] = fuzzy.trimf(steering.universe, [-30, 0, 30])
steering['right'] = fuzzy.trimf(steering.universe, [15, 45, 75])
steering['hard_right'] = fuzzy.trimf(steering.universe, [45, 90, 90])

# Нечіткі правила
rule1 = ctrl.Rule(distance['close'] & obstacle_position['left'], steering['hard_right'])
rule2 = ctrl.Rule(distance['close'] & obstacle_position['center'], steering['hard_right'])
rule3 = ctrl.Rule(distance['close'] & obstacle_position['right'], steering['hard_right'])
rule4 = ctrl.Rule(distance['medium'] & obstacle_position['left'], steering['right'])
rule5 = ctrl.Rule(distance['medium'] & obstacle_position['center'], steering['straight'])
rule6 = ctrl.Rule(distance['medium'] & obstacle_position['right'], steering['left'])
rule7 = ctrl.Rule(distance['far'] & obstacle_position['left'], steering['left'])
rule8 = ctrl.Rule(distance['far'] & obstacle_position['center'], steering['straight'])
rule9 = ctrl.Rule(distance['far'] & obstacle_position['right'], steering['right'])

# Створення системи керування
car_steering_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
car_steering = ctrl.ControlSystemSimulation(car_steering_ctrl)

# Дослідження системи на різних значеннях вхідних даних
car_steering.input['distance'] = 70
car_steering.input['obstacle_position'] = -20
car_steering.compute()
print(car_steering.output['steering'])

car_steering.input['distance'] = 30
car_steering.input['obstacle_position'] = 40
car_steering.compute()
print(car_steering.output['steering'])

# Зберегти параметри FIS у файл MATLAB
with open("car_steering_control.fis", "w") as f:
    f.write("distance = [0 0 40 20 50 80 60 100 100];\n")
    f.write("obstacle_position = [-50 -50 0 -25 0 25 0 50 50];\n")
    f.write("steering = [-90 -90 -45 -75 -45 -15 -30 0 30 15 45 75 45 90 90];\n")
    f.write("ruleList = [1 1 1 3 1 5 1 7 1; 1 1 2 3 1 5 1 7 2; 1 1 3 3 1 5 1 7 3;\n")
    f.write("            1 2 1 3 2 5 1 7 4; 1 2 2 3 2 5 1 7 5; 1 2 3 3 2 5 1 7 6;\n")
    f.write("            1 3 1 3 3 5 1 7 7; 1 3 2 3 3 5 1 7 8; 1 3 3 3 3 5 1 7 9];\n")

print("Fuzzy Inference System parameters saved to car_steering_control.fis.")