import math

def calculate_sector_area(radius, angle):
    # Переведення кута з градусів у радіани
    angle_radians = math.radians(angle)
    # Обчислення площі сектора за формулою
    area = 0.5 * radius**2 * angle_radians
    return area

def main():
    # Запит користувача на введення радіуса кола та кута сектора
    radius = float(input("Будь ласка, введіть радіус кола: "))
    angle = float(input("Будь ласка, введіть кут сектора (у градусах): "))
    
    # Обчислення площі сектора кола за допомогою функції
    area = calculate_sector_area(radius, angle)
    
    # Виведення результату
    print("Площа сектора кола з радіусом", radius, "та кутом", angle, "дорівнює", area)

if __name__ == "__main__":
    main()
