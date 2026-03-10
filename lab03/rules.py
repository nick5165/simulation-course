def get_age_multiplier(age: int) -> float:
    if age < 30: return 0.4      # Молодой лес (плохо горит)
    if age < 200: return 1.8     # Зрелый лес (горит отлично)
    return 0.6                   # Гниющий (плохо горит)

def get_wind_multiplier(wind_dx, wind_dy, wind_speed, neigh_dx, neigh_dy):
    if wind_speed == 0 or (wind_dx == 0 and wind_dy == 0):
        return 1.0
    # Если ветер дует ОТ соседа К нам, шанс выше
    # Используем скалярное произведение векторов
    dot = (-neigh_dx * wind_dx) + (-neigh_dy * wind_dy)
    if dot > 0: return 1.0 + (wind_speed * 2.0)
    if dot < 0: return 0.2
    return 1.0

def calculate_ignition_chance(base, age_m, wind_m, hum):
    return base * age_m * wind_m * (1.1 - hum)