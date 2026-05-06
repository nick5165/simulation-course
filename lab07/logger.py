# logger.py
import csv
from config import CSV_FILENAME, STATES

class CSVLogger:
    def __init__(self):
        self.filename = CSV_FILENAME
        self.setup_file()

    def setup_file(self):
        # Создаем файл с новыми заголовками
        with open(self.filename, mode='w', newline='', encoding='utf-8-sig') as file:
            writer = csv.writer(file)
            writer.writerow(["Day", "Weather", "Chi_Square", "Hypothesis"])

    def log_step(self, day, state, chi_sq, hyp_result):
        # Пишем день, погоду и статистику гипотезы
        with open(self.filename, mode='a', newline='', encoding='utf-8-sig') as file:
            writer = csv.writer(file)
            weather_name = STATES[state][4:]
            writer.writerow([day, weather_name, chi_sq, hyp_result])