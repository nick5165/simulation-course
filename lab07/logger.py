import csv
import os
from config import CSV_FILENAME, STATES

class CSVLogger:
    def __init__(self):
        self.filename = CSV_FILENAME
        self.setup_file()

    def setup_file(self):
        # Создаем новый файл (или перезаписываем старый) и пишем заголовки
        with open(self.filename, mode='w', newline='', encoding='utf-8-sig') as file:
            writer = csv.writer(file)
            writer.writerow(["Start_Day", "End_Day", "Weather", "Duration"])

    def log_step(self, start_time, end_time, state, duration):
        # Открываем в режиме 'a' (append) для добавления строки
        with open(self.filename, mode='a', newline='', encoding='utf-8-sig') as file:
            writer = csv.writer(file)
            weather_name = STATES[state][4:] # Убираем "1 - ", "2 - " из названия
            writer.writerow([
                round(start_time, 2), 
                round(end_time, 2), 
                weather_name, 
                round(duration, 2)
            ])