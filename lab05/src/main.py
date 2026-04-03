from generator import MultiplicateConGenerator
from modeler import EventModeler
from gui import AppGUI

def main():
    generator = MultiplicateConGenerator()

    yes_no_events = [
        ("Да, побалуй себя, закажи пиццу", 0.65),
        ("Нет, сэкономь деньги и приготовь гречку", 0.35)
    ]

    ball_events = [
        ("Сделать генеральную уборку", 0.15),
        ("Почитать полезную книгу", 0.25),
        ("Посмотреть новый сериал", 0.20),
        ("Пойти на долгую прогулку", 0.15),
        ("Учить Python", 0.15),
        ("Просто спать весь день", 0.10)
    ]

    modeler_yes_no = EventModeler(generator, yes_no_events)
    modeler_8ball = EventModeler(generator, ball_events)

    app = AppGUI(modeler_yes_no, modeler_8ball)
    app.mainloop()

if __name__ == "__main__":
    main()