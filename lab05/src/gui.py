import customtkinter as ctk

class AppGUI(ctk.CTk):
    def __init__(self, modeler_yes_no, modeler_8ball):
        super().__init__()

        self.modeler_yes_no = modeler_yes_no
        self.modeler_8ball = modeler_8ball

        self.title("Моделирование случайных событий")
        self.geometry("600x400")
        self.resizable(False, False)
        
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.tabview = ctk.CTkTabview(self, width=550, height=350)
        self.tabview.pack(padx=20, pady=20)

        self.tabview.add("Да / Нет")
        self.tabview.add("Шар предсказаний")

        self.setup_yes_no_tab()
        self.setup_8ball_tab()

    def setup_yes_no_tab(self):
        tab = self.tabview.tab("Да / Нет")
        
        title_label = ctk.CTkLabel(tab, text="Заказывать ли сегодня доставку еды?", font=ctk.CTkFont(size=20, weight="bold"))
        title_label.pack(pady=(30, 20))

        self.yes_no_result = ctk.CTkLabel(tab, text="Нажми на кнопку, чтобы узнать", font=ctk.CTkFont(size=16), text_color="gray")
        self.yes_no_result.pack(pady=(0, 30))

        action_button = ctk.CTkButton(tab, text="Получить ответ", width=200, height=40, command=self.roll_yes_no)
        action_button.pack()

    def setup_8ball_tab(self):
        tab = self.tabview.tab("Шар предсказаний")
        
        title_label = ctk.CTkLabel(tab, text="Чем заняться на этих выходных?", font=ctk.CTkFont(size=20, weight="bold"))
        title_label.pack(pady=(30, 20))

        self.ball_result = ctk.CTkLabel(tab, text="Потряси шар...", font=ctk.CTkFont(size=16), text_color="gray")
        self.ball_result.pack(pady=(0, 30))

        action_button = ctk.CTkButton(tab, text="Потрясти шар", width=200, height=40, command=self.roll_8ball)
        action_button.pack()

    def roll_yes_no(self):
        result = self.modeler_yes_no.get_random_event()
        self.yes_no_result.configure(text=result, text_color="white")

    def roll_8ball(self):
        result = self.modeler_8ball.get_random_event()
        self.ball_result.configure(text=result, text_color="white")