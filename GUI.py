from AnimalFeaturesClassifier import AnimalFeaturesClassifier
from AnimalImageClassifier import AnimalImageClassifier
from AnimalPredictor import AnimalPredictor

import tkinter as tk
from tkinter import filedialog, messagebox
import os

class AnimalClassifierApp:
    def __init__(self, root, logger, path):
        self.root = root
        self.logger = logger
        self.path = path
        self.feature_classifier = None
        self.image_classifier = None
        self.combined_classifier = None
        self.selected_image_path = None
        self.feature_sliders = {}
        self.input_features = {}
        self.create_start_page()

    def create_start_page(self):
        """
        Strona startowa z przyciskami do wyboru trybu analizy.
        """
        self.clear_window()

        label = tk.Label(self.root, text="Wybierz opcję:", font=("Arial", 16))
        label.pack(pady=10)

        button_features = tk.Button(self.root, text="Tylko dane", font=("Arial", 14), command=self.create_feature_input_page)
        button_features.pack(pady=20)

        button_image = tk.Button(self.root, text="Tylko zdjęcie", font=("Arial", 14), command=self.create_image_input_page)
        button_image.pack(pady=20)

        button_features_and_image = tk.Button(self.root, text="Dane i zdjęcie", font=("Arial", 14), command=self.create_features_page_first)
        button_features_and_image.pack(pady=20)

    def create_feature_input_page(self, next_page=None):
        """
        Strona do wprowadzania cech zwierzęcia.
        """
        self.clear_window()

        label = tk.Label(self.root, text="Wprowadź cechy zwierzęcia (0-100):", font=("Arial", 16))
        label.pack(pady=10)

        self.feature_sliders = {}
        features = ["lojalnosc", "towarzyskosc", "lenistwo", "troskliwosc", "pozytywnosc", "niezaleznosc",
                    "inteligencja", "ambicja", "energicznosc", "spryt", "odwaga", "pracowitosc", "pewnosc_siebie"]

        for feature in features:
            frame = tk.Frame(self.root)
            label = tk.Label(frame, text=feature.capitalize(), font=("Arial", 12))
            label.pack(side=tk.LEFT, padx=10)
            slider = tk.Scale(frame, from_=0, to=100, orient=tk.HORIZONTAL, length=200)
            slider.pack(side=tk.LEFT)
            frame.pack(pady=5)
            self.feature_sliders[feature] = slider

        if next_page:
            button_next = tk.Button(self.root, text="Dalej", font=("Arial", 14), command=next_page)
            button_next.pack(pady=20)
        else:
            button_analyze = tk.Button(self.root, text="Analiza", font=("Arial", 14), command=self.analyze_animal_from_features)
            button_analyze.pack(pady=20)

    def create_image_input_page(self):
        """
        Strona do wczytywania zdjęcia zwierzęcia.
        """
        self.clear_window()

        label = tk.Label(self.root, text="Wczytaj zdjęcie zwierzęcia:", font=("Arial", 16))
        label.pack(pady=10)

        button_select_image = tk.Button(self.root, text="Wybierz zdjęcie", font=("Arial", 14), command=self.select_image_file)
        button_select_image.pack(pady=20)

        button_analyze = tk.Button(self.root, text="Analiza", font=("Arial", 14), command=self.analyze_animal_from_image)
        button_analyze.pack(pady=20)

        self.image_label = tk.Label(self.root, text="Brak wybranego zdjęcia", font=("Arial", 12))
        self.image_label.pack(pady=10)

    def create_features_page_first(self):
        """
        Strona do wprowadzania cech, która prowadzi do strony wczytywania zdjęcia.
        """
        self.create_feature_input_page(next_page=self.create_image_page_after_features)

    def create_image_page_after_features(self):
        """
        Strona do wczytywania zdjęcia po wprowadzeniu cech.
        """
        # Zapisanie danych z suwaków przed usunięciem widżetów
        self.input_features = {feature: slider.get() for feature, slider in self.feature_sliders.items()}
        
        # Czyszczenie okna
        self.clear_window()

        # Tworzenie widżetów dla strony z wyborem zdjęcia
        label = tk.Label(self.root, text="Wczytaj zdjęcie zwierzęcia:", font=("Arial", 16))
        label.pack(pady=10)

        button_select_image = tk.Button(self.root, text="Wybierz zdjęcie", font=("Arial", 14), command=self.select_image_file)
        button_select_image.pack(pady=20)

        button_analyze = tk.Button(self.root, text="Analiza", font=("Arial", 14), command=self.analyze_animal_from_features_and_image)
        button_analyze.pack(pady=20)

        self.image_label = tk.Label(self.root, text="Brak wybranego zdjęcia", font=("Arial", 12))
        self.image_label.pack(pady=10)


    def select_image_file(self):
        """
        Wybiera plik zdjęcia.
        """
        file_path = filedialog.askopenfilename(filetypes=[("Pliki obrazów", "*.jpg;*.jpeg;*.png")])
        if file_path:
            self.selected_image_path = file_path
            self.image_label.config(text=f"Wybrano: {os.path.basename(file_path)}")

    def analyze_animal_from_features(self):
        self._analyze("features")

    def analyze_animal_from_image(self):
        self._analyze("image")

    def analyze_animal_from_features_and_image(self):
        self._analyze("combined")

    def _analyze(self, mode):
        try:
            # Sprawdzenie zdjęcia, jeśli potrzebne
            if mode in ["image", "combined"] and not self.selected_image_path:
                messagebox.showerror("Błąd", "Nie wybrano żadnego zdjęcia.")
                return

            # Pobranie danych z suwaków, jeśli potrzebne
            if mode in ["features", "combined"] and not self.input_features:
                # Ustawienie input_features na podstawie suwaków (jeśli nie zostały zapisane wcześniej)
                self.input_features = {feature: slider.get() for feature, slider in self.feature_sliders.items()}

            # Inicjalizacja klasyfikatorów
            file_id = '1Mlu7a5gCSGnOBxBuPC65QkFmMYjxGfPA'
            self.feature_classifier = AnimalFeaturesClassifier(drive_file_id=file_id, local_path=self.path, logger=self.logger)

            folder_id = '1AWlHGCgRAXGkkGML_NK4FKsjURPMEtzm'
            self.image_classifier = AnimalImageClassifier(drive_folder_id=folder_id, local_path=self.path, logger=self.logger)

            self.combined_classifier = AnimalPredictor(features_classifier=self.feature_classifier, image_classifier=self.image_classifier, logger=self.logger)

            # Analiza na podstawie wybranego trybu
            if mode == "features":
                top_animals = self.combined_classifier.predict_top_5(input_features=self.input_features)
            elif mode == "image":
                top_animals = self.combined_classifier.predict_top_5(image_path=self.selected_image_path)
            else:
                top_animals = self.combined_classifier.predict_top_5(input_features=self.input_features, image_path=self.selected_image_path)

            # Wyświetlenie wyników
            self.show_results(top_animals)
            
            # Resetowanie ścieżki zdjęcia po zakończeniu analizy
            if mode == "image":
                self.selected_image_path = None  # Resetowanie ścieżki zdjęcia

        except Exception as e:
            messagebox.showerror("Błąd", f"Wystąpił problem podczas analizy: {e}")

    def show_results(self, top_animals):
        """
        Wyświetla wyniki analizy.
        """
        self.clear_window()

        label = tk.Label(self.root, text="Top 5 zwierząt:", font=("Arial", 16))
        label.pack(pady=10)

        for idx, (animal, score) in enumerate(top_animals):
            result_label = tk.Label(self.root, text=f"{idx + 1}. {animal}", font=("Arial", 12))
            result_label.pack(pady=5)

        button_back = tk.Button(self.root, text="Powrót", font=("Arial", 14), command=self.create_start_page)
        button_back.pack(pady=20)

    def clear_window(self):
        """
        Czyści zawartość okna.
        """
        for widget in self.root.winfo_children():
            widget.destroy()
