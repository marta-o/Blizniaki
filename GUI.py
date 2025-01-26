from AnimalFeaturesClassifier import AnimalFeaturesClassifier
from AnimalImageClassifier import AnimalImageClassifier
from AnimalPredictor import AnimalPredictor

import gdown
import zipfile

from PIL import Image, ImageTk

import tkinter as tk
from tkinter import filedialog, messagebox
import os

class AnimalClassifierApp:
    def __init__(self, root, logger, path):

        
        self.root = root
        self.root.title("Bliźniaki")
        self.root.configure(bg="#FFFDEC")
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

        label = tk.Label(self.root, text="Wybierz opcję:", font=("Century Schoolbook", 14), bg="#FFFDEC")
        label.pack(pady=5)

        button_width = 30
        button_height = 1
        button_font = ("Century Schoolbook", 16)

        button_bg = "#FFE2E2"  # Kolor przed kliknięciem
        button_active_bg = "#FFCFCF"  # Kolor po kliknięciu

        button_features = tk.Button(self.root, text="Tylko dane", font=button_font, width=button_width, height=button_height, bg=button_bg, 
        activebackground=button_active_bg, command=self.create_feature_input_page)
        button_features.pack(pady=5)

        button_image = tk.Button(self.root, text="Tylko zdjęcie", font=button_font, width=button_width, height=button_height, bg=button_bg, 
        activebackground=button_active_bg, command=self.create_image_input_page)
        button_image.pack(pady=5)

        button_features_and_image = tk.Button(self.root, text="Dane i zdjęcie", font=button_font, width=button_width, height=button_height, bg=button_bg, 
        activebackground=button_active_bg, command=self.create_features_page_first)
        button_features_and_image.pack(pady=5)

        button_quit = tk.Button(self.root, text="Wyjście", font=button_font, width=button_width, height=button_height, bg=button_bg, 
        activebackground=button_active_bg, command=self.quit_app)
        button_quit.pack(side=tk.BOTTOM, pady=30)

    def quit_app(self):
        """
        Zamknięcie aplikacji.
        """
        self.root.destroy()

    def create_feature_input_page(self, next_page=None):
        """
        Strona do wprowadzania cech.
        """
        self.clear_window()

        label = tk.Label(self.root, text="Wprowadź cechy (0-100):", font=("Century Schoolbook", 16), bg="#FFFDEC")
        label.pack(pady=10)

        self.feature_sliders = {}  # Resetowanie suwaków
        self.input_features = {}  # Resetowanie wprowadzonych danych cech

        features = ["lojalnosc", "towarzyskosc", "lenistwo", "troskliwosc", "pozytywnosc", "niezaleznosc",
                    "agresywnosc", "spryt", "odwaga", "pracowitosc"]
        
        feature_labels = {
            "lojalnosc": "Lojalność",
            "towarzyskosc": "Towarzyskość",
            "lenistwo": "Lenistwo",
            "troskliwosc": "Troskliwość",
            "pozytywnosc": "Pozytywność",
            "niezaleznosc": "Niezależność",
            "agresywnosc": "Agresywność",
            "spryt": "Spryt",
            "odwaga": "Odwaga",
            "pracowitosc": "Pracowitość"
            }

        for feature in features:
            frame = tk.Frame(self.root, bg="#FFFDEC", highlightbackground="#FFFDEC", highlightcolor="#FFFDEC", highlightthickness=4)
            frame.pack(pady=5, fill="x")

            display_text = feature_labels.get(feature)

            label = tk.Label(frame, highlightbackground="#FF0000", text=display_text, font=("Century Schoolbook", 12), bg="#FFFDEC", width=15)
            label.grid(row=0, column=0, padx=10, sticky="w")

            slider = tk.Scale(frame, highlightbackground="#FFCFCF", from_=0, to=100, orient=tk.HORIZONTAL, length=400, bg="#FFFDEC", troughcolor="#FFE2E2", highlightthickness=0)
            slider.grid(row=0, column=1, padx=10, sticky="w")

            self.feature_sliders[feature] = slider

        if next_page:
            button_next = tk.Button(
                self.root, text="Dalej", font=("Century Schoolbook", 14), width=30, height=1,
                bg="#FFE2E2", activebackground="#FFCFCF", command=next_page)
            button_next.pack(side=tk.BOTTOM, pady=30)
        
            button_back = tk.Button(
                self.root, text="Wróć", font=("Century Schoolbook", 14), width=30, height=1,
                bg="#FFE2E2", activebackground="#FFCFCF", command=self.create_start_page)
            button_back.pack(side=tk.BOTTOM, pady=30)

            button_analyze = tk.Button(
                self.root, text="Analiza", font=("Century Schoolbook", 14), width=30, height=1,
                bg="#FFE2E2", activebackground="#FFCFCF", command=self.analyze_animal_from_features)
            button_analyze.pack(pady=20)

        button_back = tk.Button(
            self.root, text="Wróć", font=("Century Schoolbook", 14), width=30, height=1,
            bg="#FFE2E2", activebackground="#FFCFCF", command=self.create_start_page)
        button_back.pack(side=tk.BOTTOM, pady=30)

        button_analyze = tk.Button(
            self.root, text="Analiza", font=("Century Schoolbook", 14), width=30, height=1,
            bg="#FFE2E2", activebackground="#FFCFCF", command=self.analyze_animal_from_features)
        button_analyze.pack(pady=20)

    def reset_and_return_to_start(self):
        """
        Funkcja resetująca dane i wracająca do strony startowej.
        """
        self.selected_image_path = None  # Resetowanie ścieżki zdjęcia
        self.input_features = {}  # Resetowanie wprowadzonych danych cech
        self.feature_sliders = {}  # Resetowanie suwaków
        self.create_start_page()  # Powrót do strony startowej

    def create_image_input_page(self):
        """
        Strona do wczytywania zdjęcia.
        """
        self.clear_window()

        label = tk.Label(self.root, text="Wczytaj zdjęcie:", font=("Century Schoolbook", 16), bg="#FFFDEC")
        label.pack(pady=10)

        button_select_image = tk.Button(self.root, text="Wybierz zdjęcie", font=("Century Schoolbook", 14), bg="#FFE2E2", 
        activebackground="#FFCFCF", command=self.select_image_file)
        button_select_image.pack(pady=20)

        button_analyze = tk.Button(self.root, text="Analiza", font=("Century Schoolbook", 14), bg="#FFE2E2", 
        activebackground="#FFCFCF", command=self.analyze_animal_from_image)
        button_analyze.pack(pady=20)

        self.image_label = tk.Label(self.root, text="Brak wybranego zdjęcia", font=("Century Schoolbook", 12), bg="#FFFDEC")
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
        label = tk.Label(self.root, text="Wczytaj zdjęcie:", font=("Century Schoolbook", 16), bg="#FFFDEC")
        label.pack(pady=10)

        button_select_image = tk.Button(self.root, text="Wybierz zdjęcie", font=("Century Schoolbook", 14), bg="#FFE2E2", 
        activebackground="#FFCFCF", command=self.select_image_file)
        button_select_image.pack(pady=20)

        button_analyze = tk.Button(self.root, text="Analiza", font=("Century Schoolbook", 14), bg="#FFE2E2", 
        activebackground="#FFCFCF", command=self.analyze_animal_from_features_and_image)
        button_analyze.pack(pady=20)

        self.image_label = tk.Label(self.root, text="Brak wybranego zdjęcia", font=("Century Schoolbook", 12), bg="#FFFDEC")
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
            file_id = '179GmVjydVw8D9RqUB1hQ2FPq6JRYURv3'
            self.feature_classifier = AnimalFeaturesClassifier(drive_file_id=file_id, local_path=self.path, logger=self.logger)

            folder_id = '15SPPgjtECp5FWawf2z_lWKhvlpy6EnMU'
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
        images_folder_path = os.path.join(self.path, "najlepsze_zdjecia")

        # Sprawdzenie, czy folder zdjęć już istnieje
        if not os.path.exists(images_folder_path):
            try:
                # Jeśli folder nie istnieje, pobieramy zdjęcia
                self.download_best_images_from_drive()
            except Exception as e:
                # Jeśli wystąpi błąd podczas pobierania zdjęć, wyświetlamy komunikat
                error_label = tk.Label(self.root, text="Nie udało się pobrać zdjęć.", font=("Century Schoolbook", 14), fg="red", bg="#FFFDEC")
                error_label.pack(pady=10)
                return
        
        self.clear_window()

        animal_labels = {
            "delfin": "Delfin",
            "jelen": "Jeleń",
            "jez": "Jeż",
            "koala": "Koala",
            "kon": "Koń",
            "kot": "Kot",
            "krolik": "Królik",
            "lew": "Lew",
            "lis": "Lis",
            "mrowka": "Mrówka",
            "panda": "Panda",
            "papuga": "Papuga",
            "pies": "Pies",
            "pszczola": "Pszczoła",
            "rekin": "Rekin",
            "sowa": "Sowa",
            "surykatka": "Surykatka",
            "tygrys": "Tygrys",
            "wilk": "Wilk",
            "zolw": "Żółw",
            "zyrafa": "Żyrafa"
            }

        top_animal_name=top_animals[0][0]

        label = tk.Label(self.root, text="Twoje top 5 zwierząt:", font=("Century Schoolbook", 19), bg="#FFFDEC")
        label.pack(pady=10)

        animal_image_path = os.path.join(self.path, "najlepsze_zdjecia", f"naj_{top_animals[0][0]}.jpg")
        if os.path.exists(animal_image_path):
            animal_image = Image.open(animal_image_path)
            animal_image = ImageTk.PhotoImage(animal_image)
            image_label = tk.Label(self.root, image=animal_image, bg="#FFFDEC")
            image_label.image = animal_image
            image_label.pack(pady=10)


        # Wyświetlenie nazwy top 1 zwierzęcia na 1 miejscu
        top_animal_display_name = animal_labels.get(top_animal_name)
        top_animal_label = tk.Label(self.root, text=top_animal_display_name, font=("Century Schoolbook", 22, "bold"), bg="#FFFDEC")
        top_animal_label.pack(pady=10)

        # Wyświetlenie reszty zwierząt
        for idx, (animal, score) in enumerate(top_animals[1:], start=2):
            animal_display_name = animal_labels.get(animal, animal.capitalize())
            result_label = tk.Label(self.root, text=f"{idx}. {animal_display_name}", font=("Century Schoolbook", 12), bg="#FFFDEC")
            result_label.pack(pady=5)


        button_back = tk.Button(self.root, text="Powrót", font=("Century Schoolbook", 14), bg="#FFE2E2", 
        activebackground="#FFCFCF", command=self.create_start_page)
        button_back.pack(pady=20)

        button_quit = tk.Button(self.root, text="Wyjście", font=("Century Schoolbook", 14), width=30, height=1, bg="#FFE2E2", 
        activebackground="#FFCFCF", command=self.quit_app)
        button_quit.pack(side=tk.BOTTOM, pady=30)

    def download_best_images_from_drive(self):
        """
        Pobiera najlepsze zdjęcia zwierząt z Google Drive do lokalnego folderu.
        """
        try:
            self.logger.info("Pobieranie zdjęć z Google Drive za pomocą gdown...")
            output_zip_path = os.path.join(self.path, "drive_best_images.zip")
            file_url = "https://drive.google.com/uc?id=1E-6sMRsuS08m1MOv3hAOrLZJjKBWE1ik"
            gdown.download(file_url, output_zip_path, quiet=False)
            self.logger.info(f"Rozpakowywanie zdjęć z: {output_zip_path}...")
            with zipfile.ZipFile(output_zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.path)
            os.remove(output_zip_path)

            self.logger.info(f"Zdjęcia pobrano i rozpakowano do: {self.path}\\naj_zdjecia")
        except Exception as e:
            self.logger.critical("Błąd podczas pobierania lub rozpakowywania zdjęć: %s", str(e))
            raise RuntimeError("Nie udało się pobrać zdjęć z Google Drive.")

    def clear_window(self):
        """
        Czyści zawartość okna.
        """
        for widget in self.root.winfo_children():
            widget.destroy()
