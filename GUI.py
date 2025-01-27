from AnimalFeaturesClassifier import AnimalFeaturesClassifier
from AnimalImageClassifier import AnimalImageClassifier
from AnimalPredictor import AnimalPredictor

import gdown
import zipfile

from PIL import Image, ImageTk

import tkinter as tk
from tkinter import filedialog, messagebox
import os

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Paragraph, Frame
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from datetime import datetime
NOW = datetime.now()
FORMATTED_DATE = NOW.strftime("%Y-%m-%d %H:%M")
FORMATTED_FILENAME_DATE = NOW.strftime("%Y-%m-%d_%H%M")

import base64

class AnimalClassifierApp:
    def __init__(self, root, logger, path):

        self.root = root
        self.root.title("Bliźniaki")
        self.root.configure(bg="#FFFDEC")
        self.logger = logger

        self.path = path
        self.logo_path = os.path.join(self.path, "logo.png")
        self.wstep_rodo_path = os.path.join(self.path, "wstep_rodo.txt")
        self.opisy_path = os.path.join(self.path, "opisy.txt")

        self.animal_labels = {
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

        self.feature_classifier = None
        self.image_classifier = None
        self.combined_classifier = None
        self.selected_image_path = None
        self.feature_sliders = {}
        self.input_features = {}
        self.create_start_page()

    def download_logo(self):
        """
        Pobieranie logo aplikacji.
        """
        logo_url = "https://drive.google.com/uc?id=1-hoa5_PiXkpNEBVcFSxP2d3A_uSNh5Ai"
        gdown.download(logo_url, self.logo_path, quiet=False)

    def download_file(self, url, output):
        """
        Pobieranie pliku z podanego URL-a.
        """
        gdown.download(url, output, quiet=False)

    def get_app_description(self, txt_path):
        """
        Odczytanie wstępu (pierwszego akapitu) z pliku tekstowego.
        """
        lines = []
        with open(txt_path, "r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    lines.append(line.strip())
                if len(lines) == 2:
                    break
        return "\n".join(lines)

    def get_rodo_info(self, txt_path):
        """
        Odczytanie trzeciej i czwartej niepustej linii z pliku tekstowego.
        """
        lines = []
        with open(txt_path, "r", encoding="utf-8") as file:
            line_count = 0
            for line in file:
                if line.strip():
                    line_count += 1
                    if line_count == 3 or line_count == 4:
                        lines.append(line.strip())
                if len(lines) == 2:
                    break
        return " ".join(lines)

    def create_start_page(self):
        """
        Strona startowa z przyciskami do wyboru trybu analizy.
        """
        self.clear_window()
        
        # Dodanie obszaru Canvas
        canvas = tk.Canvas(self.root, bg="#FFFDEC", width=600, height=760, highlightthickness=0)
        scrollable_frame = tk.Frame(canvas, bg="#FFFDEC")
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.place(relx=0.5, rely=0.5, anchor="center")

        # Sprawdzenie, czy logo już istnieje
        if not os.path.exists(self.logo_path):
            try:
                self.download_logo()
            except Exception as e:
                error_label = tk.Label(scrollable_frame, text="Wystąpił błąd podczas pobierania logo.", font=("Century Schoolbook", 14), fg="red", bg="#FFFDEC")
                error_label.pack(pady=10)
                return

        # Sprawdzenie, czy plik wstępu i rodo już istnieje
        if not os.path.exists(self.wstep_rodo_path):
            try:
                txt_url = "https://drive.google.com/uc?id=1xRdUwaccOD0nfoZ4LY9C7u3UPst2y2Pr"
                self.download_file(txt_url, self.wstep_rodo_path)
            except Exception as e:
                error_label = tk.Label(scrollable_frame, text="Wystąpił błąd podczas pobierania plików.", font=("Century Schoolbook", 14), fg="red", bg="#FFFDEC")
                error_label.pack(pady=10)
                return

        # Sprawdzenie, czy plik z opisami zwierząt już istnieje
        if not os.path.exists(self.opisy_path):
            try:
                txt_url = "https://drive.google.com/uc?id=1iX8wOVqkG9INQHYEsPLT43JfbUlMREPK"
                self.download_file(txt_url, self.opisy_path)
            except Exception as e:
                error_label = tk.Label(scrollable_frame, text="Wystąpił błąd podczas pobierania plików.", font=("Century Schoolbook", 14), fg="red", bg="#FFFDEC")
                error_label.pack(pady=10)
                return

        # Dodawanie logo na górze strony głównej
        img = Image.open(self.logo_path)
        img_tk = ImageTk.PhotoImage(img)

        img_label = tk.Label(scrollable_frame, image=img_tk, bg="#FFFDEC")
        img_label.image = img_tk
        img_label.pack(pady=(5,0))

        # Dodawanie opisu aplikacji
        app_description = self.get_app_description(self.wstep_rodo_path)
        if app_description:
            text_label = tk.Label(scrollable_frame, text=app_description, font=("Century Schoolbook", 12), bg="#FFFDEC", fg="#B34C6D", wraplength=600, justify="center")
            text_label.pack(pady=(0, 10))
        else:
            error_label = tk.Label(scrollable_frame, text="Wystąpił błąd podczas ładowania opisu aplikacji.", font=("Century Schoolbook", 14), fg="red", bg="#FFFDEC")
            error_label.pack(pady=10)

        label = tk.Label(scrollable_frame, text="Wybierz opcję:", font=("Century Schoolbook", 14), bg="#FFFDEC")
        label.pack(pady=(5, 5))

        button_width = 30
        button_height = 1
        button_font = ("Century Schoolbook", 16)

        button_bg = "#FFE2E2"  # Kolor przed kliknięciem
        button_active_bg = "#FFCFCF"  # Kolor po kliknięciu

        button_features = tk.Button(scrollable_frame, text="Tylko dane", font=button_font, width=button_width, height=button_height, bg=button_bg, 
                                    activebackground=button_active_bg, command=self.create_feature_input_page)
        button_features.pack(pady=5)

        button_image = tk.Button(scrollable_frame, text="Tylko zdjęcie", font=button_font, width=button_width, height=button_height, bg=button_bg, 
                                activebackground=button_active_bg, command=self.create_image_input_page)
        button_image.pack(pady=5)

        button_features_and_image = tk.Button(scrollable_frame, text="Dane i zdjęcie", font=button_font, width=button_width, height=button_height, bg=button_bg, 
                                            activebackground=button_active_bg, command=self.create_features_page_first)
        button_features_and_image.pack(pady=5)

        # Dodawanie informacji o rodo
        rodo_info = self.get_rodo_info(self.wstep_rodo_path)
        if rodo_info:
            text_label = tk.Label(scrollable_frame, text=rodo_info, font=("Century Schoolbook", 7), bg="#FFFDEC", wraplength=600, justify="center")
            text_label.pack(pady=(20,5))  
        else:
            error_label = tk.Label(scrollable_frame, text="Wystąpił błąd podczas ładowania informacji o RODO.", font=("Century Schoolbook", 14), fg="red", bg="#FFFDEC")
            error_label.pack(pady=10)

        button_quit = tk.Button(scrollable_frame, text="Wyjście", font=button_font, width=button_width, height=button_height, bg=button_bg, 
                                activebackground=button_active_bg, command=self.quit_app)
        button_quit.pack(side=tk.BOTTOM, pady=(5,10))  # Odstęp na dole


    def quit_app(self):
        """
        Zamknięcie aplikacji.
        """
        self.root.destroy()

    def create_feature_input_page(self, next_page=None):
        """
        Strona do wprowadzania cech bez suwaka, ale z użyciem Canvas.
        """
        self.clear_window()

        # Dodanie obszaru Canvas
        canvas = tk.Canvas(self.root, bg="#FFFDEC", width=600, height=760, highlightthickness=0)
        canvas.pack(fill="both", expand=True)
        canvas.place(relx=0.5, rely=0.5, anchor="center")

        # Dodanie tytułu
        label = tk.Label(canvas, text="Wprowadź cechy (0-100):", font=("Century Schoolbook", 16), bg="#FFFDEC")
        label.pack(pady=(30, 10))

        # Resetowanie danych
        self.feature_sliders = {}
        self.input_features = {}

        # Lista cech
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

        # Tworzenie suwaków dla cech
        for feature in features:
            frame = tk.Frame(canvas, bg="#FFFDEC")
            frame.pack(pady=5, fill="x", anchor="center")  # Wyśrodkowanie każdego frame'a

            display_text = feature_labels.get(feature)

            label = tk.Label(frame, text=display_text, font=("Century Schoolbook", 12), bg="#FFFDEC", width=15)
            label.grid(row=0, column=0, padx=10, sticky="w")

            slider = tk.Scale(frame, from_=0, to=100, orient=tk.HORIZONTAL, length=400, bg="#FFFDEC", troughcolor="#FFE2E2", highlightthickness=0)
            slider.grid(row=0, column=1, padx=10, sticky="w")

            self.feature_sliders[feature] = slider

        # Dodanie przycisków na dole
        if next_page:
            button_next = tk.Button(
                canvas, text="Dalej", font=("Century Schoolbook", 14), width=30, height=1,
                bg="#FFE2E2", activebackground="#FFCFCF", command=next_page)
            button_next.pack(side=tk.BOTTOM, pady=20)

        if not next_page:
            # Jeśli next_page nie jest przekazane, pokaż przycisk "Analiza"
            button_analyze = tk.Button(
                canvas, text="Analiza", font=("Century Schoolbook", 14), width=30, height=1,
                bg="#FFE2E2", activebackground="#FFCFCF", command=self.analyze_animal_from_features)
            button_analyze.pack(pady=10)

        button_back = tk.Button(
            canvas, text="Wróć", font=("Century Schoolbook", 14), width=30, height=1,
            bg="#FFE2E2", activebackground="#FFCFCF", command=self.create_start_page)
        button_back.pack(side=tk.BOTTOM, pady=5)


    def create_image_input_page(self):
        """
        Strona do wczytywania zdjęcia bez suwaka.
        """
        self.clear_window()

        canvas = tk.Canvas(self.root, bg="#FFFDEC", width=600, height=760, highlightthickness=0)
        canvas.pack(fill="both", expand=True)

        label = tk.Label(canvas, text="Wczytaj zdjęcie:", font=("Century Schoolbook", 16), bg="#FFFDEC")
        label.pack(pady=(200, 10))

        button_select_image = tk.Button(canvas, text="Wybierz zdjęcie", font=("Century Schoolbook", 14), bg="#FFE2E2", 
        activebackground="#FFCFCF", command=self.select_image_file)
        button_select_image.pack(pady=20)

        button_analyze = tk.Button(canvas, text="Analiza", font=("Century Schoolbook", 14), bg="#FFE2E2", 
        activebackground="#FFCFCF", command=self.analyze_animal_from_image)
        button_analyze.pack(pady=20)

        self.image_label = tk.Label(canvas, text="Brak wybranego zdjęcia", font=("Century Schoolbook", 12), bg="#FFFDEC")
        self.image_label.pack(pady=10)

        button_back = tk.Button(
            canvas, text="Wróć", font=("Century Schoolbook", 14), width=30, height=1,
            bg="#FFE2E2", activebackground="#FFCFCF", command=self.create_start_page)
        button_back.pack(side=tk.BOTTOM, pady=30)

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
        
        self.clear_window()

        # Tworzenie widżetów dla strony z wyborem zdjęcia
        label = tk.Label(self.root, text="Wczytaj zdjęcie:", font=("Century Schoolbook", 16), bg="#FFFDEC")
        label.pack(pady=(200,10))

        button_select_image = tk.Button(self.root, text="Wybierz zdjęcie", font=("Century Schoolbook", 14), bg="#FFE2E2", 
        activebackground="#FFCFCF", command=self.select_image_file)
        button_select_image.pack(pady=20)

        button_analyze = tk.Button(self.root, text="Analiza", font=("Century Schoolbook", 14), bg="#FFE2E2", 
        activebackground="#FFCFCF", command=self.analyze_animal_from_features_and_image)
        button_analyze.pack(pady=20)

        self.image_label = tk.Label(self.root, text="Brak wybranego zdjęcia", font=("Century Schoolbook", 12), bg="#FFFDEC")
        self.image_label.pack(pady=10)

        button_back = tk.Button(
            self.root, text="Wróć", font=("Century Schoolbook", 14), width=30, height=1,
            bg="#FFE2E2", activebackground="#FFCFCF", command=self.create_start_page)
        button_back.pack(side=tk.BOTTOM, pady=30)


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
            if mode in ["image", "combined"] and not self.selected_image_path:
                messagebox.showerror("Błąd", "Nie wybrano żadnego zdjęcia.")
                return

            # Pobranie danych z suwaków, jeśli potrzebne
            if mode in ["features", "combined"] and not self.input_features:
                # Ustawienie input_features na podstawie suwaków (jeśli nie zostały zapisane wcześniej)
                while not self.input_features:
                    self.input_features = {feature: slider.get() for feature, slider in self.feature_sliders.items() if slider.get() != 0}
                    if not self.input_features:
                        messagebox.showerror("Błąd", "Wprowadź przynajmniej jedną cechę.")
                        return

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

            self.show_results(top_animals)
            
            # Resetowanie ścieżki zdjęcia po zakończeniu analizy
            if mode == "image":
                self.selected_image_path = None

        except Exception as e:
            messagebox.showerror("Błąd", f"Wystąpił problem podczas analizy: {e}")
            self.logger.error(f"Wystąpił problem podczas analizy: {e}")

    def show_results(self, top_animals):
        """
        Wyświetla wyniki analizy.
        """
        images_folder_path = os.path.join(self.path, "najlepsze_zdjecia")

        # Sprawdzenie, czy folder zdjęć już istnieje
        if not os.path.exists(images_folder_path):
            try:
                self.download_best_images_from_drive()
            except Exception as e:
                error_label = tk.Label(self.root, text="Nie udało się pobrać zdjęć.", font=("Century Schoolbook", 14), fg="red", bg="#FFFDEC")
                error_label.pack(pady=10)
                return
        
        self.clear_window()

        canvas = tk.Canvas(self.root, bg="#FFFDEC", width=600, height=760, highlightthickness=0)
        canvas.pack(fill="both", expand=True)

        top_animal_name = top_animals[0][0]

        label = tk.Label(canvas, text="Ranking Twoich zwierzęcych bliźniaków:", font=("Century Schoolbook", 24), bg="#FFFDEC")
        label.pack(pady=10)

        animal_image_path = os.path.join(self.path, "najlepsze_zdjecia", f"naj_{top_animals[0][0]}.jpg")
        if os.path.exists(animal_image_path):
            animal_image = Image.open(animal_image_path)
            animal_image = animal_image.resize((230, 230))
            animal_image = ImageTk.PhotoImage(animal_image)
            image_label = tk.Label(canvas, image=animal_image, bg="#FFFDEC")
            image_label.image = animal_image
            image_label.pack(pady=10)

        # Wyświetlenie nazwy top 1 zwierzęcia tak, aby była wyróżniona od pozostałych miejsc w rankingu
        top_animal_display_name = self.animal_labels.get(top_animal_name)
        top_animal_label = tk.Label(canvas, text=top_animal_display_name, font=("Century Schoolbook", 23, "bold"), bg="#FFFDEC")
        top_animal_label.pack(pady=10)

        # Wyświetlenie opisu top 1 zwierzęcia
        animal_description = self.get_animal_description(top_animal_display_name)
        if animal_description:
            text_label = tk.Label(canvas, text=animal_description, font=("Century Schoolbook", 14), bg="#FFFDEC", fg="#B34C6D", wraplength=600, justify="center")
            text_label.pack(pady=(0, 10))
        else:
            error_label = tk.Label(canvas, text="Wystąpił błąd podczas wczytywania opisu zwierzęcia.", font=("Century Schoolbook", 14), fg="red", bg="#FFFDEC")
            error_label.pack(pady=10)

        # Wyświetlenie reszty zwierząt
        for idx, (animal, score) in enumerate(top_animals[1:], start=2):
            animal_display_name = self.animal_labels.get(animal, animal.capitalize())
            result_label = tk.Label(canvas, text=f"{idx}. {animal_display_name}", font=("Century Schoolbook", 12), bg="#FFFDEC")
            result_label.pack(pady=2)

        button_back = tk.Button(canvas, text="Strona główna", font=("Century Schoolbook", 14), width=30, height=1, bg="#FFE2E2", 
        activebackground="#FFCFCF", command=self.create_start_page)
        button_back.pack(pady=(20, 5))

        button_generate = tk.Button(canvas, text="Generuj raport", font=("Century Schoolbook", 14), width=30, height=1, bg="#FFE2E2", 
        activebackground="#FFCFCF", command=lambda: self.generate_raport(top_animals))
        button_generate.pack(pady=5)

        button_quit = tk.Button(canvas, text="Wyjście", font=("Century Schoolbook", 14), width=30, height=1, bg="#FFE2E2", 
        activebackground="#FFCFCF", command=self.quit_app)
        button_quit.pack(pady=5)


    def download_best_images_from_drive(self):
        """
        Pobiera najlepsze zdjęcia zwierząt z Google Drive do lokalnego folderu.
        """
        try:
            self.logger.info("Pobieranie zdjęć z Google Drive za pomocą gdown...")
            output_zip_path = os.path.join(self.path, "drive_best_images.zip")
            file_url = "https://drive.google.com/uc?id=19Png8-wztFJNBv9AOEKfu_MrAc4PVWAK"
            gdown.download(file_url, output_zip_path, quiet=False)
            self.logger.info(f"Rozpakowywanie zdjęć z: {output_zip_path}...")
            with zipfile.ZipFile(output_zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.path)
            os.remove(output_zip_path)

            self.logger.info(f"Zdjęcia pobrano i rozpakowano do: {self.path}\\naj_zdjecia")
        except Exception as e:
            self.logger.critical("Błąd podczas pobierania lub rozpakowywania zdjęć: %s", str(e))
            raise RuntimeError("Nie udało się pobrać zdjęć z Google Drive.")

    def get_animal_description(self, animal_name):
        """
        Odczytanie opisu zwierzęcia na podstawie podanej nazwy zwierzęcia z pliku tekstowego.
        """
        try:
            with open(self.opisy_path, "r", encoding="utf-8") as file:
                lines = file.readlines()

            lines = [line.strip() for line in lines if line.strip()]
        
            # Iteracja po liniach pliku w celu znalezienia nazwy zwierzęcia
            for i in range(0, len(lines), 2):
                name = lines[i]
                if name.lower() == animal_name.lower():
                    description = lines[i + 1]
                    return description
                
            self.logger.warning(f"Opis dla zwierzęcia '{animal_name}' nie został znaleziony.")
            return f"Opis dla zwierzęcia '{animal_name}' nie został znaleziony."
    
        except FileNotFoundError:
            self.logger.critical(f"Plik z opisami nie został znaleziony: {self.opisy_path}")
        except Exception as e:
            self.logger.critical(f"Wystąpił błąd: {str(e)}")
        
    def generate_raport(self, top_animals):
        """
        Generuje raport z analizy.
        """
        try:
            # Pobierz ścieżkę do folderu "Dokumenty" użytkownika
            documents_path = os.path.join(os.path.expanduser("~"), "Documents", "blizniaki")
            if not os.path.exists(documents_path):
                os.makedirs(documents_path)
            
            # Ścieżka do zapisu pliku PDF
            pdf_path = os.path.join(documents_path, f"raport_{FORMATTED_FILENAME_DATE}.pdf")

            # Ścieżka do zapisu pliku HTML
            html_path = os.path.join(documents_path, f"raport_{FORMATTED_FILENAME_DATE}.html")

            self.generate_raport_pdf(top_animals, pdf_path)
            self.generate_raport_html(top_animals, html_path)
            self.logger.info(f"Raporty zostały zapisane w: {documents_path}")
            messagebox.showinfo("Sukces", f"Raporty zostały zapisane w: {documents_path}")

        except Exception as e:
            self.logger.error(f"Wystąpił błąd podczas generowania raportu: {str(e)}")
            messagebox.showerror("Błąd", f"Wystąpił błąd podczas generowania raportu: {str(e)}")

    def generate_raport_pdf(self, top_animals, pdf_path):
        """
        Generuje raport z analizy do pliku pdf.
        """
        # Ścieżka do czcionki
        pdfmetrics.registerFont(TTFont("CenturySchoolbook", os.path.join(self.path, "fonts", "CENSCBK.ttf")))
        pdfmetrics.registerFont(TTFont("CenturySchoolbook-Bold", os.path.join(self.path, "fonts", "SCHLBKB.TTF")))

        # Tworzenie nowego pliku PDF            
        c = canvas.Canvas(pdf_path, pagesize=letter)
        width, height = letter

        # Dodanie logo
        logo_image = ImageReader(self.logo_path)
        c.drawImage(logo_image, (width - 150) /2, height - 100, width=150, height=100, preserveAspectRatio=True)        

        # Tytuł raportu
        c.setFont("CenturySchoolbook-Bold", 24)
        c.drawCentredString(width / 2, height - 130, "Ranking Twoich zwierzęcych bliźniaków")

        # Dodanie dnia i godziny generowania raportu
        c.setFont("CenturySchoolbook", 15)
        c.drawCentredString(width / 2, height - 150, f"Data: {FORMATTED_DATE}")
        
        # Zdjęcie top 1 zwierzęcia
        top_animal_name = top_animals[0][0]
        top_animal_display_name = self.animal_labels.get(top_animal_name, top_animal_name.capitalize())
        c.setFont("CenturySchoolbook-Bold", 16)
        c.drawCentredString(width / 2, height - 190, f"1. {top_animal_display_name}")

        animal_image_path = os.path.join(self.path, "najlepsze_zdjecia", f"naj_{top_animal_name}.jpg")
        if os.path.exists(animal_image_path):
            animal_image = ImageReader(animal_image_path)
            c.drawImage(animal_image, (width - 220) / 2, height - 420, width=220, height=220, preserveAspectRatio=True, anchor='nw')

        # Dodanie opisu top 1 zwierzęcia
        animal_description = self.get_animal_description(top_animal_display_name)
        styles = getSampleStyleSheet()
        style = styles["BodyText"]
        style.fontName = "CenturySchoolbook"
        style.fontSize = 14
        style.leading = 14
        style.alignment = TA_CENTER 
        frame = Frame(50, height - 520, width - 100, 100, showBoundary=0)
        paragraph = Paragraph(animal_description, style)
        frame.addFromList([paragraph], c)

        # Dodanie pozostałych zwierząt
        c.setFont("CenturySchoolbook", 14)
        y_offset = 480
        for idx, animal in enumerate(top_animals[1:], start=2):
            animal_display_name = self.animal_labels.get(animal[0], animal[0].capitalize())
            c.drawCentredString(width / 2, height - y_offset - (idx * 20), f"{idx}. {animal_display_name}")

        # Zakończenie tworzenia PDF
        c.save()

    def generate_raport_html(self, top_animals, html_path):
        """
        Generuje raport z analizy do pliku html.
        """
        # Nazwa top 1 zwierzęcia
        top_animal_name = top_animals[0][0]
        top_animal_display_name = self.animal_labels.get(top_animal_name, top_animal_name.capitalize())
        animal_image_path = os.path.join(self.path, "najlepsze_zdjecia", f"naj_{top_animal_name}.jpg")

        # Konwertuj obraz do Base64
        with open(self.logo_path, "rb") as img_file:
            base64_string_logo = base64.b64encode(img_file.read()).decode("utf-8")

        with open(animal_image_path, "rb") as img_file:
            base64_string_animal = base64.b64encode(img_file.read()).decode("utf-8")

        # Tworzenie struktury HTML
        html_content = f"""
        <!DOCTYPE html>
        <html lang="pl">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Raport Blizniaka</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    padding: 0;
                    line-height: 1.6;
                }}
                h1 {{
                    text-align: center;
                    color: #333;
                }}
                h2 {{
                    margin-top: 40px;
                    color: #444;
                }}
                .animal {{
                    margin: 20px 0;
                }}
                img {{
                    display: block;
                    margin: 0 auto;
                    max-width: 300px;
                    height: auto;
                }}
                .description {{
                    text-align: center;
                    margin: 10px 0;
                }}
            </style>
        </head>
        <body>
            <img src="data:image/jpeg;base64,{base64_string_logo}" alt="Logo aplikacji" style="display: block; margin: 0 auto; max-width: 150px; height: auto;">
            <h1>Ranking Twoich zwierzęcych bliźniaków</h1>
            <p style="text-align: center;">Data generowania raportu: {FORMATTED_DATE}</p>
            
            <div class="animal">
                <h2 style="text-align: center;">1. {self.animal_labels.get(top_animal_name, top_animal_name.capitalize())}</h2>
                <img src="data:image/jpeg;base64,{base64_string_animal}" alt="Zdjęcie zwierzęcia">
                <p class="description">{self.get_animal_description(top_animal_display_name)}</p>
            </div>
                        <ul>
        """
        # Dodanie listy pozostałych zwierząt
        for idx, animal in enumerate(top_animals[1:], start=2):
            animal_name = self.animal_labels.get(animal[0], animal[0].capitalize())
            html_content += f"<p style='text-align: center'>{idx}. {animal_name}</p>"

        # Zamknięcie znaczników HTML
        html_content += """
            </ul>
        </body>
        </html>
        """

        # Zapis pliku HTML
        with open(html_path, "w", encoding="utf-8") as html_file:
            html_file.write(html_content)

    def clear_window(self):
        """
        Czyści zawartość okna.
        """
        for widget in self.root.winfo_children():
            widget.destroy()
