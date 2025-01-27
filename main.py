from GUI import AnimalClassifierApp 
import logging
import os
import tkinter as tk

# Konfiguracja loggera
path = r'C:\Users\marta\OneDrive - biurox365ml\Pulpit\studia\sem5\inzynieria_oprogramowania\coding'  # Ścieżka do zapisu danych
log_file = os.path.join(path, "animal_classifier.log")

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="w", encoding="UTF-8"),
    ]
)
logger = logging.getLogger("AnimalClassifierLog")

# Tworzenie głównego okna Tkinter
root = tk.Tk()
root.title("Klasyfikator Zwierząt")
root.attributes("-fullscreen", True)

# Tworzenie aplikacji
app = AnimalClassifierApp(root, logger, path)

# Uruchomienie aplikacji GUI
root.mainloop()

# from AnimalFeaturesClassifier import AnimalFeaturesClassifier
# from AnimalImageClassifier import AnimalImageClassifier
# from AnimalPredictor import AnimalPredictor

# # --- PRZYKŁADOWE UŻYCIE ---

# # klasyfikator cech zwierząt
# file_id = '179GmVjydVw8D9RqUB1hQ2FPq6JRYURv3'
# feature_classifier = AnimalFeaturesClassifier(drive_file_id=file_id, local_path=path, logger=logger)

# new_animal = {
#     "lojalnosc": 60,
#     "towarzyskosc": 50,
#     "lenistwo": 30,
#     "pracowitosc" : 70,
#     "agresywnosc": 40, # nieznana cecha - pominięta
#     # Brakuje niektórych cech - będą uzupełnione medianą
# }

# top_animals = feature_classifier.predict_top_10(input_features=new_animal)
# print(top_animals)

# # klasyfikator obrazów
# folder_id = '1AWlHGCgRAXGkkGML_NK4FKsjURPMEtzm'
# image = r'' # ścieżka do obrazu

# image_classifier = AnimalImageClassifier(drive_folder_id=folder_id, local_path=path, logger=logger)

# połączony klasyfikator

# combined_classifier = AnimalPredictor(features_classifier=feature_classifier, image_classifier=image_classifier, logger=logger)

# top_animals = combined_classifier.predict_top_5(image_path=image, input_features=new_animal)
# print("Top 5 zwierząt:", top_animals)

# top_animals_from_image = combined_classifier.predict_top_5(image_path=image)
# print("Top 5 zwierząt z obrazu:", top_animals_from_image)

# top_animals_from_features = combined_classifier.predict_top_5(input_features=new_animal)
# print("Top 5 zwierząt z cech:", top_animals_from_features)

