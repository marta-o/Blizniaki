from AnimalFeaturesClassifier import AnimalFeaturesClassifier
from AnimalImageClassifier import AnimalImageClassifier

import logging
import os

path = '' # ścieżka do zapisu plików

log_file = os.path.join(path, "animal_classifier.log")

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="w", encoding = "UTF-8"),
    ]
)

logger = logging.getLogger("AnimalClassifierLog")

# --- PRZYKŁADOWE UŻYCIE ---

file_id = '1Mlu7a5gCSGnOBxBuPC65QkFmMYjxGfPA'

feature_classifier = AnimalFeaturesClassifier(drive_file_id=file_id, local_path=path, logger=logger)

# Przewidywanie dla nowego zwierzęcia
new_animal = {
    "lojalnosc": 60,
    "towarzyskosc": 50,
    "lenistwo": 30,
    "agresywnosc": 40, # nieznana cecha - pominięta
    # Brakuje niektórych cech - będą uzupełnione medianą
}

predicted_animal = feature_classifier.predict_top_10(new_animal)
print(f"Nowe zwierzę zostało sklasyfikowane jako: {predicted_animal}")

folder_id = '1AWlHGCgRAXGkkGML_NK4FKsjURPMEtzm'
image = '' # ścieżka do obrazu

image_classifier = AnimalImageClassifier(drive_folder_id=folder_id, local_path=path, logger=logger)

result = image_classifier.predict_top_10(image)
print(f"Najbardziej podobne zwierzę: {result}")








