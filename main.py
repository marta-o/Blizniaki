from AnimalFeaturesClassifier import AnimalFeaturesClassifier

# --- PRZYKŁADOWE UŻYCIE ---

file_id = '1Mlu7a5gCSGnOBxBuPC65QkFmMYjxGfPA'

# 1. Inicjalizacja klasyfikatora
classifier = AnimalFeaturesClassifier(drive_file_id=file_id)

# 2. Trening modelu (jeśli model jeszcze nie istnieje)
if classifier.model is None:
    classifier.train_model()

# 3. Przewidywanie dla nowego zwierzęcia
new_animal = {
    "lojalnosc": 60,
    "towarzyskosc": 50,
    "lenistwo": 30,
    # Brakuje niektórych cech - będą uzupełnione medianą
}

predicted_animal = classifier.predict(new_animal)
print(f"Nowe zwierzę zostało sklasyfikowane jako: {predicted_animal}")

