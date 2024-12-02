from AnimalFeaturesClassifier import AnimalFeaturesClassifier

# --- PRZYKŁADOWE UŻYCIE ---

file_id = '1Mlu7a5gCSGnOBxBuPC65QkFmMYjxGfPA'
path = '' # Ścieżka do zapisu danych    

# Inicjalizacja klasyfikatora
classifier = AnimalFeaturesClassifier(drive_file_id=file_id, path=path)

# Przewidywanie dla nowego zwierzęcia
new_animal = {
    "lojalnosc": 60,
    "towarzyskosc": 50,
    "lenistwo": 30,
    "agresywnosc": 40, # nieznana cecha - pominięta
    # Brakuje niektórych cech - będą uzupełnione medianą
}

predicted_animal = classifier.predict(new_animal)
print(f"Nowe zwierzę zostało sklasyfikowane jako: {predicted_animal}")




