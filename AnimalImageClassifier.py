from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import numpy as np
import logging
import zipfile
import os
import gdown
import joblib
import shutil
from PIL import Image

class AnimalImageClassifier:
    def __init__(self, drive_folder_id: str, local_path: str, logger: logging.Logger):
        """
        info:
            Inicjalizacja klasyfikatora obrazów.
        args:
            drive_folder_id: str - Id folderu na Google Drive
            local_path: str - Lokalna ścieżka do zapisu danych
            logger: logging.Logger - Logger do logowania informacji
        """
        self.drive_folder_id = drive_folder_id
        self.path = local_path
        self.model = None
        self.classes = None
        self.logger = logger

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.logger.info("Inicjalizacja klasyfikatora obrazów.")

        try:
            self.model = load_model(os.path.join(self.path, 'models', 'animal_image_model.h5'))
            self.classes = joblib.load(os.path.join(self.path, 'models', 'animal_image_classes.joblib'))
            self.logger.info("Model i klasy zostały pomyślnie wczytane.")
        except FileNotFoundError:
            self.logger.info("Model nie istnieje. Należy go wytrenować.")
            try:
                self.download_images_from_drive()
                self.train_model()
            except Exception as e:
                self.logger.critical("Nie udało się wytrenować modelu: %s", str(e))
                raise RuntimeError(f"Błąd inicjalizacji: {e}")

    
    def download_images_from_drive(self):
        """
        Pobiera zdjęcia z Google Drive do lokalnego folderu.
        """
        try:
            self.logger.info("Pobieranie zdjęć z Google Drive za pomocą gdown...")
            output_zip_path = os.path.join(self.path, "drive_images.zip")
            folder_url = f"https://drive.google.com/uc?id={self.drive_folder_id}&export=download"
            gdown.download(folder_url, output_zip_path, quiet=False, fuzzy=True)
            self.logger.info(f"Rozpakowywanie zdjęć z: {output_zip_path}...")
            with zipfile.ZipFile(output_zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.path)
            os.remove(output_zip_path)

            self.logger.info(f"Zdjęcia pobrano i rozpakowano do: {self.path}\\baza_zdjecia")
        except Exception as e:
            self.logger.critical("Błąd podczas pobierania lub rozpakowywania zdjęć: %s", str(e))
            raise RuntimeError("Nie udało się pobrać zdjęć z Google Drive.")
    
    def train_model(self):
        """
        Trenuje model klasyfikacji obrazów.
        """
        try:
            data_dir = os.path.join(self.path, 'baza_zdjecia')
            self.logger.info("Przygotowywanie danych do treningu...")

            datagen = ImageDataGenerator(
                rescale=1.0 / 255.0,
                validation_split=0.2
            )

            train_generator = datagen.flow_from_directory(
                data_dir,
                target_size=(224, 224),
                batch_size=32,
                class_mode='categorical',
                subset='training'
            )

            val_generator = datagen.flow_from_directory(
                data_dir,
                target_size=(224, 224),
                batch_size=32,
                class_mode='categorical',
                subset='validation'
            )

            self.classes = list(train_generator.class_indices.keys())
            self.logger.info(f"Zidentyfikowane klasy: {self.classes}")

            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            x = Flatten()(base_model.output)
            x = Dense(512, activation='relu')(x)
            x = Dropout(0.5)(x)
            output = Dense(len(self.classes), activation='softmax')(x)
            self.model = Model(inputs=base_model.input, outputs=output)

            for layer in base_model.layers:
                layer.trainable = False  # Zamrożenie warstw ResNet50

            self.model.compile(optimizer=Adam(learning_rate=0.001),
                               loss='categorical_crossentropy',
                               metrics=['accuracy'])

            # Trening modelu
            self.logger.info("Rozpoczynanie treningu modelu...")
            self.model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=10,
                steps_per_epoch=train_generator.samples // train_generator.batch_size,
                validation_steps=val_generator.samples // val_generator.batch_size
            )

            # Zapisanie modelu i klas
            models_path = os.path.join(self.path, 'models')
            if not os.path.exists(models_path):
                os.makedirs(models_path)

            joblib.dump(self.classes, os.path.join(models_path, 'animal_image_classes.joblib'))
            self.model.save(os.path.join(models_path, 'animal_image_model.h5'))

            self.logger.info("Model wytrenowano i zapisano.")
        except Exception as e:
            self.logger.critical("Błąd podczas treningu modelu: %s", str(e))
            raise RuntimeError("Nie udało się wytrenować modelu.")
    
    
    def predict_top_10(self, image_path: str) -> list:
        """
        info:
            Przewiduje 10 najbardziej prawdopodobnych zwierząt na podstawie zdjęcia.
        args:
            image_path: str - Ścieżka do zdjęcia.
        return:
            list - Lista 10 zwierząt z prawdopodobieństwami ([(nazwa_zwierzęcia, prawdopodobieństwo)])
        """
        if not self.model or not self.classes:
            self.logger.critical("Model nie został wczytany ani wytrenowany.")
            raise RuntimeError("Model musi zostać wczytany lub wytrenowany przed użyciem.")

        try:
            image = Image.open(image_path).resize((224, 224))
            image_array = np.array(image) / 255.0  # Normalizacja
            image_array = np.expand_dims(image_array, axis=0)  # Dodanie wymiaru batch

            # Przewidywanie
            predictions = self.model.predict(image_array)[0]
            sorted_indices = np.argsort(predictions)[::-1]  # Sortowanie malejące
            top_10 = [(self.classes[i], predictions[i]) for i in sorted_indices[:10]]

            self.logger.info(f"Top 10 przewidywań: {top_10}")
            return top_10
        except Exception as e:
            self.logger.critical("Błąd podczas predykcji: %s", str(e))
            raise RuntimeError("Nie udało się przewidzieć klasy obrazu.")