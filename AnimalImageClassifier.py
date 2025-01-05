import os
import zipfile
import logging
import joblib
import gdown
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder

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
        self.image_size = (224, 224)
        self.batch_size = 10

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.logger.info("Inicjalizacja klasyfikatora obrazów.")

        try:
            model_path = os.path.join(self.path, 'models', 'animal_image_model.h5')
            classes_path = os.path.join(self.path, 'models', 'animal_image_classes.joblib')
            
            if os.path.exists(model_path) and os.path.exists(classes_path):
                self.model = tf.keras.models.load_model(model_path)
                self.classes = joblib.load(classes_path)
                self.logger.info("Model i klasy zostały pomyślnie wczytane.")
            else:
                self.logger.info("Model nie istnieje. Należy go wytrenować.")
                self.download_images_from_drive()
                self.train_model()
        except Exception as e:
            self.logger.critical("Nie udało się wczytać lub wytrenować modelu: %s", str(e))
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
            self.logger.info("Rozpoczynanie treningu modelu...")
            data_dir = os.path.join(self.path, 'baza_zdjecia')

            train_generator, val_generator = self._prepare_data_generators(data_dir)
            checkpoint_path = os.path.join(self.path, "models", "animals_classification_checkpoint.weights.h5")

            input_shape = (*self.image_size, 3)
            num_classes = len(train_generator.class_indices)

            model = self._build_custom_model(input_shape=input_shape, num_classes=num_classes)
            model.compile(
                optimizer=Adam(learning_rate=1e-5),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
            checkpoint_callback = ModelCheckpoint(checkpoint_path, save_weights_only=True, monitor="val_accuracy", save_best_only=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

            model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=50,
                callbacks=[early_stopping, checkpoint_callback, reduce_lr]
            )

            self.model = model
            self.classes = list(train_generator.class_indices.keys())
            joblib.dump(self.classes, os.path.join(self.path, 'models', 'animal_image_classes.joblib'))
            model.save(os.path.join(self.path, 'models', 'animal_image_model.h5'))
            self.logger.info("Model wytrenowano i zapisano.")
        except Exception as e:
            self.logger.critical("Błąd podczas treningu modelu: %s", str(e))
            raise RuntimeError("Nie udało się wytrenować modelu.")
        
    def _build_custom_model(self, input_shape, num_classes):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        return model
    
    def _prepare_data_generators(self, data_dir):
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255.0,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.2
        )

        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training'
        )

        val_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation'
        )
        return train_generator, val_generator
    
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