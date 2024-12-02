import io
import requests
import sqlite3
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
import joblib
import logging
import os

class AnimalFeaturesClassifier:
    def __init__(self, drive_file_id : str, path):
        """
        info:
            Inicjalizacja klasyfikatora.
        args:
            db_path: str - Ścieżka do bazy SQLite z danymi
            model_path: str - Ścieżka do pliku z modelem
        """
        self.drive_file_id = drive_file_id
        self.path = path
        self.model = None
        self.imputer = None
        self.features = None

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        log_file = os.path.join(self.path, "animal_classifier.log")
        logging.basicConfig(
            filename=log_file,               # Ścieżka do pliku logu
            level=logging.INFO,              # Poziom logowania
            format="%(asctime)s - %(levelname)s - %(message)s"  # Format logu
        )
        logging.info("Inicjalizacja klasyfikatora.")

        try:
            self.model = joblib.load(self.path + r'\models\animal_features_model.joblib')
            self.imputer = joblib.load(self.path +r'\models\animal_features_imputer.joblib')
            self.features = joblib.load(self.path + r'\models\animal_features_features.joblib')
            logging.info("Model wczytano pomyslnie.")
        except FileNotFoundError:
            logging.warning("Model nie istnieje. Nalezy go wytrenowac.")
            self.conn = self.load_data_from_drive()
            self.train_model()

    def load_data_from_drive(self):
        """
        info:
            Pobiera bazę danych SQLite z Google Drive bez zapisywania lokalnie
        """
        logging.info("Pobieranie bazy danych z Google Drive...")
        download_url = f"https://drive.google.com/uc?id={self.drive_file_id}&export=download"
        response = requests.get(download_url)
        response.raise_for_status()
        db_bytes = io.BytesIO(response.content)

        temp_db_path = os.path.join(self.path, 'temp_db.sqlite')

        with open(temp_db_path, 'wb') as temp_file:
            temp_file.write(db_bytes.read())  # Tymczasowo zapisujemy plik

        # Utwórz połączenie z bazą danych w pamięci
        conn = sqlite3.connect(':memory:')
        with sqlite3.connect(temp_db_path) as file_conn:
            file_conn.backup(conn)  # Kopiujemy dane do pamięci

        logging.info("Baza danych zaladowana do pamieci.")
        return conn

    def load_data(self) -> pd.DataFrame:
        """
        info:
            Wczytuje dane z bazy SQLite i zwraca DataFrame.
        return:
            pd.DataFrame - DataFrame z cechami zwierząt
        """
        query = "SELECT * FROM cechy"
        df = pd.read_sql_query(query, self.conn)
        return df

    def train_model(self):
        """
            Trenuje model na danych z bazy SQLite i zapisuje go na dysk
        """
        data = self.load_data()
        self.features = data.columns.drop(['id', 'zwierze'])

        X = data[self.features]
        y = data['zwierze']

        # Uzupełnianie braków medianą
        self.imputer = SimpleImputer(strategy="median")
        X_imputed = self.imputer.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
        
        # Przeprowadzenie Grid Search do znalezienia najlepszych parametrów
        best_model = self.tune_model(X_train, y_train)

        y_pred = best_model.predict(X_test)
        logging.info("Raport klasyfikacji dla najlepszego modelu:\n")
        logging.info(classification_report(y_test, y_pred))

        # Zapis modelu i imputera na dysk
        self.model = best_model

        models_path = os.path.join(self.path, 'models')
        if not os.path.exists(models_path):
            os.makedirs(models_path)
        
        joblib.dump(self.model, os.path.join(models_path, 'animal_features_model.joblib'))
        joblib.dump(self.imputer, os.path.join(models_path, 'animal_features_imputer.joblib'))
        joblib.dump(self.features.tolist(), os.path.join(models_path, 'animal_features_features.joblib'))
        logging.info("Model, imputer i cechy zapisano na dysk.")
        
    def tune_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
        """
        info: 
            Używa GridSearchCV do wyszukania najlepszych parametrów dla Random Forest
        args:
            X_train: pd.DataFrame - Zbiór treningowy cech
            y_train: pd.Series - Zbiór treningowy etykiet
        return:
            RandomForestClassifier - Najlepszy model
        """
        logging.info("Rozpoczynanie Grid Search...")

        rf = RandomForestClassifier(random_state=42)

        # Definicja siatki parametrów
        param_grid = {
            'n_estimators': [50, 100, 200],       # liczba drzew
            'max_depth': [10, 20, 30],            # maksymalna głębokość drzewa
            'min_samples_split': [2, 5, 10],      # minimalna liczba próbek do podziału w węźle
            'min_samples_leaf': [1, 2, 4],        # minimalna liczba próbek w liściu
            'max_features': ['sqrt', 'log2'],     # liczba cech do rozważenia przy każdym podziale
        }

        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=5,               # 5-krotna walidacja krzyżowa
            n_jobs=-1,          # Wykorzystanie wszystkich procesorów
            scoring='accuracy', # Metryka do optymalizacji
        )

        grid_search.fit(X_train, y_train)

        logging.info(f"Najlepsze parametry: {grid_search.best_params_}")
        logging.info(f"Najlepsza dokladnosc walidacji krzyzowej: {grid_search.best_score_}")

        return grid_search.best_estimator_

    def predict(self, input_features: dict) -> str:
        """
        info:
            Przewiduje etykietę zwierzęcia na podstawie podanych cech.
            Obsługuje brakujące cechy.
        args:
            input_features: dict - Słownik z cechami np. {"Lojalność": 50, "Towarzyskość": 60}
        return:
            str - Nazwa zwierzęcia
        """
        if self.model is None or self.imputer is None:
            logging.critical("Model i imputer musza zostac wczytane lub wytrenowane.")
        if self.features is None:
            logging.critical("Lista cech modelu nie zostala wczytana.")
        
        input_vector = pd.DataFrame([input_features])

        # pominięcie nieznanych cech
        for feature in input_vector.columns:
            if feature not in self.features:
                input_vector.drop(columns=feature, inplace=True)
                logging.warning(f"Nieznana cecha: {feature}")

        # uzupełnienie brakujących cech medianą
        for feature in self.features:
            if feature not in input_vector.columns:
                input_vector[feature] = None
        input_vector_imputed = self.imputer.transform(input_vector)

        prediction = self.model.predict(input_vector_imputed)[0]
        return prediction

