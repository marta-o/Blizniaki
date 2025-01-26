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
    def __init__(self, drive_file_id : str, local_path: str, logger: logging.Logger):
        """
        Inicjalizacja klasyfikatora obrazów.
        args:
            drive_file_id: str - Id pliku na Google Drive
            local_path: str - Lokalna ścieżka do zapisu danych
            logger: logging.Logger - Wspólny logger
        """
        self.drive_file_id = drive_file_id
        self.path = local_path
        self.model = None
        self.imputer = None
        self.features = None
        self.logger = logger

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.logger.info("Inicjalizacja klasyfikatora cech zwierząt.")

        try:
            self.model = joblib.load(os.path.join(self.path, 'models', 'animal_features_model.joblib'))
            self.imputer = joblib.load(os.path.join(self.path, 'models', 'animal_features_imputer.joblib'))
            self.features = joblib.load(os.path.join(self.path, 'models', 'animal_features_features.joblib'))
            self.logger.info("Model wczytano pomyślnie.")
        except FileNotFoundError:
            self.logger.info("Model nie istnieje. Należy go wytrenować.")
            try:
                self.conn = self.load_data_from_drive()
                self.train_model()
            except Exception as e:
                self.logger.critical("Nie udało się wytrenować modelu: %s", str(e))
                raise RuntimeError(f"Błąd inicjalizacji: {e}")

    def load_data_from_drive(self):
        """
        Pobiera bazę danych SQLite z Google Drive.
        """
        try:
            self.logger.info("Pobieranie bazy danych z Google Drive...")
            download_url = f"https://drive.google.com/uc?id={self.drive_file_id}&export=download"
            response = requests.get(download_url, timeout=60)
            response.raise_for_status()  # Wywołuje wyjątek dla kodów błędów HTTP
        except requests.exceptions.Timeout:
            self.logger.error("Przekroczono limit czasu podczas próby połączenia z Google Drive.")
            raise ConnectionError("Przekroczono limit czasu podczas próby połączenia z Google Drive.")
        except requests.exceptions.RequestException as e:
            self.logger.error("Wystąpił błąd podczas pobierania danych: %s", str(e))
            raise ConnectionError(f"Błąd pobierania danych: {e}")

        db_bytes = io.BytesIO(response.content)
        temp_db_path = os.path.join(self.path, 'animal_db.sqlite')

        with open(temp_db_path, 'wb') as temp_file:
            temp_file.write(db_bytes.read()) 

        conn = sqlite3.connect(':memory:') 
        with sqlite3.connect(temp_db_path) as file_conn:
            file_conn.backup(conn) 

        self.logger.info("Baza danych załadowana do pamięci.")
        return conn

    def load_data(self) -> pd.DataFrame:
        """
        Wczytuje dane z bazy SQLite i zwraca DataFrame.
        return:
            pd.DataFrame - DataFrame z cechami zwierząt
        """
        try:
            query = "SELECT * FROM cechy"
            df = pd.read_sql_query(query, self.conn)

            if df.empty:
                self.logger.error("Baza danych nie zawiera żadnych danych.")
                raise ValueError("Baza danych jest pusta.")
            
            self.logger.info("Dane załadowano poprawnie. Liczba wierszy: %d", len(df))
            return df
        except sqlite3.DatabaseError as e:
            self.logger.critical("Błąd podczas wczytywania danych z bazy SQLite: %s", str(e))
            raise sqlite3.DatabaseError(f"Błąd podczas wczytywania danych: {e}")

    def train_model(self):
        """
        Trenuje model na danych z bazy SQLite i zapisuje go lokalnie.
        """
        data = self.load_data()
        self.features = data.columns.drop(['id', 'zwierze'])

        X = data[self.features]
        y = data['zwierze']

        self.imputer = SimpleImputer(strategy="median") # Uzupełnianie braków medianą
        X_imputed = self.imputer.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
        
        if X_train.shape[0] == 0 or len(y_train) == 0:
            self.logger.critical("Zbiór treningowy jest pusty. Nie można wytrenować modelu.")
            raise ValueError("Zbiór treningowy jest pusty. Sprawdź dane wejściowe.")
        
        best_model = self.tune_model(X_train, y_train) # Przeprowadzenie Grid Search do znalezienia najlepszych parametrów

        if not best_model:
            self.logger.critical("GridSearchCV nie zwrócił modelu. Trening nie powiódł się.")
            raise RuntimeError("Trening modelu nie powiódł się.")
        
        y_pred = best_model.predict(X_test)
        self.logger.info("Raport klasyfikacji dla najlepszego modelu:\n")
        self.logger.info(classification_report(y_test, y_pred))

        self.model = best_model

        models_path = os.path.join(self.path, 'models')
        if not os.path.exists(models_path):
            os.makedirs(models_path)
        
        joblib.dump(self.model, os.path.join(models_path, 'animal_features_model.joblib'))
        joblib.dump(self.imputer, os.path.join(models_path, 'animal_features_imputer.joblib'))
        joblib.dump(self.features.tolist(), os.path.join(models_path, 'animal_features_features.joblib'))
        self.logger.info("Model, imputer i cechy zapisano lokalnie.")
        
    def tune_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
        """
        Używa GridSearchCV do wyszukania najlepszych parametrów dla Random Forest
        args:
            X_train: pd.DataFrame - Zbiór treningowy cech
            y_train: pd.Series - Zbiór treningowy etykiet
        return:
            RandomForestClassifier - Najlepszy model
        """
        self.logger.info("Rozpoczynanie Grid Search...")
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

        self.logger.info(f"Najlepsze parametry: {grid_search.best_params_}")
        self.logger.info(f"Najlepsza dokladnosc walidacji krzyzowej: {grid_search.best_score_}")

        return grid_search.best_estimator_    

    def predict_top_10(self, input_features: dict) -> list:
        """
        Przewiduje 10 najbardziej prawdopodobnych zwierząt na podstawie cech.
        args:
            input_features: dict - Słownik z cechami np. {"Lojalność": 50, "Towarzyskość": 60}
        return:
            list - Lista 10 zwierząt z prawdopodobieństwami
        """
        if self.model is None or self.imputer is None:
            self.logger.critical("Model i imputer muszą zostać wczytane lub wytrenowane.")
        if self.features is None:
            self.logger.critical("Lista cech modelu nie zostala wczytana.")
        
        if not isinstance(input_features, dict):
            self.logger.error("Dane wejściowe muszą być słownikiem. Otrzymano: %s", type(input_features))
            raise ValueError("Dane wejściowe muszą być słownikiem z cechami zwierzęcia.")

        if not input_features:
            self.logger.error("Słownik cech jest pusty.")
            raise ValueError("Słownik cech nie może być pusty.")

        for key, value in input_features.items():
            if not isinstance(value, (int, float)):
                self.logger.warning("Nieprawidłowa wartość cechy %s: %s (typ: %s). Oczekiwano liczby.", key, value, type(value))
                raise ValueError(f"Nieprawidłowa wartość cechy '{key}': {value}. Oczekiwano liczby.")

        input_vector = pd.DataFrame([input_features])
        input_vector = input_vector.loc[:, input_vector.columns.isin(self.features)]

        # Dodawanie brakujących cech
        for feature in self.features:
            if feature not in input_vector.columns:
                input_vector[feature] = None
        
        input_vector = input_vector[self.features]
        input_vector_imputed = self.imputer.transform(input_vector)

        # Przewidywanie prawdopodobieństw
        probabilities = self.model.predict_proba(input_vector_imputed)[0]
        classes = self.model.classes_

        # Tworzenie listy zwierząt z prawdopodobieństwami
        predictions = sorted(zip(classes, probabilities), key=lambda x: x[1], reverse=True)
        top_10_predictions = predictions[:10]

        self.logger.info(f"Top 10 przewidywań: {top_10_predictions}")
        return top_10_predictions

