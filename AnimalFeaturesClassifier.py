import io
import requests
import sqlite3
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
import joblib  # Biblioteka do zapisywania i wczytywania modelu


class AnimalFeaturesClassifier:
    def __init__(self, drive_file_id, path=r'C:\Users\marta\OneDrive - biurox365ml\Pulpit\studia\sem5\inzynieria_oprogramowania'):
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

        # Pobierz bazę danych z Google Drive
        self.conn = self.load_data_from_drive()
        
        try:
            self.model = joblib.load(self.path + r'\animal_features_model.joblib')
            self.imputer = joblib.load(self.path +r'\animal_features_imputer.joblib')
            self.features = joblib.load(self.path + r'\animal_features_features.joblib')
            print("Model wczytano pomyślnie.")
        except FileNotFoundError:
            print("Model nie istnieje. Należy go wytrenować.")

    def load_data_from_drive(self):
        """Pobiera bazę danych SQLite z Google Drive bez zapisywania lokalnie"""
        print("Pobieranie bazy danych z Google Drive...")
        download_url = f"https://drive.google.com/uc?id={self.drive_file_id}&export=download"
        response = requests.get(download_url)
        response.raise_for_status()  # Wyrzuć błąd w przypadku niepowodzenia
        db_bytes = io.BytesIO(response.content)
        
        # Utwórz połączenie z bazą danych w pamięci
        conn = sqlite3.connect(':memory:')  # Tworzymy bazę w pamięci
        with open('temp_db.sqlite', 'wb') as temp_file:
            temp_file.write(db_bytes.read())  # Tymczasowo zapisujemy plik
        with sqlite3.connect('temp_db.sqlite') as file_conn:
            file_conn.backup(conn)  # Kopiujemy dane do pamięci

        print("Baza danych załadowana do pamięci.")
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
        # Wczytanie danych
        data = self.load_data()
        self.features = data.columns.drop(['id', 'zwierze'])

        # Podział na cechy (X) i etykiety (y)
        X = data[self.features]
        y = data['zwierze']

        # Uzupełnianie braków medianą
        self.imputer = SimpleImputer(strategy="median")
        X_imputed = self.imputer.fit_transform(X)

        # Podział na zbiór treningowy i testowy
        X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
        
        # Przeprowadzenie Grid Search do znalezienia najlepszych parametrów
        best_model = self.tune_model(X_train, y_train)

        # Ewaluacja najlepszego modelu
        y_pred = best_model.predict(X_test)
        print("Raport klasyfikacji dla najlepszego modelu:\n")
        print(classification_report(y_test, y_pred))

        # Zapis modelu i imputera na dysk
        self.model = best_model
        joblib.dump(self.model, self.path + r'\animal_features_model.joblib')
        joblib.dump(self.imputer, self.path + r'\animal_features_imputer.joblib')
        joblib.dump(self.features.tolist(), self.path + r'\animal_features_features.joblib')
        print("Model, imputer i cechy zapisano na dysk.")
        
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
        print("Rozpoczynanie Grid Search...")

        # Definiowanie modelu Random Forest
        rf = RandomForestClassifier(random_state=42)

        # Definicja siatki parametrów
        param_grid = {
            'n_estimators': [50, 100, 200],       # liczba drzew
            'max_depth': [10, 20, 30],            # maksymalna głębokość drzewa
            'min_samples_split': [2, 5, 10],      # minimalna liczba próbek do podziału w węźle
            'min_samples_leaf': [1, 2, 4],        # minimalna liczba próbek w liściu
            'max_features': ['sqrt', 'log2'],     # liczba cech do rozważenia przy każdym podziale
        }

        # Inicjalizacja GridSearchCV
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=5,               # 5-krotna walidacja krzyżowa
            n_jobs=-1,          # Wykorzystanie wszystkich procesorów
            scoring='accuracy', # Metryka do optymalizacji
            verbose=2           # Wyświetlanie postępu
        )

        # Dopasowanie GridSearchCV do danych treningowych
        grid_search.fit(X_train, y_train)

        # Wyświetlenie najlepszych parametrów i wyniku
        print(f"Najlepsze parametry: {grid_search.best_params_}")
        print(f"Najlepsza dokładność walidacji krzyżowej: {grid_search.best_score_}")

        # Zwrócenie najlepszego modelu
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
            raise ValueError("Model i imputer muszą zostać wczytane lub wytrenowane.")
        if self.features is None:
            raise ValueError("Lista cech modelu nie została wczytana.")
        
        # Utworzenie wektora wejściowego jako DataFrame
        input_vector = pd.DataFrame([input_features])

        # Dodanie brakujących cech jako kolumn z wartością NaN
        for feature in self.features:
            if feature not in input_vector.columns:
                input_vector[feature] = None

        # Uzupełnienie brakujących wartości za pomocą imputera
        input_vector_imputed = self.imputer.transform(input_vector)

        # Przewidywanie klasy
        prediction = self.model.predict(input_vector_imputed)[0]
        return prediction

