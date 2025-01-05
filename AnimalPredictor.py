import logging
import numpy as np
from AnimalFeaturesClassifier import AnimalFeaturesClassifier
from AnimalImageClassifier import AnimalImageClassifier

class AnimalPredictor:
    def __init__(self, features_classifier: AnimalFeaturesClassifier, image_classifier: AnimalImageClassifier, logger: logging.Logger):
        """
        Inicjalizacja połączonego klasyfikatora zwierząt.
        args:
            features_classifier: AnimalFeaturesClassifier - Klasyfikator oparty na cechach
            image_classifier: AnimalImageClassifier - Klasyfikator oparty na obrazach
            logger: logging.Logger - Logger do logowania informacji
        """
        self.features_classifier = features_classifier
        self.image_classifier = image_classifier
        self.logger = logger
        self.logger.info("Inicjalizacja połączonego klasyfikatora zwierząt.")

    def combine_predictions(self, features_predictions, image_predictions, weight_image=0.7, weight_features=0.3):
        """
        Łączy wyniki obu klasyfikatorów według zadanego wagi.
        args:
            features_predictions: list - Lista [(zwierzę, prawdopodobieństwo)] z klasyfikatora cech
            image_predictions: list - Lista [(zwierzę, prawdopodobieństwo)] z klasyfikatora obrazów
            weight_image: float - Waga dla predykcji obrazów
            weight_features: float - Waga dla predykcji cech
        return:
            list - Lista 5 najlepszych przewidywań połączonych
        """
        combined_scores = {}

        # Sumowanie zważonych wyników z predykcji cech i obrazów
        for animal, score in image_predictions:
            combined_scores[animal] = combined_scores.get(animal, 0) + score * weight_image

        for animal, score in features_predictions:
            combined_scores[animal] = combined_scores.get(animal, 0) + score * weight_features

        # Sortowanie według złączonych wyników i pobranie top 5
        top_5_combined = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:5]

        self.logger.info(f"Top 5 połączonych przewidywań: {top_5_combined}")
        return top_5_combined

    def predict_top_5(self, image_path: str = None, input_features: dict = None) -> list:
        """
        Przewiduje 5 najbardziej prawdopodobnych zwierząt, łącząc klasyfikację obrazową i cechową.
        args:
            image_path: str - Ścieżka do zdjęcia (opcjonalnie)
            input_features: dict - Słownik cech zwierzęcia (opcjonalnie)
        return:
            list - Lista 5 najbardziej prawdopodobnych zwierząt
        """
        if image_path:
            image_predictions = self.image_classifier.predict_top_10(image_path)
        else:
            image_predictions = []

        if input_features:
            features_predictions = self.features_classifier.predict_top_10(input_features)
        else:
            features_predictions = []

        # Jeśli brakuje danych do jednej z klasyfikacji, użyj tylko dostępnych predykcji
        if not image_predictions:
            return features_predictions[:5]
        if not features_predictions:
            return image_predictions[:5]

        # Łącz predykcje, dając większą wagę klasyfikatorowi obrazów
        return self.combine_predictions(features_predictions, image_predictions)
