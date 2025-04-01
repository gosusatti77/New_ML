import os 
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,AdaBoostClassifier)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsClassifier
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    """
    Model Trainer Configuration
    """
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.models = {
            'RandomForest': RandomForestRegressor(),
            'GradientBoosting': GradientBoostingRegressor(),
            'DecisionTree': DecisionTreeRegressor(),
            'XGBoost': XGBRegressor(),
            'CatBoost': CatBoostClassifier(verbose=0),
            'KNeighbors': KNeighborsClassifier(),
            'AdaBoost': AdaBoostClassifier()
        }
    
    def initiate_model_trainer(self, train_array, test_array):
        """
        This function is responsible for model training.
        """
        try:
            logging.info("Splitting training and testing input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, self.models) 
            

            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = self.models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")

            
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            prediction = best_model.predict(X_test)
            r2_square = r2_score(y_test, prediction)
            return r2_square
        except Exception as e:
            raise CustomException(e, sys) from e