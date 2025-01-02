import sys
import os

from src.logger import logging
from src.exception import CustonmException
from src.utill import save_object,evaluate_models

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor

from dataclasses import dataclass

@dataclass
class modelTrainerConfig:
    trained_model_path = os.path.join("artifacts","model.pkl")

class modelTrainer:
    def __init__(self):
        self.model_train_config = modelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            x_train, y_train, x_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            
            models ={
                "LinearRegressor" : LinearRegression(),
                "DecisionTreee": DecisionTreeRegressor(),
                "KNeighbour" : KNeighborsRegressor(),
                "RandomForest": RandomForestRegressor(),
                "AdaBoostRegressor" : AdaBoostRegressor(),
                "GradientBoostingRegressor" : GradientBoostingRegressor()
            }

            logging.info("model training initiated")

            model_report:dict = evaluate_models(x_train,y_train,x_test,y_test,models)


            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            if best_model_score < 0.6:
                raise CustonmException("No best model found")
            logging.info(f"Best model :{best_model_name} score :{best_model_score}")

            best_model = models[best_model_name]
            save_object(obj=best_model, file_path=self.model_train_config.trained_model_path)
        except Exception as e:
            raise CustonmException(e,sys)