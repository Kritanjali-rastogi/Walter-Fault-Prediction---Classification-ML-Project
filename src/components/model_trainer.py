import os
import sys
import pickle
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from src.utils import evaluate_model
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Model Trainer Configuration 

@dataclass
class model_trainer_config:
    model_obj_file_path = os.path.join('artifacts', 'model.pkl')


# Model Trainer Class

class model_trainer:

    def __init__(self):
        self.model_trainer_config = model_trainer_config()

    def initiate_model_training(self, train_arr, test_arr):

        try:
            logging.info("Model training initiated")

            train_input_data = train_arr[:,:-1]
            train_target_data = train_arr[:,-1]

            test_input_data = test_arr[:,:-1]
            test_target_data = test_arr[:,-1]
            
            logging.info("Train and Test data and independent and dependent variables segreggated")
            
            models = {
                'LogisticRegression': LogisticRegression(),
                'DecisionTree': DecisionTreeClassifier(),
                'SVM': SVC(),
                'NaiveBayes': GaussianNB(),
                'RandomForest': RandomForestClassifier(),
                'KNN': KNeighborsClassifier()}   

            params = {
                'LogisticRegression': {},
                'DecisionTree': {},
                'SVM': {},
                'NaiveBayes': {},
                'RandomForest': {},
                'KNN': {}}
            
            
            model_report: dict = evaluate_model(train_input_data, train_target_data,test_input_data,
                                                test_target_data, models, params)
            

            best_modal_score = max(sorted(model_report.values()))
            best_modal_name = list(model_report.keys())[list(model_report.values()).index(best_modal_score)]
            best_modal = models[best_modal_name]

            logging.info(f'Best model found: {best_modal_name} with accuracy of {best_modal_score}')
            print(f'Best model found: {best_modal_name}')

            save_object(file_path=self.model_trainer_config.model_obj_file_path, object=best_modal)

            logging.info('Model pickle file created')
            
            best_modal.fit(train_input_data, train_target_data)
            y_pred = best_modal.predict(test_input_data)
            accuracy = accuracy_score(test_target_data, y_pred)
            class_report = classification_report(test_target_data, y_pred)
            con_mat = confusion_matrix(test_target_data, y_pred)

            print(f"Model accuracy score is {accuracy}")
            print(class_report)
            print(con_mat)

            return accuracy
        
        except Exception as e:
            logging.info("Error occured in model trainer")
            raise CustomException(e,sys)


