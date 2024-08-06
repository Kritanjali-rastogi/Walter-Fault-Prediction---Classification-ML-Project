import os
import sys
import pickle
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def save_object(file_path, object):
    try:
        
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok= True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(object, file_obj )

    except Exception as e:
            logging.info("Error occured in dumping pickle file")
            raise CustomException(e,sys)


def evaluate_model(x_train, y_train, x_test, y_test, models, params):

    try:
         report = {}
         for model_name, model in models.items():
              estimator = model
              param = params[model_name]
              gs = GridSearchCV(estimator, param_grid=param, cv=5)
              gs.fit(x_train, y_train)
              gs.set_params(**gs.best_params_)
              gs.fit(x_train, y_train)
              prediction = gs.predict(x_test)
              acc_score = accuracy_score(y_test, prediction)
              report[model_name] = acc_score
         
         return report
    
    except Exception as e:
            logging.info("Error occured in model_evaluation")
            raise CustomException(e,sys)

def load_object(file_path):
     try:
          with open(file_path, 'rb') as file_obj:
               return pickle.load(file_obj)
          
     except Exception as e:
            logging.info("Error occured in loading pickel file")
            raise CustomException(e,sys)