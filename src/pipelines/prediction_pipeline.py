import os
import sys
import pickle
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):

        preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
        model_path = os.path.join('artifacts', 'model.pkl')

        preprocessor_obj = load_object(preprocessor_path)
        model_obj = load_object(model_path)

        scaled_data = preprocessor_obj.transform(features)
        prediction = model_obj.predict(scaled_data)

        return prediction
    

class CustomData:

    def __init__(self, age,sex, cp, trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal):
        self.age = age
        self.sex = sex
        self.cp = cp
        self.trestbps = trestbps
        self.chol = chol
        self.fbs = fbs
        self.restecg = restecg
        self.thalach = thalach
        self.exang = exang
        self.oldpeak = oldpeak
        self.slope = slope
        self.ca = ca
        self.thal = thal

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {'age': [self.age],
                                 'sex': [self.sex], 
                                 'cp': [self.cp], 
                                 'trestbps': [self.trestbps],
                                 'chol': [self.chol],
                                 'fbs': [self.fbs],
                                 'restecg': [self.restecg],
                                 'thalach': [self.thalach],
                                 'exang': [self.exang],
                                 'oldpeak':[self.oldpeak],
                                 'slope': [self.slope],
                                 'ca':[self.ca],
                                 'thal':[self.thal]}
            
            return pd.DataFrame(custom_data_input_dict)
        

        except Exception as e:
            raise CustomException(e,sys)


