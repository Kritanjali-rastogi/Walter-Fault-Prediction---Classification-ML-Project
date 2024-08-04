import os
import sys
import pickle
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass

# Data Transformation Class Config

@dataclass
class data_transformation_config:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

# Data Transformation class

class data_transformation:
    def __init__(self):
        self.data_transformation_config = data_transformation_config()

    def get_preprocessor_obj(self):

        try:

            logging.info("Data Transformation Started")

            numerical_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
                     'exang', 'oldpeak', 'slope', 'ca', 'thal']
            
            logging.info("Columns segreegated into numerical and categorical")

            numerical_pipeline = Pipeline(steps = [
                ('Imputer', SimpleImputer(strategy= 'median')),('Scaler', StandardScaler())])
            
            logging.info("Numerical pipeline built")

            preprocessor = ColumnTransformer([('Numerical_pipeline', numerical_pipeline, numerical_columns)])

            return preprocessor
        
            logging.info("Preprocessor object obtained")
        
        except Exception as e:
            logging.info("Error occured in getting preprocessor object")
            raise CustomException(e,sys)
        
    def inititate_data_transformation(self, train_data_path, test_data_path):

        try:
            logging.info("Initiating data transformation using preprocessor")

            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info("Train and test data read as dataframe")
            logging.info(f'Train Data Frame: {train_df.head().to_string()}')
            logging.info(f'Train Data Frame: {test_df.head().to_string()}')

            
            train_df_independent_features = train_df.drop(columns=['target'],axis=1)
            train_df_target = train_df['target']

            test_df_independent_features = test_df.drop(columns=['target'],axis=1)
            test_df_target = test_df['target']

            logging.info("Input and target separated for train and test dataframe")

            preprocessor_obj = self.get_preprocessor_obj()

            train_independent_arr = preprocessor_obj.fit_transform(train_df_independent_features)
            test_independent_arr = preprocessor_obj.transform(test_df_independent_features)

            logging.info("preprocessing applied on train and test dataset")

            train_arr = np.c_[train_independent_arr, np.array(train_df_target)]
            test_arr = np.c_[test_independent_arr, np.array(test_df_target)]
            
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path, object=preprocessor_obj)

            logging.info("preprocessor object saved")

            return (train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path)
        
        except Exception as e:
            logging.info("Error occured in getting preprocessor object")
            raise CustomException(e,sys)
        




