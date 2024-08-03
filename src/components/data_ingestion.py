import os
import sys
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass

# Data Ingestion Configuration

@dataclass
class data_ingestion_configuration:
    train_data_path = os.path.join('artifacts', 'train.csv')
    test_data_path = os.path.join('artifacts', 'test.csv')
    raw_data_path = os.path.join('artifacts', 'raw.csv')

# Data Ingestion 

class data_ingestion:
    
    def __init__(self):
        self.ingestion_config = data_ingestion_configuration()

    def initiate_data_ingestion(self):
        try:
            logging.info("Data ingestion process started")

            df = pd.read_csv(os.path.join('notebooks\data','heart.csv'))

            logging.info("Data read as pandas datafrane")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok= True)

            df.to_csv(self.ingestion_config.raw_data_path, header= True, index= False)

            logging.info('Train Test Split Started')

            train_set, test_set = train_test_split(df, test_size = 0.20, random_state= 42)

            train_set.to_csv(self.ingestion_config.train_data_path, header = True, index = False)
            test_set.to_csv(self.ingestion_config.test_data_path, header = True, index = False)

            logging.info("Train Test Split Complete")

            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)
        
        except Exception as e:
            logging.info("Error occured in data ingestion")
            raise CustomException(e,sys)
        
if __name__=='__main__':
    ingestion_obj = data_ingestion()
    train_data_path, test_data_path = ingestion_obj.initiate_data_ingestion()
    print(train_data_path, test_data_path)