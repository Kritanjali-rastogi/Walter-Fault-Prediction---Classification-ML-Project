import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import data_ingestion

if __name__=='__main__':
    ingestion_obj = data_ingestion()
    train_data_path, test_data_path = ingestion_obj.initiate_data_ingestion()
    print(train_data_path, test_data_path)
