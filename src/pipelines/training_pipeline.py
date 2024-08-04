import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import data_ingestion
from src.components.data_transformation import data_transformation
from src.components.model_trainer import model_trainer

if __name__=='__main__':
    ingestion_obj = data_ingestion()
    train_data_path, test_data_path = ingestion_obj.initiate_data_ingestion()
    print(train_data_path, test_data_path)

    transformation_obj = data_transformation()
    train_arr, test_arr, _ = transformation_obj.inititate_data_transformation(train_data_path, test_data_path) 

    model_trainer_obj = model_trainer()
    accuracy_score = model_trainer_obj.initiate_model_training(test_arr, test_arr)