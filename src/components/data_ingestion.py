import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from src.components.data_transformation import Data_Tranformation,data_transformation_config

#from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import Model_trainer

@dataclass
class Data_Ingestion_Config():
    train_data_path:str=os.path.join("artifacts","train.csv")
    test_data_path:str=os.path.join("artifacts","test.csv")
    raw_data_path:str=os.path.join("artifacts","data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=Data_Ingestion_Config()
    
    def initiate_data_ingestion(self):
        logging.info(" data ingestion is started")
        try:
            df=pd.read_csv("../../notebook/data/insurance.csv")
            logging.info("data loaded as dataframe")
            # create directory
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train Test spilit started")
            
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Data ingestion is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )


        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj= DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    data_transformation=Data_Tranformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    model_trainer=Model_trainer()
    r2_score=model_trainer.initiate_model_trainer(train_arr,test_arr)
    print(r2_score)
