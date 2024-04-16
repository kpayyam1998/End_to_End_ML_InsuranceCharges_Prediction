""""
    This file have filtering,Converting categorcial to numerical 
"""

import os
import sys
import numpy as np
import pandas as pd

from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.utills import save_object


@dataclass
class data_transformation_config:
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")

class Data_Tranformation:
    def __init__(self) :
        self.data_transformation_config=data_transformation_config()
    def get_data_transformer_obj(self):
        try:
            """
                This function is responsible for data tranformation based on our data
            """
            numerical_columns=['age', 'bmi', 'children']
            cat_columns=['sex', 'smoker', 'region']

            logging.info("Started Preprocessing and Scaling our numerical and categrocial values")

            """
                if you want to do data filter here do it Before creating pipeline
            """

            #Pipeline which will do multiple task at the sametime
            numerical_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")), #Handling missing values
                    ("scalar",StandardScaler()) # Doing scalar our numerical data
                ]
            )

            categorical_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("encoder",OneHotEncoder()),
                    ("scalar",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Numerical scaling completed{numerical_columns}")
            logging.info(f"Categorical scaling completed{cat_columns}")

            preprocessor=ColumnTransformer(
                [
                    ("numerical",numerical_pipeline,numerical_columns),
                    ("categorcial",categorical_pipeline,cat_columns)
                ]
            )
            logging.info("Done Preprocessing...")

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Reading train and test data is completed")

            logging.info("Preprocessing is started")

            preprocessing_obj=self.get_data_transformer_obj() #it returns preocessor obj

            target_col="charges"

            inuput_feature_train_df=train_df.drop(columns=[target_col],axis=1)
            target_feature_train_df=train_df[target_col]

            inuput_feature_test_df=test_df.drop(columns=[target_col],axis=1)
            target_feature_test_df=test_df[target_col]

            logging.info("Applying preprocessing object on training and testing dataframe")
            
            input_fearture_train_arr=preprocessing_obj.fit_transform(inuput_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(inuput_feature_test_df)

            train_arr=np.c_[
                input_fearture_train_arr,np.array(target_feature_train_df)
            ]
            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]

            logging.info("saving object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
                )
            
            logging.info("Done")
            return (
                train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path
            )
            

        except Exception as e:
            raise CustomException(e,sys)
