import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utills import load_object

class PredictPipeline:
    def __init__(self) :
        pass
    def predict(self,features):
        try:
            model_path=os.path.join("src/components/artifacts/","model.pkl")
            preprocessor_path=os.path.join("src/components/artifacts/","preprocessor.pkl")
            logging.info(f"Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            logging.info(f"After loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                age:int,
                sex:str,
            	bmi:int,
                children:int,
                smoker:str,
                region:str,
                ):
            self.age=age
            self.sex=sex
            self.bmi=bmi
            self.children=children
            self.smoker=smoker
            self.region=region
    def get_as_dataframe(self):
        try:
            custom_data_input_dict={    
                "age":[self.age],
                "sex":[self.sex],
                "bmi":[self.bmi],
                "children":[self.children],
                "smoker":[self.smoker],
                "region":[self.region]

            }
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e,sys)

                    

        
     