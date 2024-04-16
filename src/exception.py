import sys
from src.logger import logging
def error_message_details(error,error_details:sys): # this error_details our system will thorw
    _,_,exc_tb=error_details.exc_info() # 3rd param will have all err details
    file_name=exc_tb.tb_frame.f_code.co_filename # Return the error file name
    error_message="error occured in python script [{0}] line number [{1}] error message [{2}]".format(
        file_name,exc_tb.tb_lineno,str(error)
    )

    return error_message

class CustomException(Exception):
    def __init__(self, error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message=error_message_details(error_message,error_detail)

    def __str__(self) :
        return self.error_message
    
if __name__=="__main__":

    try:
        a=1/0
    except Exception as e:
        logging.info("Divided by zero")
        raise CustomException(e,sys)
