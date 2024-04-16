import os
import logging
from datetime import datetime

LOG_FILE=F"{datetime.now().strftime('%m_%d_%y_%H_%M_%S')}.log"
log_path=os.path.join(os.getcwd(),"logs",LOG_FILE)
os.makedirs(log_path,exist_ok=True) # creating directory

logs_file_path=os.path.join(log_path,LOG_FILE)

logging.basicConfig(
    filename=logs_file_path,
    format="[%(asctime)s] %(lineno)d %(name)s -%(levelname)s -%(message)s",
    level=logging.INFO
)

# if __name__=="__main__":
#     logging.info("Logging started..")


