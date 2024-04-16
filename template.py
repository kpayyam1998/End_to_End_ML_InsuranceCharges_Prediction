import os 
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO,format='[%(asctime)s]:%(message)s')
list_of_file=[
    "src/components/__init__.py",
    "src/components/data_ingestion.py",
    "src/components/data_transformation.py",
    "src/components/model_trainer.py",
    "src/pipeline/__init__.py",
    "src/pipeline/train_pipeline.py",
    "src/pipeline/predict_pipeline.py",
    "src/__init__.py",
    "src/logger.py",
    "src/exception.py",
    "src/utills.py",
    "static/style.css",
    "template/index.html",
    "app.py"
]

for files in list_of_file:
    file_path=Path(files)

    file_dir,file_name=os.path.split(files)

    if file_dir!="":
        os.makedirs(file_dir,exist_ok=True)
        logging.info(f"Creating file directory:{file_dir} for the file :{file_name}")

    if (not os.path.exists(file_path)) or(os.path.getsize(file_path)==0):
        with open(file_path,'w') as f:
            pass
            logging.info(f"Creating empty file:{file_path}")
    else:
        logging.info(f"{file_name} is already created")