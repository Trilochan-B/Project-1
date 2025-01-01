import os
import sys
from src.exception import CustonmException
from src.logger import logging
import dill

def save_object(obj, file_path):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok= True)
        with open(file_path,"wb") as file :
            dill.dump(obj, file)
    except Exception as e:
        raise CustonmException(e,sys)