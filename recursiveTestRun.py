import time
import tensorflow as tf

from data.createTestDataRecursiveModels import createRecursiveTestData
from test_run import main as mn
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd




def _createRecursiveTestData(ModelNumber,testSample):
    df=pd.read_csv("model_Details.csv")
    frame_start=int(df[df['model_no'] == ModelNumber]['start'].reset_index(drop=True)[0])
    frame_end=int(df[df['model_no'] == ModelNumber]['end'].reset_index(drop=True)[0])
    valid_data_paths=df[df['model_no'] == ModelNumber]['valid_data_paths'].reset_index(drop=True)[0]
    scaleData=createRecursiveTestData(ModelNumber,frame_start,frame_end,valid_data_paths,int(testSample))
    return scaleData
def main(ModelNumber,testSample,recombination):
    if ModelNumber== 1:
        df=pd.read_csv("model_Details.csv")
        valid_data_paths=df[df['model_no'] == ModelNumber]['valid_data_paths'].reset_index(drop=True)[0]
        if tf.gfile.Exists(valid_data_paths.rsplit('/',1)[0]+'/'):
            tf.gfile.DeleteRecursively(valid_data_paths.rsplit('/',1)[0]+'/')
        tf.gfile.MakeDirs(valid_data_paths.rsplit('/',1)[0]+'/')

        mn(" ",1,testSample,False,recombination)

    elif ModelNumber >= 1: #Actual condition ModelNumber > 1
        df=pd.read_csv("model_Details.csv")
        valid_data_paths=df[df['model_no'] == ModelNumber]['valid_data_paths'].reset_index(drop=True)[0]
        if tf.gfile.Exists(valid_data_paths.rsplit('/',1)[0]+'/'):
            tf.gfile.DeleteRecursively(valid_data_paths.rsplit('/',1)[0]+'/')
        tf.gfile.MakeDirs(valid_data_paths.rsplit('/',1)[0]+'/')
        _createRecursiveTestData(ModelNumber,testSample)
        mn(" ",ModelNumber,testSample,True,recombination)


# if __name__ == '__main__':
#   start_time=time.time()
#   main(3,1)
#   print("--- %s seconds ---" % (time.time() - start_time))
