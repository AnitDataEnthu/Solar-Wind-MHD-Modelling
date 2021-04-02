import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.measure import compare_ssim
import pandas as pd
from pyhdf.SD  import *
print(os.getcwd())

def readfile_br(path,start,end):
    sequence=[]
    count=end-start
    for i in range(start,end):
        thispath=path+"br002_"+str(i)+".hdf"
        #print("Reading Files :",thispath)
        image_sequence = SD(thispath, SDC.READ)
        sds_obj = image_sequence.select('Data-Set-2')
        dim3 = sds_obj.get()
        frame=[]
        for i in range(0,141):
            frame.append(dim3[:,:,i])
        frame=np.array(frame)
        sequence.append(frame)
    sequence=np.array(sequence)
    data=sequence.reshape(count,141,128,110,1)
    #data=pad_the_frame(data)
    return data

def scale(data):
    from sklearn.preprocessing import minmax_scale
    shape = data.shape
    data = minmax_scale(data.ravel(), feature_range=(0,255)).reshape(shape)
    return data


def draw_ssim_graph(data):
    ssim=[]
    for i in range(0,140):
        ssim.append(compare_ssim(data[0,1,:,:,0],data[0,i + 1,:,:,0],multichannel=False))
    return ssim


def generatecsv(sample_num):
    start=1
    end=sample_num+1
    sequence_length=10
    path="data/sampleData/"
    frame_start=0
    frame_end=141
    data=scale(readfile_br(path,start,end))
    pd.DataFrame(draw_ssim_graph(data),columns=['SSIM'] ).to_csv("ssim_graph.csv",index=False)

generatecsv(1)