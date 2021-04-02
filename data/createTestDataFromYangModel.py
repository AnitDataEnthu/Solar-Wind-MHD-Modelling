from pyhdf.SD  import *
import numpy as np 
import matplotlib.pyplot as plt

def readData(predicted_samples,original_samples):
    
    predicted=np.load(predicted_samples)
    print("shape of the predicted samples",predicted.shape)

    original=np.load(original_samples)
    print("shape of the original samples",original.shape)
    return len(original),predicted,original

def getIntrestingFrame(num_samples,sequence_length,predicted,original):    
    f_data=np.zeros((num_samples,sequence_length,128,110))
    f_data[:,0:5,:,:]=predicted[:,:,:,:]
    f_data[:,5:10,:,:]=original[:,136:141,:,:]
    print("shape of the intresting frames",f_data.shape)
    return f_data.reshape(num_samples,sequence_length,128,110,1)
def scale(data):
    from sklearn.preprocessing import minmax_scale
    shape = data.shape
    data = minmax_scale(data.ravel(), feature_range=(0,255)).reshape(shape)
    return data

def printAP(a,d,n): 
    curr_term=a 
    series=[]
    for i in range(1,n+1): 
        series.append(int(curr_term))
        curr_term =int(curr_term + d)
    return series

def createPatchyData_no_overlap(data,window):
    image_size=(data.shape[2],data.shape[3])
    #window=(32,32)
    #stride=(3,3)
    stride = window
    new_data=[]
    frame_num=-1
    for file in range(data.shape[0]):
        h_stop=0
        for h in range(0,image_size[0],stride[0]):
            if h_stop:
                break
            if h + window[0] >= image_size[0]:
                h=image_size[0] - window[0]
                h_stop=1
            v_stop=0
            for v in range(0,image_size[1],stride[1]):
                if v_stop:
                    break
                if v + window[1] >= image_size[1]:
                    v=image_size[1] - window[1]
                    v_stop=1
                frame_num+=1
                new_data.append(data[file,:,h:h + window[0],v:v + window[1],:])
    new_data=np.array(new_data)
    return new_data

def createTestDataFromModel1(predicted_samples,original_samples,sequence_length,sample_number):

    num_samples,predicted,original=readData(predicted_samples,original_samples)
    f_data=getIntrestingFrame(num_samples,sequence_length,predicted,original)
    #s_data=scale(f_data)
    s_data=f_data
    print("shape of the data before patching",s_data.shape)
    fin_Data=createPatchyData_no_overlap(s_data[sample_number].reshape(1,10,128,110,1),(64,64))
    print("shape of the data after patching and selecting sample_number",fin_Data.shape)


    gen_images = np.transpose(fin_Data, [0,1,4,2,3])
    final_data=gen_images.reshape(gen_images.shape[0]*10,1,64,64)

    myclip=np.zeros((2,fin_Data.shape[0],2),int)
    a = 0 # starting number 
    d = sequence_length # Common difference 
    n = fin_Data.shape[0] # N th term to be find 

    x=printAP(a, d, n)
    myclip[0,:,0]=x

    a = 5 # starting number 
    d = sequence_length # Common difference 
    n = fin_Data.shape[0] # N th term to be find 
    print(fin_Data.shape[0])
    y=printAP(a, d, n)
    myclip[1,:,0]=y
    myclip[0,:,1]=5
    myclip[1,:,1]=5
    print("Shape of the clip file : ",myclip.shape)
    print("############ To check the my clip and validate from mnist datset : ",myclip.shape , "############")
    print('myclips[0,:,0]',myclip[0,:,0])
    print('myclips[1,:,0]',myclip[1,:,0])
    print('myclips[0,:,1]',myclip[0,:,1])
    print('myclips[1,:,1]',myclip[1,:,1])
    print("############ End of Verification ############")

    dims=np.array((1,64,64),'int32')
    my_dims=dims.reshape(1, 3)
    print("Shape of the dims file : ",my_dims.shape)

    return myclip,my_dims,final_data

def createTestDataFromYangModel(size,sample_number):
    predicted_samples='data/15_samples_5_frames_prediction.npy'
    original_samples='data/15_samples_140_frames_real_new.npy'
    sequence_length=10
    sample_number=sample_number
    #'clips', 'dims', 'input_raw_data'
    path="sampleData/"


    clips_valid,dims_valid,input_raw_data_valid=createTestDataFromModel1(predicted_samples,original_samples,sequence_length,sample_number)

    np.savez('data/PSI/PSI-valid.npz', clips=clips_valid, dims=dims_valid,input_raw_data=input_raw_data_valid)
    print("Load arrays from the 'PSI-valid.npz' file:")
    with np.load('data/PSI/PSI-valid.npz') as data:
        x2 = data['clips']
        y2 = data['dims']
        z2 = data['input_raw_data']
        print(x2.shape)
        print(y2.shape)
        print(z2.shape)