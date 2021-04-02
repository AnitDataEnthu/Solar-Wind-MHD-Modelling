from pyhdf.SD import *
import numpy as np
import matplotlib.pyplot as plt
import os

print(os.getcwd())


def readfile(path,start,end):
    sequence=[]
    count=end - start
    for i in range(start,end):
        thispath=path + "br002_" + str(i) + ".hdf"
        # print("Reading Files :",thispath)
        image_sequence=SD(thispath,SDC.READ)
        sds_obj=image_sequence.select('Data-Set-2')
        dim3=sds_obj.get()
        frame=[]
        for i in range(0,141):
            frame.append(dim3[:,:,i])
        frame=np.array(frame)
        sequence.append(frame)

    sequence=np.array(sequence)
    data=sequence.reshape(count,141,128,110,1)
    return data


def scale(data):
    from sklearn.preprocessing import minmax_scale
    shape=data.shape
    data=minmax_scale(data.ravel(),feature_range=(0,255)).reshape(shape)
    return data


def getPredictedForNextInput(ModelNumber):
    pathToGet='final_outcome/Model-' + str(ModelNumber-1) + '/'
    inputSequence=[]
    for i in range(0,5):
        readImagePath=pathToGet + 'predicted_sample-' + str(i) + '.npy'
        image=np.load(readImagePath)
        inputSequence.append(image)
    return np.array(inputSequence).reshape(1,5,128,110,1)

def getIntrestingFrames(ModelNumber,data,frame_start,frame_end):
    print('todelete shape of data: ',data.shape)
    short_data=np.zeros((data.shape[0],5,128,110,1))  # tf.v1
    for file in range(data.shape[0]):
        #short_data[file,0:5]=data[file,frame_start:frame_start + 5,:,:,:]  # tf.v1
        short_data[file,0:5]=data[file,frame_start + 5:frame_end + 1,:,:,:]  # tf.v1
    outputSequence=np.array(short_data)
    inputSequence=getPredictedForNextInput(ModelNumber)
    return np.concatenate((inputSequence, outputSequence), axis=1)


def getIntrestingFrames_susequent_frames(data,samp_no):
    data_i=[]
    for i in range(0,135,5):
        short_data=np.zeros((1,10,128,110,1))
        x_ip=data[samp_no,i:i + 5,:,:,:]
        x_op=data[samp_no,i + 5:i + 10,:,:,:]
        # print("x_in",i,i+5,x.shape)
        # print("x_op",i+5,i+10,x.shape)
        short_data[0,0:5,:,:,:]=x_ip
        short_data[0,5:10,:,:,:]=x_op
        data_i.append(short_data)
    data_np=np.array(data_i)
    data_np=data_np.reshape(27,10,128,110,1)
    data_np_fin=data[samp_no,131:141,:,:,:]
    data_np=np.append(data_np,data_np_fin)
    data_np=data_np.reshape(28,10,128,110,1)
    return data_np


def printAP(a,d,n):
    curr_term=a
    series=[]
    for i in range(1,n + 1):
        series.append(int(curr_term))
        curr_term=int(curr_term + d)
    return series


def createPatchyData(data,window,stride):
    image_size=(data.shape[2],data.shape[3])
    # window=(32,32)
    # stride=(3,3)
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


def createPatchyData_no_overlap(data,window):
    image_size=(data.shape[2],data.shape[3])
    # window=(32,32)
    # stride=(3,3)
    stride=window
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


def createData(start,end,sequence_length,path,frame_start,frame_end,ModelNumber):
    data=readfile(path,start,end)  # read the data with the given count
    print("After readfile:",data.shape)
    data_f=getIntrestingFrames(ModelNumber,data,frame_start,frame_end)  # select 0-4 and 135-141 images
    print("After selectInterestingFrames:",data_f.shape)
    f_data=data_f
    s_data=scale(f_data)
    fin_Data=createPatchyData_no_overlap(s_data,(64,64))  # to create 64 x 64 images
    print("After createPatchyData: ",fin_Data.shape)
    gen_images=np.transpose(fin_Data,[0,1,4,2,3])
    final_data=gen_images.reshape(gen_images.shape[0] * 10,1,64,64)
    print("After transpose: ",final_data.shape)

    myclip=np.zeros((2,fin_Data.shape[0],2),int)
    a=0  # starting number
    d=sequence_length  # Common difference
    n=fin_Data.shape[0]  # N th term to be find

    x=printAP(a,d,n)
    myclip[0,:,0]=x

    a=5  # starting number
    d=sequence_length  # Common difference
    n=fin_Data.shape[0]  # N th term to be find
    print(fin_Data.shape[0])
    y=printAP(a,d,n)
    myclip[1,:,0]=y
    myclip[0,:,1]=5
    myclip[1,:,1]=5
    print("Shape of the clip file : ",myclip.shape)
    print("############ To check the my clip and validate from mnist datset : ",myclip.shape,"############")
    print('myclips[0,:,0]',myclip[0,:,0])
    print('myclips[1,:,0]',myclip[1,:,0])
    print('myclips[0,:,1]',myclip[0,:,1])
    print('myclips[1,:,1]',myclip[1,:,1])
    print("############ End of Verification ############")

    dims=np.array((1,64,64),'int32')
    my_dims=dims.reshape(1,3)
    print("Shape of the dims file : ",my_dims.shape)

    return myclip,my_dims,final_data


def createRecursiveTestData(ModelNumber,frame_start,frame_end,valid_data_paths,test_sample=1):
    path=os.getcwd() + "/data/sampleData/"
    print(path)
    sequence_length=10
    clips_valid,dims_valid,input_raw_data_valid=createData(test_sample,test_sample + 1,sequence_length,path,frame_start,
                                                           frame_end,ModelNumber)

    #return createData(test_sample,test_sample + 1,sequence_length,path,frame_start,frame_end,ModelNumber)
    valid_data_location=os.getcwd() + "/" + valid_data_paths
    np.savez(valid_data_location,clips=clips_valid,dims=dims_valid,
             input_raw_data=input_raw_data_valid)
    # # print("Load arrays from the 'PSI-valid.npz' file:")
    # # with np.load(valid_data_location) as data:
    # #     x2 = data['clips']
    # #     y2 = data['dims']
    # #     z2 = data['input_raw_data']
    # #     print(x2.shape)
    # #     print(y2.shape)
    # #     print(z2.shape)

# create_train_data(0,9,"/data/PSI/model-1/PSI-valid.npz")