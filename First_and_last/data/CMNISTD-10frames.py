from pyhdf.SD  import *
import numpy as np 
import matplotlib.pyplot as plt

def readfile(path,start,end):
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
    return data

def scale(data):
    from sklearn.preprocessing import minmax_scale
    shape = data.shape
    data = minmax_scale(data.ravel(), feature_range=(0,255)).reshape(shape)
    return data

def all_visual(which,data):
    length=data.shape[1]
    for i in range(length):
        # create plot
        fig=plt.figure(figsize=(10,10))
        ax=fig.add_subplot(121)
        ax.text(1,-3,('true tframe:' + str(i)),fontsize=20,color='b')
        toplot_true=data[which,i,:,:,0]
        plt.imshow(toplot_true)
        
        
        
def getIntrestingFrames(data):
    print('todelete shape of data: ', data.shape)
    short_data = np.zeros((data.shape[0],10,128,110,1))           #tf.v1
    for file in range(data.shape[0]):
        short_data[file,0:5]=data[file,0:5,:,:,:]   #tf.v1
        short_data[file,5:10]=data[file,136:141,:,:,:]     #tf.v1
    s=np.array(short_data)
    return s


def createPatchyData(data,window,stride):
    image_size=(data.shape[2],data.shape[3])
    #window=(32,32)
    #stride=(3,3)
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

def printAP(a,d,n): 
    curr_term=a 
    series=[]
    for i in range(1,n+1): 
        series.append(int(curr_term))
        curr_term =int(curr_term + d)
    return series

def transformLikeMNIST(start,end,sequence_length,path):

    #count=48
    #sequence_length=20
    #path="sampleData/"
    data=readfile(path,start,end) #read the data with the given count 
    f_data=getIntrestingFrames(data) # select 0-4 and 135-141 images
    print("shape of the data after selecting interesting frames :",f_data.shape)

    s_data=scale(f_data)


    fin_Data=createPatchyData(s_data,(64,64),(4,4)) # to create 64 x 64 images
    #T=fin_Data.reshape(10608,20,64,64,1) #getting rid of the
    gen_images = np.transpose(fin_Data, [0,1,4,2,3])
    final_data=gen_images.reshape(fin_Data.shape[0]*10,1,64,64) # combining all the images into one block
    print("After creating patch data and restructuring the axes :",final_data.shape)

    


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

#'clips', 'dims', 'input_raw_data'
path="sampleData/"
sequence_length=10
start=0
end=45
clips_train,dims_train,input_raw_data_train=transformLikeMNIST(start,end,sequence_length,path)
#all_visual(1,f_data) #Visualize with shape(:,:,:,:,:)

path="sampleData/"
sequence_length=10
start=45
end=48
clips_valid,dims_valid,input_raw_data_valid=transformLikeMNIST(start,end,sequence_length,path)

np.savez('PSI/PSI-train.npz', clips=clips_train, dims=dims_train,input_raw_data=input_raw_data_train)
print("Load arrays from the 'PSI-train.npz' file:")
with np.load('PSI/PSI-train.npz') as data:
    x2 = data['clips']
    y2 = data['dims']
    z2 = data['input_raw_data']
    print(x2.shape)
    print(y2.shape)
    print(z2.shape)
    
    
np.savez('PSI/PSI-valid.npz', clips=clips_valid, dims=dims_valid,input_raw_data=input_raw_data_valid)
print("Load arrays from the 'PSI-valid.npz' file:")
with np.load('PSI/PSI-valid.npz') as data:
    x2 = data['clips']
    y2 = data['dims']
    z2 = data['input_raw_data']
    print(x2.shape)
    print(y2.shape)
    print(z2.shape)
