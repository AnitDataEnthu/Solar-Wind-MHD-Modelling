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

def getIntrestingFrames(data,frame_start,frame_end):
    print('todelete shape of data: ', data.shape)
    short_data = np.zeros((data.shape[0],10,128,110,1))           #tf.v1
    for file in range(data.shape[0]):
        short_data[file,0:5]=data[file,frame_start:frame_start+5,:,:,:]   #tf.v1
        short_data[file,5:10]=data[file,frame_start+5:frame_end+1,:,:,:]     #tf.v1
    s=np.array(short_data)
    return s


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

def image_from_patches(image_size, window):
    #   image_size = (data.shape[2],data.shape[3])
    #   window = (32,32)
    stride = window # YOU NEED TO GIVE THE STRIDE WHICH WAS GIVEN WHILE CREATING PATCHES, can be other than window
    full_image = np.full(image_size, -1)
    folder=0 # IF THE FOLDER NAME STARTS WITH 1, IF STARTS WITH 0 THEN INITIAL VALUE WILL BE -1
    seq_of_full_images = []
    for output_seq_num in range(1,6): # GIVE THE NUMBER OF SEQUENCE IN OUTPUT
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
                folder+=1
                full_img_patch = full_image[h:h + window[0],v:v + window[1]]
                patch = plt.imread(str(folder)+'/'+str(output_seq_num)+'.jpg')
                output_patch = np.add(full_img_patch, patch, full_img_patch=a>-1)
                output_patch[full_img_patch==-1] = patch[full_img_patch==-1]
                output_patch[full_img_patch==-1] = output_patch[full_img_patch==-1] * 2
                output_patch = output_patch / 2
                full_image[h:h + window[0],v:v + window[1]] = output_patch
        seq_of_full_images.append(full_image)

def createTestData(path,sequence_length,start,end,size):
    data=readfile(path,start,end) #read the data with the given count 
    f_data=getIntrestingFrames(data,frame_start,frame_end)# select 0-4 and 135-141 images
    s_data=scale(f_data)  
    fin_Data=createPatchyData_no_overlap(s_data,(size,size)) # to create 64 x 64 images
    gen_images = np.transpose(fin_Data, [0,1,4,2,3])
    final_data=gen_images.reshape(gen_images.shape[0]*10,1,size,size)
    
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
    
    dims=np.array((1,size,size),'int32')
    my_dims=dims.reshape(1, 3)
    print("Shape of the dims file : ",my_dims.shape)
    
    return myclip,my_dims,final_data

def createtestsample(size,start):
#'clips', 'dims', 'input_raw_data'
    path="data/sampleData/"
    sequence_length=10
    start=start
    end=start+1

    clips_valid,dims_valid,input_raw_data_valid=createTestData(path,sequence_length,start,end,size)

    np.savez('data/PSI/PSI-valid.npz', clips=clips_valid, dims=dims_valid,input_raw_data=input_raw_data_valid)
    print("Load arrays from the 'PSI-valid.npz' file:")
    with np.load('data/PSI/PSI-valid.npz') as data:
        x2 = data['clips']
        y2 = data['dims']
        z2 = data['input_raw_data']
        print(x2.shape)
        print(y2.shape)
        print(z2.shape)
   