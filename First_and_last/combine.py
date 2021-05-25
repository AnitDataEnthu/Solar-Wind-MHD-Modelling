import numpy as np
import matplotlib.pyplot as plt
import copy
import tensorflow as tf



def image_from_patches_polar(model_location,image_size,window,key):
    #      image_size = (data.shape[2],data.shape[3])
    # window = (64,64)
    stride=window  # YOU NEED TO GIVE THE STRIDE WHICH WAS GIVEN WHILE CREATING PATCHES, can be other than window
    seq_of_full_images=[]
    patch_size=(64,64)
    image_size=(128,110)
    black_box_side_len=4
    overlap=18
    for output_seq_num in range(1,6):  # GIVE THE NUMBER OF SEQUENCE IN OUTPUT
        if key == 'train_input':
            patch1=np.load(model_location + '1/' + 'gt0' + str(output_seq_num) + '.npy')
            patch2=np.load(model_location + '2/' + 'gt0' + str(output_seq_num) + '.npy')
            patch3=np.load(model_location + '3/' + 'gt0' + str(output_seq_num) + '.npy')
            patch4=np.load(model_location + '4/' + 'gt0' + str(output_seq_num) + '.npy')
        elif key == 'target_output':
            patch1=np.load(model_location + '1/' + 'gt0' + str(output_seq_num + 5) + '.npy')
            patch2=np.load(model_location + '2/' + 'gt0' + str(output_seq_num + 5) + '.npy')
            patch3=np.load(model_location + '3/' + 'gt0' + str(output_seq_num + 5) + '.npy')
            patch4=np.load(model_location + '4/' + 'gt0' + str(output_seq_num + 5) + '.npy')
        elif key == 'predicted_output':
            patch1=np.load(model_location + '1/' + 'pd0' + str(output_seq_num + 5) + '.npy')
            patch2=np.load(model_location + '2/' + 'pd0' + str(output_seq_num + 5) + '.npy')
            patch3=np.load(model_location + '3/' + 'pd0' + str(output_seq_num + 5) + '.npy')
            patch4=np.load(model_location + '4/' + 'pd0' + str(output_seq_num + 5) + '.npy')

        patch1_new=copy.copy(patch1)
        patch1_new[:,:black_box_side_len]=patch3[:,:black_box_side_len]
        patch1_new[:,patch_size[1] - black_box_side_len:]=patch2[:,overlap - black_box_side_len:overlap]
        patch1_new[:black_box_side_len,:]=patch3[patch_size[0] - black_box_side_len:,:]

        patch2_new=copy.copy(patch2)
        patch2_new[:,:black_box_side_len]=patch1[:,patch_size[0] - overlap:patch_size[0] - overlap + black_box_side_len]
        patch2_new[:,patch_size[1] - black_box_side_len:]=patch4[:,patch_size[1] - black_box_side_len:]
        patch2_new[:black_box_side_len,:]=patch4[patch_size[0] - black_box_side_len:,:]

        patch3_new=copy.copy(patch3)
        patch3_new[:,:black_box_side_len]=patch1[:,:black_box_side_len]
        patch3_new[patch_size[0] - black_box_side_len:,:]=patch1[:black_box_side_len,:]
        patch3_new[:,patch_size[1] - black_box_side_len:]=patch4[:,overlap - black_box_side_len:overlap]

        patch4_new=copy.copy(patch4)
        patch4_new[:,:black_box_side_len]=patch3[:,patch_size[0] - overlap:patch_size[0] - overlap + black_box_side_len]
        patch4_new[patch_size[0] - black_box_side_len:,:]=patch2[:black_box_side_len,:]
        patch4_new[:,patch_size[0] - black_box_side_len:]=patch2[:,patch_size[1] - black_box_side_len:]

        full_image=np.full(image_size,-1).astype('float64')

        full_image[:patch_size[0],:patch_size[1]]=patch1_new
        full_image[:patch_size[0],image_size[1] - patch_size[1]:]=patch2_new
        full_image[patch_size[0]:,:patch_size[1]]=patch3_new
        full_image[patch_size[0]:,image_size[1] - patch_size[1]:]=patch4_new

        seq_of_full_images.append(full_image)
    return seq_of_full_images


def image_from_patches_maxpooling(model_location,image_size,window,key):
    #      image_size = (data.shape[2],data.shape[3])
    # window = (64,64)
    stride=window  # YOU NEED TO GIVE THE STRIDE WHICH WAS GIVEN WHILE CREATING PATCHES, can be other than window
    seq_of_full_images=[]
    patch_size=(64,64)
    image_size=(128,110)
    black_box_side_len=4
    overlap=18
    for output_seq_num in range(1,6):  # GIVE THE NUMBER OF SEQUENCE IN OUTPUT
        if key == 'train_input':
            patch1=np.load(model_location + '1/' + 'gt0' + str(output_seq_num) + '.npy')
            patch2=np.load(model_location + '2/' + 'gt0' + str(output_seq_num) + '.npy')
            patch3=np.load(model_location + '3/' + 'gt0' + str(output_seq_num) + '.npy')
            patch4=np.load(model_location + '4/' + 'gt0' + str(output_seq_num) + '.npy')
        elif key == 'target_output':
            patch1=np.load(model_location + '1/' + 'gt0' + str(output_seq_num + 5) + '.npy')
            patch2=np.load(model_location + '2/' + 'gt0' + str(output_seq_num + 5) + '.npy')
            patch3=np.load(model_location + '3/' + 'gt0' + str(output_seq_num + 5) + '.npy')
            patch4=np.load(model_location + '4/' + 'gt0' + str(output_seq_num + 5) + '.npy')
        elif key == 'predicted_output':
            patch1=np.load(model_location + '1/' + 'pd0' + str(output_seq_num + 5) + '.npy')
            patch2=np.load(model_location + '2/' + 'pd0' + str(output_seq_num + 5) + '.npy')
            patch3=np.load(model_location + '3/' + 'pd0' + str(output_seq_num + 5) + '.npy')
            patch4=np.load(model_location + '4/' + 'pd0' + str(output_seq_num + 5) + '.npy')

        patch1[:,patch_size[0] - overlap:]=np.maximum(patch1[:,patch_size[0] - overlap:],patch2[:,:overlap])
        patch2[:,:overlap]=np.maximum(patch1[:,patch_size[0] - overlap:],patch2[:,:overlap])
        patch3[:,patch_size[0] - overlap:]=np.maximum(patch3[:,patch_size[0] - overlap:],patch4[:,:overlap])
        patch4[:,:overlap]=np.maximum(patch3[:,patch_size[0] - overlap:],patch4[:,:overlap])

        full_image=np.full(image_size,-1).astype('float64')

        full_image[:patch_size[0],:patch_size[1]]=patch1
        full_image[:patch_size[0],image_size[1] - patch_size[1]:]=patch2
        full_image[patch_size[0]:,:patch_size[1]]=patch3
        full_image[patch_size[0]:,image_size[1] - patch_size[1]:]=patch4

        seq_of_full_images.append(full_image)
    return seq_of_full_images

def image_from_patches(model_location,image_size, window,key,Recombine):
    if Recombine == "Advanced Polar technique":
        return image_from_patches_polar(model_location,image_size, window,key)
    if Recombine == "Max Pooling":
        return image_from_patches_maxpooling(model_location,image_size,window,key)
    else:
        #     image_size = (data.shape[2],data.shape[3])
        # window = (64,64)
        stride = window # YOU NEED TO GIVE THE STRIDE WHICH WAS GIVEN WHILE CREATING PATCHES, can be other than window
        full_image = np.full(image_size, -1).astype('float64')
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
                    if key=='train_input':
                        path=model_location+ str(folder)+'/'+'gt0'+str(output_seq_num)+'.npy'
                    elif key=='target_output':
                        path=model_location + str(folder) + '/' + 'gt0' + str(output_seq_num+5) + '.npy'
                    elif key=='predicted_output':
                        path=model_location + str(folder) + '/' + 'pd0' + str(output_seq_num+5) + '.npy'
                    print(path)
                    #patch = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
                    patch=np.load(path)
                    output_patch = np.add(patch, full_img_patch, where=full_img_patch>-1)
                    output_patch[full_img_patch==-1] = patch[full_img_patch==-1]
                    output_patch[full_img_patch==-1] = output_patch[full_img_patch==-1] * 2
                    output_patch = output_patch / 2
                    full_image[h:h + window[0],v:v + window[1]] = output_patch
            seq_of_full_images.append(full_image)
            folder=0
        return seq_of_full_images

def savesample(model_no,model_location,size,advancedRecombine):
    path='First_and_last/final_outcome'+'/'
    if tf.gfile.Exists(path):
        tf.gfile.DeleteRecursively(path)
    tf.gfile.MakeDirs(path)

    seq_of_full_images=image_from_patches(model_location,(128,110),size,'train_input',advancedRecombine)

    for i in range(len(seq_of_full_images)):
        file_name=path+'train_sample-'+str(i)+'.png'
        plt.imsave(file_name,seq_of_full_images[i])

    seq_of_full_images=image_from_patches(model_location,(128,110),size,'target_output',advancedRecombine)
    for i in range(len(seq_of_full_images)):
        file_name=path+'target_sample-'+str(i)+'.png'
        plt.imsave(file_name,seq_of_full_images[i])

    seq_of_full_images=image_from_patches(model_location,(128,110),size,'predicted_output',advancedRecombine)
    for i in range(len(seq_of_full_images)):
        file_name=path+'predicted_sample-'+str(i)
        plt.imsave(file_name +'.png',seq_of_full_images[i])
        np.save(file_name,seq_of_full_images[i])
    print(" &******** END &********")
#savesample((64,64))