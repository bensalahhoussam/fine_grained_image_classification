import os
import cv2 as cv
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import keras.backend as K
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.optimizers import Adam
from mtcnn.mtcnn import MTCNN


path_dataset="D://Deep_Learning_projects/new_projects/computer_vision/skin_dataset/"
dataset_file = "D://skin_dataset/"

def make_folder(file):
    if os.path.exists(file):
        pass
    else:
        return os.makedirs(file)
def get_index(j):
    if j<10:
        return '00'
    elif j >=10 and j <100:
        return "0"
    else:
        return ""
def data_preparation(file):
    class_name = [name for name in os.listdir(file)]
    for i in range(len(class_name)):
        total_images=[img for img in os.listdir(file+class_name[i])]
        for j in range(len(total_images)):
            if j < 1100:
                image_path=file+class_name[i]+"/"+total_images[j]
                image=cv.imread(image_path)
                file_1=path_dataset+"train"+"/"+class_name[i]+"/"
                make_folder(file_1)
                cv.imwrite(file_1 + "/" + "image_" + get_index(j) + str(j) + ".jpg", image)
            elif j >= 1100 and j<1200:
                image_path = file + class_name[i] + "/" + total_images[j]
                image = cv.imread(image_path)
                file_1 = path_dataset + "valid" + "/" + class_name[i] + "/"
                make_folder(file_1)
                cv.imwrite(file_1 + "/" + "image_" + get_index(j) + str(j) + ".jpg", image)
            else:
                image_path = file + class_name[i] + "/" + total_images[j]
                image = cv.imread(image_path)
                file_1 = path_dataset + "test" + "/" + class_name[i] + "/"
                make_folder(file_1)
                cv.imwrite(file_1 + "/" + "image_" + get_index(j) + str(j) + ".jpg", image)

def extract_face(image):
    detector=MTCNN()
    faces=detector.detect_faces(image)
    x1,y1,width,height=faces[0]["box"]
    x2,y2=x1+width,y1+height
    face=image[y1:y2,x1:x2]
    face=cv.resize(face,(224,224))
    return face

image_file="D://Deep_Learning_projects/new_projects/computer_vision/skin_dataset/train/combination/image_037.jpg"
path_1="D://Deep_Learning_projects/new_projects/computer_vision/skin_dataset/"

def data_preprocessing(path):
    names=[name for name in os.listdir(os.path.join(path_1,path))]
    x_train=[]
    y_train=[]
    for i in range(len(names)):
        total_images=[img for img in os.listdir(os.path.join(path_1,path,names[i]))]
        for img in total_images:
            image_path=path_1+path+"/"+names[i]+"/"+img
            try:
                image=cv.imread(image_path)
                face=extract_face(image)
                x_train.append(face)
                y_train.append(i)

            except:
                pass
    return np.array(x_train),np.array(y_train)
