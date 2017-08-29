import os, sys
import numpy as np 
import matplotlib.pyplot as plt 
import cv2 

# Load Dataset
def load_subset(subsets):
    data = []
    label = []
    base_path = '/Users/thomasan/Documents/UCSD/CSE152/HW5/data/yaleBfaces/'
    for subset in subsets:
        directory = os.path.join(base_path, "subset" + str(subset))
        files = os.listdir(directory)
        for img in files:
            face = cv2.imread(os.path.join(directory,img), cv2.IMREAD_GRAYSCALE)
            data.append(face)
            label.append(int(img.split('person')[1].split('_')[0]))
    return data, label

# Draw faces
def draw_faces(img_list, col):
    fig = plt.figure(figsize = (30,30))
    if len(img_list) < col:
        col = len(img_list)
        row = 1
    else:
        row = int(len(img_list)/col)
    for sub_img in range(1,row*col+1):
        ax = fig.add_subplot(row, col, sub_img)
        ax.imshow(img_list[sub_img-1], cmap='gray')
        ax.axis('off')
    plt.show()



########### EXAMPLE #################

# TODO: Update base_path to point yaleBfaces directory 
# data,label = load_subset([0],'/Users/thomasan/Documents/UCSD/CSE152/HW5/data/yaleBfaces/') 
# draw_faces(data, 10)