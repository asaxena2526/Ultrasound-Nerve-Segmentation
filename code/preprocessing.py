import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os

IMAGE_SIZE = np.array(Image.open("../data-samples/1_1.tif")).shape
HEIGHT = IMAGE_SIZE[0]
WIDTH = IMAGE_SIZE[1]

def rle_to_mask(rle):
	mask = np.zeros([HEIGHT*WIDTH,])
	if(rle==0):
		return np.zeros([HEIGHT,WIDTH])

	rle = rle.split(' ')

	for i in range(0,len(rle),2):
		mask[int(rle[i])-1:int(rle[i])+int(rle[i+1])-1]=255

	mask = (mask.reshape(WIDTH,HEIGHT)).T

	return mask


# Training Samples

mask_df = pd.read_csv('../data-samples/train_masks.csv')

for i in range(len(mask_df)):
	image_path = str(mask_df['subject'][i])+'_'+str(mask_df['img'][i])
	mask_path = image_path+'_mask'
	np.save('../data_train/'+image_path,np.array(Image.open('../train/'+image_path+'.tif')))
	np.save('../data_train/'+mask_path,rle_to_image(mask_df['pixels'][i]))

# Submission samples

df = pd.read_csv('../data-samples/sample_submission.csv')

for i in range(len(df)):
	image_path = str(df['img'][i])
	mask_path = image_path+'_mask'
	np.save('../data_test/'+image_path,np.array(Image.open('../test/'+image_path+'.tif')))
	np.save('../data_test/'+mask_path,rle_to_image(0))
