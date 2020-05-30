from torchvision import transforms
from torch.utils.data import  Dataset
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2


class Ultrasound_Dataset(Dataset):
	def __init__(self,df,train=True,transform=None):
	    self.df = df
	    self.transform = transform
	    self.train = train
  
	def __len__(self):
		return (self.df).shape[0]
    
	def __getitem__(self, index):
		name = str(self.df['subject'][index])+'_'+str(self.df['img'][index])
		if self.train:
			image = np.load('../data_train/'+name+'.npy')
			mask = np.load('../data_train/'+name+'_mask.npy')
		else:
			image = np.load('../data_test/'+name+'.npy')
			mask = np.load('../data_test/'+name+'_mask.npy')
		label = [0,1] if np.sum(mask) else [1,0]
		image.shape = (image.shape[0],image.shape[1],1)
		mask.shape = (mask.shape[0],mask.shape[1],1)

		if self.transform:
		    transformed_img = self.transform(image=image,mask=mask)
		    image = transformed_img['image']
		    mask = transformed_img['mask']

		return image/255.0,mask//255,torch.Tensor(label) 
