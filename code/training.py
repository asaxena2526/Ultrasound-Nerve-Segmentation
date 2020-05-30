from models import Unet
from dataset import Ultrasound_Dataset
from loss_function import CustomLoss
import sklearn.metrics as met
from statistics import mean
import copy
import torch
from torch import nn , optim
import torchvision
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms , datasets
from torch.utils.data import  Dataset,DataLoader




def dice_coef(mask,mask_pred,smooth=1):
	mask_flat = mask.reshape(mask.shape[0],-1)
	mask_pred_flat = mask_pred.reshape(mask.shape[0],-1)
	intersection = torch.sum(mask_flat*mask_pred_flat,dim=1)

	return (2. * intersection + smooth) / (torch.sum(mask_flat,dim=1) + torch.sum(mask_pred_flat,dim=1) + smooth)

def train_(model,optimizer,scheduler,criterion,train_loader,valid_loader,epochs=1):

	training_losses=[]
	valid_losses = []
	max_score=0.0

	for epoch in range(epochs):

		training_loss=0
		valid_loss=0
		score = 0.0

		model.train()
		tk0 = tqdm(train_loader, desc="Train")
		for step ,batch in enumerate(tk0):

			images = batch[0]
			masks = batch[1]
			labels = batch[2]

			if use_cuda and torch.cuda.is_available():
				images=images.cuda()
				masks=masks.cuda()
				labels=labels.cuda()

			optimizer.zero_grad()

			outputs = model(images)
			mask_preds = outputs[0]
			preds = outputs[1]

			loss = criterion(masks,mask_preds,labels,preds)
			loss.backward()
			optimizer.step()
			training_loss += loss.item()
			del images
			del masks
			del labels
			del outputs
			del mask_preds
			del preds
			del loss


		model.eval()
		
		


		tk0 = tqdm(valid_loader, desc="Validate")
		for step ,batch in enumerate(tk0):
		  
			images = batch[0]
			masks = batch[1]
			labels = batch[2]

			if use_cuda and torch.cuda.is_available():
				images=images.cuda()
				masks=masks.cuda()
				labels=labels.cuda()

			with torch.no_grad():
				outputs = model(images)
				mask_preds = outputs[0]
				preds = outputs[1]

				loss = criterion(masks,mask_preds,labels,preds)


				preds = torch.sigmoid(preds)

				valid_loss += loss.item()
				for i in range(masks.shape[0]):
					mask_preds[i] = torch.sigmoid(mask_preds[i])*preds[i].max(0)[1]
				score += torch.sum(dice_coef(masks, mask_preds))/masks.shape[0]

			del images
			del masks
			del labels
			del outputs
			del mask_preds
			del preds
			del loss
		    
		    
		    
		model.train()
		  

		training_losses.append([epoch,training_loss/len(train_loader)])
		valid_losses.append([epoch,valid_loss/len(valid_loader)])
		scheduler.step(valid_losses[-1][1])
		score /= len(valid_loader)

		print('Epoch {}/{}\n----------------------------\n\
		Traininig loss : {}\tValidation loss : {}\n\
		Validation Score : {}'.format(epoch+1,epochs,training_loss/len(train_loader),valid_loss/len(valid_loader),score))

	return np.array(training_losses),np.array(valid_losses) ,model

def plot_learning_curve(train_losses,valid_losses):
	plt.plot(train_losses[:,0],train_losses[:,1],color='c',label='Training Loss')
	plt.plot(valid_losses[:,0],valid_losses[:,1],color='r',label='Validation Loss')
	plt.xlabel('Ephocs')
	plt.ylabel('Loss')
	plt.legend()
	plt.show()


if __name__ == '__main__':

	df = pd.read_csv('./train_masks.csv')
	df['sample_type'] = 'train'

	sample_idx = df.sample(frac=0.2, random_state=42).index
	df.loc[sample_idx, 'sample_type'] = 'valid'
	valid_df = df[df['sample_type'] == 'valid']
	valid_df.reset_index(drop=True, inplace=True)

	train_df = df[df['sample_type'] == 'train']
	train_df.reset_index(drop=True, inplace=True)

	# Tranforms and Dataloader

	transforms_train = A.Compose([
    A.Resize(height=512, width=512, p=1.0),
    A.Flip(),
    A.ShiftScaleRotate(rotate_limit=1.0, p=0.8),

    # A.Normalize(mean=(0),std=(255),p=1.0),
    ToTensorV2(p=1.0),
	])
	transforms_valid = A.Compose([
	    A.Resize(height=512, width=512, p=1.0),
	    
	    # A.Normalize(mean=(0),std=(255),p=1.0),
	    ToTensorV2(p=1.0),
	])
	train_data = Ultrasound_Dataset(train_df,transform=transforms_train)
	train_loader = DataLoader(train_data , batch_size = 4,shuffle=True)

	valid_data = Ultrasound_Dataset(valid_df,transform=transforms_valid)
	valid_loader = DataLoader(valid_data , batch_size = 4,shuffle=False)

	# Checking GPU Avalaibility

	use_cuda = True
	if use_cuda and torch.cuda.is_available():
	  print('yes')
	print(torch.cuda.is_available())


	# Model Initialization

	model = Unet(1,net_type='semi_inception',version='b',add_residual=True)
	
	if use_cuda and torch.cuda.is_available():
	  model.cuda()
	
	criterion = CustomLoss(0.5,1)

	optimizer = optim.Adam(model.parameters(),5e-6)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 3)
	training_loss,valid_loss,model,saved_model=train_(model,optimizer,scheduler,criterion,train_loader,valid_loader,epochs=5)
	plot_learning_curve(training_loss,valid_loss)

	# save model for further use

	torch.save(model.state_dict(),'./saved_models/Mymodel')


