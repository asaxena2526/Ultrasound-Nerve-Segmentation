from models import Unet
from dataset import Ultrasound_Dataset
import torch
from torch import nn , optim
import torchvision
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import pandas as pd
from itertools import chain


def img_to_rle(mask_preds):
	mask_preds = (mask_preds>0.5).float()
	code = []
	for i in range(mask_preds.shape[0]):
		x = mask_preds[i].cpu().numpy().transpose().flatten()
		y = np.where(x)[0]
		if len(y) < 10:
			ans = ''
		else:
			z = np.where(np.diff(y) > 1)[0]
			start = np.insert(y[z + 1], 0, y[0])
			end = np.append(y[z], y[-1])
			length = end - start
			res = [[s + 1, l + 1] for s, l in zip(list(start), list(length))]
			res = list(chain.from_iterable(res))
			ans = ' '.join([str(r) for r in res])

	code.append(ans)

	return code

def predict(Model,loader):
	Model.eval()

	tk0 = tqdm(loader, desc="submission")
	start = 0
	for step ,batch in enumerate(tk0):

		images = batch[0]
		if use_cuda and torch.cuda.is_available():
			images=images.cuda()

		with torch.no_grad():
			outputs = Model(images)
			mask_preds = outputs[0]
			preds = outputs[1]


			preds = torch.sigmoid(preds)


			for i in range(mask_preds.shape[0]):
				mask_preds[i] = torch.sigmoid(mask_preds[i])*preds[i].max(0)[1]

			rles = img_to_rle(mask_preds)
			file_name = '../submission.csv'
			with open(file_name, 'a+') as f:
			
				for i in range(mask_preds.shape[0]):
					s = str(i+1+start) + ',' + rles[i]
					f.write(s + '\n')

		start+=mask_preds.shape[0]


file_name = '../submission.csv'
with open(file_name, 'a+') as f:
	f.write('img,pixels\n')

# Load saved model
model = Unet(1,add_residual=True)
model.load_state_dict(torch.load('./saved_model')) # Load trained model


if use_cuda and torch.cuda.is_available():
  model.cuda()

transforms_valid = A.Compose([
    A.Resize(height=512, width=512, p=1.0),
    
    # A.Normalize(mean=(0),std=(255),p=1.0),
    ToTensorV2(p=1.0),
])

sub = pd.read_csv('../data-samples/sample_submission.csv')
sub_data = Ultrasound_Dataset(sub,transform=transforms_valid) # Same Transform as in validation
sub_loader = DataLoader(sub_data , batch_size = 4,shuffle=False) # Same batch_size as in validation

predict(model,sub_loader)

# All prediction are saved in submission.csv

    

