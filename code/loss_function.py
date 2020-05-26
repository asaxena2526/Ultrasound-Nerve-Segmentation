import torch
from torch import nn , optim
import torchvision
import torch.nn.functional as F

class CustomLoss(nn.Module):

	def __init__(self,factor1,factor2):
		super(CustomLoss, self).__init__()
		self.factor1 = factor1
		self.factor2 = factor2

	def dice_coef(self,mask,mask_pred,smooth=1):
		mask_flat = mask.reshape(mask.shape[0],-1)
		mask_pred_flat = mask_pred.reshape(mask.shape[0],-1)
		intersection = torch.sum(mask_flat*mask_pred_flat,dim=1)

		return (2. * intersection + smooth) / (torch.sum(mask_flat,dim=1) + torch.sum(mask_pred_flat,dim=1) + smooth)

	def dice_coef_loss(self,mask,mask_pred):
		return -1*torch.log(self.dice_coef(mask,mask_pred))

	def forward(self,mask,mask_pred,labels,preds):
		mask_pred = torch.sigmoid(mask_pred)
		preds = preds.float()
		labels = labels.float()

		logprobs = F.log_softmax(preds, dim=-1)

		loss1 = -labels * logprobs
		loss1 = loss1.sum(-1)

		loss2 = self.dice_coef_loss(mask,mask_pred)
		loss2 = loss2.unsqueeze(-1)
		
		loss = self.factor1*loss1 + self.factor2*loss2*(labels.max(1)[1])

		return loss.mean()