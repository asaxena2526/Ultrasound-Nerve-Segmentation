import torch
from torch import nn , optim
import torchvision
import torch.nn.functional as F

class NConv2d(nn.Module):
	def __init__(self,in_c,out_c,kernel_size,stride,padding,actv,dummy):
		super(NConv2d,self).__init__()
		self.actv = actv
		self.conv = nn.Conv2d(in_c,out_c,kernel_size=kernel_size,stride=stride,padding=padding)
		self.norm = nn.BatchNorm2d(out_c)

	def forward(self,x):
		# print(x.shape)
		x = self.conv(x)
		x = self.norm(x)
		if self.actv:
			x = self.actv(x,inplace=True)

		return x

def inception_block(in_c,out_c,dummy1,dummy2,dummy3,actv,version='b'):
	if version=='a':
		return inception_block_v1a(in_c,out_c,actv)
	if version == 'b':
		return inception_block_v1b(in_c,out_c,actv)

class inception_block_v1a(nn.Module):
	def __init__(self,in_c,out_c,actv):
		super(inception_block_v1a,self).__init__()
		self.actv=actv

		self.conv1 = NConv2d(in_c,out_c//4,kernel_size=1,stride=1,padding=0,actv=self.actv,dummy='')
		self.conv2 = NConv2d(in_c,out_c//4,kernel_size=3,stride=1,padding=1,actv=self.actv,dummy='')
		self.conv3 = NConv2d(in_c,out_c//4,kernel_size=5,stride=1,padding=2,actv=self.actv,dummy='')

		self.pool = nn.MaxPool2d(3,stride=1,padding=1)

	def _forward(self,x):
		conv1x1 = self.conv1(x)
		conv3x3 = self.conv2(x)
		conv5x5 = self.conv3(x)

		pool = self.conv1(self.pool(x))

		outputs = [conv1x1,conv3x3,conv5x5,pool]
		return outputs

	def forward(self,x):
		outputs = self._forward(x)
		return torch.cat(outputs,1)

class inception_block_v1b(nn.Module):
	def __init__(self,in_c,out_c,actv):
		super(inception_block_v1b,self).__init__()

		self.actv = actv

		self.conv1 = NConv2d(in_c,out_c//4,kernel_size=1,stride=1,padding=0,actv=self.actv,dummy='')

		self.conv2_1 = NConv2d(in_c,out_c//8,kernel_size=1,stride=1,padding=0,actv=self.actv,dummy='')
		self.conv2_2 = NConv2d(out_c//8,out_c//4,kernel_size=3,stride=1,padding=1,actv=self.actv,dummy='')

		self.conv3_1 = NConv2d(in_c,out_c//8,kernel_size=1,stride=1,padding=0,actv=self.actv,dummy='')
		self.conv3_2 = NConv2d(out_c//8,out_c//4,kernel_size=5,stride=1,padding=2,actv=self.actv,dummy='')

		self.pool = nn.MaxPool2d(3,stride=1,padding=1)

	def _forward(self,x):
		conv1x1 = self.conv1(x)

		conv3x3 = self.conv2_1(x)
		conv3x3 = self.conv2_2(conv3x3)

		conv5x5 = self.conv3_1(x)
		conv5x5 = self.conv3_2(conv5x5)

		pool = self.conv1(self.pool(x))

		outputs = [conv1x1,conv3x3,conv5x5,pool]
		return outputs

	def forward(self,x):
		outputs = self._forward(x)
		return torch.cat(outputs,1)

class residual(nn.Module):
	def __init__(self,in_c,out_c,kernel_size,actv,scale=0.1,req=True):
		super(residual,self).__init__()
		self.conv = NConv2d(in_c,out_c,kernel_size,1,1,None,None)
		self.actv = actv
		self.scale = scale
		self.req = req

	def forward(self,x):
		if not self.req:
			return x

		res = self.conv(x)*self.scale
		x = res+x
		if self.actv:
			x = self.actv(x,inplace=True)

		return x

class Unet(nn.Module):
	def __init__(self,in_c,activation='elu',net_type='normal',version='b',add_residual=False):
		super(Unet,self).__init__()
		assert activation in ['relu','elu']
		assert net_type in ['normal','semi_inception','inception'] 
		self.actv = None
		if activation == 'relu':
			self.actv = F.leaky_relu
		else:
			self.actv = F.elu


		Conv1 = None
		Conv2 = None
		if net_type == 'normal':
			Conv1 = NConv2d
			Conv2 = NConv2d
		elif net_type == 'semi_inception':
			Conv1 = inception_block
			Conv2 = NConv2d
		elif net_type == 'inception':
			Conv1 = inception_block
			Conv2 = inception_block

		self.conv1_1 = Conv1(in_c,64,3,1,1,self.actv,version)
		self.conv1_2 = Conv2(64,64,3,1,1,self.actv,version)
		self.down1 = nn.MaxPool2d(2,stride=2)
		self.res1 = residual(64,64,3,self.actv,0.2,add_residual)


		self.conv2_1 = Conv1(64,128,3,1,1,self.actv,version)
		self.conv2_2 = Conv2(128,128,3,1,1,self.actv,version)
		self.down2 = nn.MaxPool2d(2,stride=2)
		self.res2 = residual(128,128,3,self.actv,0.2,add_residual)


		self.conv3_1 = Conv1(128,256,3,1,1,self.actv,version)
		self.conv3_2 = Conv2(256,256,3,1,1,self.actv,version)
		self.down3 = nn.MaxPool2d(2,stride=2)
		self.res3 = residual(256,256,3,self.actv,0.2,add_residual)


		self.conv4_1 = Conv1(256,512,3,1,1,self.actv,version)
		self.conv4_2 = Conv2(512,512,3,1,1,self.actv,version)
		self.down4 = nn.MaxPool2d(2,stride=2)
		self.res4 = residual(512,512,3,self.actv,0.2,add_residual)


		self.conv5_1 = NConv2d(512,1024,3,1,1,self.actv,version)
		self.conv5_2 = NConv2d(1024,1024,3,1,1,self.actv,version)

		self.aux = nn.Linear(32*32*1024,2)


		self.up4 = nn.ConvTranspose2d(1024,512,2,2)
		self.u_conv4_1 = Conv1(1024,512,3,1,1,self.actv,version)
		self.u_conv4_2 = Conv2(512,512,3,1,1,self.actv,version)


		self.up3 = nn.ConvTranspose2d(512,256,2,2)
		self.u_conv3_1 = Conv1(512,256,3,1,1,self.actv,version)
		self.u_conv3_2 = Conv2(256,256,3,1,1,self.actv,version)

		self.up2 = nn.ConvTranspose2d(256,128,2,2)
		self.u_conv2_1 = Conv1(256,128,3,1,1,self.actv,version)
		self.u_conv2_2 = Conv2(128,128,3,1,1,self.actv,version)

		self.up1 = nn.ConvTranspose2d(128,64,2,2)
		self.u_conv1_1 = Conv1(128,64,3,1,1,self.actv,version)
		self.u_conv1_2 = Conv2(64,64,3,1,1,self.actv,version)

		self.final = nn.Conv2d(64,1,1)

	def forward(self,x):
		x = self.conv1_1(x)
		x = self.conv1_2(x)
		x1 = self.res1(x)
		x = self.down1(x)

		x = self.conv2_1(x)
		x = self.conv2_2(x)
		x2 = self.res2(x)
		x = self.down2(x)

		x = self.conv3_1(x)
		x = self.conv3_2(x)
		x3 = self.res3(x)
		x = self.down3(x)

		x = self.conv4_1(x)
		x = self.conv4_2(x)
		x4 = self.res4(x)
		x = self.down4(x)

		x = self.conv5_1(x)
		x = self.conv5_2(x)

		aux = self.aux(x.reshape(x.shape[0],-1))

		x = self.up4(x)
		x = torch.cat([x4,x],1)
		x = self.u_conv4_1(x)
		x = self.u_conv4_2(x)

		x = self.up3(x)
		x = torch.cat([x3,x],1)
		x = self.u_conv3_1(x)
		x = self.u_conv3_2(x)

		x = self.up2(x)
		x = torch.cat([x2,x],1)
		x = self.u_conv2_1(x)
		x = self.u_conv2_2(x)

		x = self.up1(x)
		x = torch.cat([x1,x],1)
		x = self.u_conv1_1(x)
		x = self.u_conv1_2(x)


		output = self.final(x)


		return [output,aux]