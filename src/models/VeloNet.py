import torch
import torch.nn as nn
from torch import autograd
from torch.autograd import Variable
from utils.quaternions import hamilton_product
from utils.utils_quaternion import conjugate

#Compute the relative quaternion loss
#Function used : hamilton_product from quaternions library
#Quaternion format accepted : 2D tensor
def quaternion_loss(output,target,device):

	#Reshape 3D tensor to 2D tensor
	output = output.view(-1,output.shape[2])
	target = target.view(-1,target.shape[2])

	#Normalization of the quaternion
	q_output = torch.div(output,torch.norm(output,dim=1).unsqueeze(1).repeat(1,4))

	#Add a logarithm term otherwise the NN return a quaternion null
	log_q_out_norm = torch.log((torch.norm(q_output,dim=1).unsqueeze(1)))

	conj_target = conjugate(target,device)
	qprod = hamilton_product(q_output,conj_target)

	w,x,y,z = torch.chunk(qprod, 4, dim=-1)

	return 2*torch.sum(torch.abs(torch.cat((x,y,z),axis=-1)-log_q_out_norm.repeat(1,3)),dim=1)
		


#Custom Loss layer to balance the relative velocity loss and the relative quaternion loss automatically
class CustomMultiLossLayer2D(nn.Module):
	def __init__(self, nb_outputs = 2, **kwargs):
		super(CustomMultiLossLayer2D,self).__init__()
		self.nb_outputs = nb_outputs
		self.log_vars = []
		self.lossL2 = torch.nn.MSELoss(reduction='none')
		
		self.device = torch.device('cuda:0')
		
		#Weight definition for each loss
		for i in range(self.nb_outputs):
			self.log_vars += [torch.nn.Parameter(torch.zeros(1),requires_grad=True).to(self.device)]

	def multi_loss(self, ys_pred, ys_true):

		#Relative velocity loss
		precision = torch.exp(-self.log_vars[0][0]).to(self.device)
		loss = precision * torch.mean(torch.sum(self.lossL2(ys_pred[0],ys_true[0]),dim=2),dim=1) + self.log_vars[0][0]

		#Relative quaternion loss
		precision = torch.exp(-self.log_vars[1][0]).to(self.device)
		loss += precision * torch.sum(quaternion_loss(ys_pred[1],ys_true[1],self.device).view(ys_pred[1].shape[0],-1),dim=1) + self.log_vars[1][0]
		return torch.mean(loss,dim=0)
	

	def forward(self,output,target):
		return self.multi_loss(output,target)


#Fully connected layers to output relative quaternion and relative velocity
class FCOutputModuleMultiple(nn.Module):

	def __init__(self, input_size, n_output,nb_units, **kwargs):
		
		super(FCOutputModuleMultiple,self).__init__()
		
		self.fc = nn.Sequential(
			nn.Linear(input_size, nb_units),
			torch.nn.ReLU(inplace = True),
			nn.Linear(nb_units, n_output)
			)
		
	def forward(self,x):
		output = self.fc(x)
		return output

#Bidirectional LSTM
class LSTM_Block(nn.Module):
	def __init__(self, input_dim, hidden_dim, batch_size, training, n_layers=1):
		super(LSTM_Block,self).__init__()

		self.hidden_dim = hidden_dim
		self.input_dim = input_dim
		self.batch_size = batch_size
		self.device = torch.device('cuda:0' if training else 'cpu')
		self.hidden = None

		#Bidirectional LSTM --> hidden_dim//2
		self.lstm = nn.LSTM(input_dim,hidden_dim//2, batch_first=True, bidirectional = True)
		

	def forward(self, x):
		self.batch_size = x.shape[0]
		self.hidden = self.init_hidden()
		lstm_out,self.hidden = self.lstm(x,self.hidden)
		return lstm_out,self.hidden
		
	def init_hidden(self):
		hidden_a = torch.randn(2, self.batch_size, self.hidden_dim//2).to(self.device)
		hidden_b = torch.randn(2, self.batch_size, self.hidden_dim//2).to(self.device)

		hidden_a = Variable(hidden_a, requires_grad=True)
		hidden_b = Variable(hidden_b, requires_grad=True)

		return (hidden_a, hidden_b)

#Main Architecture
class Relative_Kinematic(nn.Module):

	def __init__(self, batch_size, training=True):
		super(Relative_Kinematic,self).__init__()

		lstm_output_1 = 100
		lstm_output_2 = 250
		lstm_output_3_ori = 100
		lstm_output_3_vel = 100
		

		channel_input_vel = 3
		channel_input_ori = 2

		units_ori = 20
		units_vel = 20

		self.batch_size = batch_size

		self.lstm_acc = LSTM_Block(3,lstm_output_1,batch_size,training)
		self.dropout_acc = nn.Dropout(p=0.15)

		self.lstm_prev_gyr = LSTM_Block(3,lstm_output_1,batch_size,training)
		self.dropout_gyr = nn.Dropout(p=0.15)
				
		
		self.lstm_prev_quat = LSTM_Block(4,lstm_output_1,batch_size,training)
		self.dropout_quat = nn.Dropout(p=0.15)

		self.lstm_dt = LSTM_Block(1,lstm_output_1,batch_size,training)
		self.dropout_dt = nn.Dropout(p=0.15)

		self.lstm_vel_1 = LSTM_Block(channel_input_vel*lstm_output_1,lstm_output_2,batch_size,training)
		self.lstm_vel_2 = LSTM_Block(lstm_output_2,lstm_output_3_vel,batch_size,training)
		
		self.lstm_orientation_1 = LSTM_Block(channel_input_ori*lstm_output_1,lstm_output_2,batch_size,training)
		self.lstm_orientation_2 = LSTM_Block(lstm_output_2,lstm_output_3_ori,batch_size,training)

		self.output_block_ori= FCOutputModuleMultiple(lstm_output_3_ori,4,units_ori)
		self.output_block_vel = FCOutputModuleMultiple(lstm_output_3_vel,3,units_vel)
		
		self.dropout_ori = nn.Dropout(p=0.15)
		self.dropout_vel = nn.Dropout(p=0.15)

	def forward(self,prev_gyr,acc,dt,prev_quat):

		prev_gyr,_ = self.lstm_prev_gyr(prev_gyr)
		prev_gyr = self.dropout_gyr(prev_gyr)

		acc,_ = self.lstm_acc(acc)
		acc = self.dropout_acc(acc)

		dt,_ = self.lstm_dt(dt)
		dt = self.dropout_dt(dt)

		prev_quat,_ = self.lstm_prev_quat(prev_quat)
		prev_quat = self.dropout_quat(prev_quat)

		feature_vec = torch.cat((acc,dt,prev_quat),axis=2)
		feature_vec_orientation = torch.cat((prev_gyr,dt),axis=2)
		
		lstm_vel_1,_ = self.lstm_vel_1(feature_vec)
		lstm_vel_1 = self.dropout_vel(lstm_vel_1)
		lstm_vel_2,_ = self.lstm_vel_2(lstm_vel_1)
		
		lstm_ori_1,_ = self.lstm_orientation_1(feature_vec_orientation)
		lstm_ori_1 = self.dropout_ori(lstm_ori_1)
		lstm_ori_2,_ = self.lstm_orientation_2(lstm_ori_1)


		output_vel  = self.output_block_vel(lstm_vel_2[:,:,:])
		output_ori = self.output_block_ori(lstm_ori_2[:,:,:])
		
		return output_vel , output_ori


	def hidden_initialize(self):
		for m in self.modules():
			if isinstance(m,LSTM_Block):
				m.init_hidden()

	def get_num_params(self):
		for p in self.parameters():
			if p.requires_grad:
				print(p.numel())
		return sum(p.numel() for p in self.parameters() if p.requires_grad)