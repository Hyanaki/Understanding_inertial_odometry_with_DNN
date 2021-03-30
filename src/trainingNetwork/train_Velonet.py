import os
import time

import matplotlib.pyplot as plt
import math

import torch

from utils.utils import prepare_data
from torch.autograd import Variable
from utils.pytorchtools import EarlyStopping


import tensorboardX as tbx
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from models.VeloNet import *
from trainingNetwork.Generate_Training_dataset_Velocity import create_dataset_Relative_Kinematic

#from Generate_Training_dataset_Velocity import *

model_path_VeloNet = "./saved_NN/neural_network.p"
torch.set_printoptions(precision=10)
windows_size = 11
lr = 1e-03

load_model = False
batch_size = 256
epoch_len = 400
custom_layer2D = CustomMultiLossLayer2D()


def get_Relative_Kinematic(training=1):
	return Relative_Kinematic(batch_size,training)

#apply quick standardization to the tensor
def quick_std(tensor):

	dictionary_std = torch.load('../temp/standardization_factor.p')

	input_loc = dictionary_std['input_loc']
	input_std = dictionary_std['input_std']
	
	tensor[:,:,0:6] = (tensor[:,:,0:6]-input_loc[0:6])/input_std[0:6]
	#tensor[:,:,6:9] = (tensor[:,:,6:9]-input_loc[6:9])/input_std[6:9]
	tensor[:,:,9:13] = (tensor[:,:,9:13]-input_loc[9:13])/input_std[9:13]
	#tensor[:,:,13] = (tensor[:,:,13]-input_loc[13])/input_std[13]

	return tensor

#apply quick normalization to the tensor
def quick_norm(tensor):
	dictionary_norm = torch.load('../temp/normalization_factor.p')

	max_quat = torch.FloatTensor(dictionary_norm['max_quat'])
	min_quat = torch.FloatTensor(dictionary_norm['min_quat'])

	max_u = torch.FloatTensor(dictionary_norm['max_u'])
	min_u = torch.FloatTensor(dictionary_norm['min_u'])


	if tensor.shape[2] <=7:
		tensor[:,:,0:3] = ((tensor[:,:,0:3]- min_vel)/(max_vel-min_vel))
		tensor[:,:,3:7] = ((tensor[:,:,3:7]- min_quat)/(max_quat-min_quat))
		return tensor
	
	tensor[:,:,0:6] = ((tensor[:,:,0:6]- min_u)/(max_u-min_u))
	tensor[:,:,6:9] = ((tensor[:,:,6:9]- min_vel)/(max_vel-min_vel))
	tensor[:,:,9:13] = ((tensor[:,:,9:13]- min_quat)/(max_quat-min_quat))
	return tensor
	
def train_velonet(args, dataset):

	training_data, training_label, valid_data, valid_label = create_dataset_Relative_Kinematic(args,dataset,windows_size)

	valid_data.requires_grad = False
	valid_label.requires_grad = False
	
	training_data = quick_std(training_data)
	valid_data = quick_std(valid_data)
	#training_label = quick_norm(training_label)
	#valid_label = quick_norm(valid_label)

	early_stopping = EarlyStopping(patience=35, verbose=True)

	device = torch.device('cuda:0' if torch.cuda.is_available() and not args.cpu else 'cpu') #device = torch.device('cpu')
	
	network = get_Relative_Kinematic().to(device)


	print('Number of train samples: {}'.format(training_data.shape[0]))
	print('Number of val samples: {}'.format(valid_data.shape[0]))

	total_params = network.get_num_params()
	print('Total number of parameters: ', total_params)

	optimizer = torch.optim.Adam(network.parameters(), lr)
	
	#If after 25 epochs the validation loss did not improve we reduce the learning rate to converge towards optimal solution
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=25, verbose=True, eps=1e-12)

	if load_model:
		dictionary = torch.load(model_path_VeloNet)
		network.load_state_dict(dictionary.get('model_state_dict'))
		optimizer.load_state_dict(dictionary.get('optimizer_state_dict'))

	start_time = time.time()

	avg_train_losses = []
	avg_valid_losses = []
	writer = SummaryWriter()

	for epoch in range(1, epoch_len+1):

		train_loss,valid_loss = train_loop_Relative_Kinematic(args,dataset,network, device ,optimizer, scheduler, training_data, valid_data, training_label, valid_label,batch_size, writer)

		avg_train_losses.append(train_loss)
		avg_valid_losses.append(valid_loss)
		
		writer.add_scalars(f'Train_loss/Validation_loss', {
						    'Train_loss': train_loss,
						    'Valid_loss': valid_loss,
						}, epoch)

		save = early_stopping(avg_valid_losses[-1], network)
		
		if save: #Save the model if the validation loss improved
		    torch.save({'model_state_dict': network.state_dict(),
			    'optimizer_state_dict': optimizer.state_dict(),
			    'epoch': epoch_len}, model_path_VeloNet)
		    print('SAVE')
			
		if early_stopping.early_stop: #Otherwise if after patience = 35 epoch, the validation loss did not improve we stop the training
		    print('Early stopping')
		    break
		
		print("Epoch number : " + str(epoch) + "/" + str(epoch_len))
		print('\tTrain_Loss: {:.9f}'.format(avg_train_losses[-1]))
		print('\tValid_Loss: {:.9f}'.format(avg_valid_losses[-1]))
		print("Amount of time spent for 1 epoch: {}s\n".format(int(time.time() - start_time)))
		start_time = time.time()
	writer.close()


 #Train loop for the training process and the validation process
def train_loop_Relative_Kinematic(args,dataset,model, device,optimizer, scheduler, training_data, valid_data, training_label, valid_label , batch_size,writer=None):
	
	start_time = time.time()
	tr_loss = 0

	model.train()
	training_iteration = 0
	for i in range(0,training_data.shape[0],batch_size):

		feat_gyr = Variable(training_data[i:i+batch_size][:,:,0:3]).to(device)
		feat_acc = Variable(training_data[i:i+batch_size][:,:,3:6]).to(device)
		feat_dt = Variable(training_data[i:i+batch_size][:,:,13].unsqueeze(2)).to(device)
		feat_prev_quat = Variable(training_data[i:i+batch_size][:,:,9:13]).to(device)

		targ = Variable(training_label[i:i+batch_size][:,:,0:3]).to(device),Variable(training_label[i:i+batch_size][:,:,3:7]).to(device)
		
		model.hidden_initialize()
		optimizer.zero_grad()
		output = model(feat_gyr,feat_acc,feat_dt,feat_prev_quat)
	
		loss_train = custom_layer2D(output,targ)

		tr_loss += loss_train.item()
		loss_train.backward()
		optimizer.step()
		training_iteration+=1
	   
	print("Amount of time spent to Train dataset: {}s\n".format(int(time.time() - start_time)))
	print('-----------------------------------------------------------------------------------------------------------------------')
	start_time = time.time()
	
	model.eval()
	validation_iteration = 0
	val_loss = 0
	with torch.no_grad():
		for i in range(0,valid_data.shape[0],batch_size):

			feat_gyr = (valid_data[i:i+batch_size][:,:,0:3]).to(device)
			feat_acc = (valid_data[i:i+batch_size][:,:,3:6]).to(device)
			feat_dt = (valid_data[i:i+batch_size][:,:,13].unsqueeze(2)).to(device)
			feat_prev_quat = (valid_data[i:i+batch_size][:,:,9:13]).to(device)

			targ = valid_label[i:i+batch_size][:,:,0:3].to(device),valid_label[i:i+batch_size][:,:,3:7].to(device)
				
			output = model(feat_gyr,feat_acc,feat_dt,feat_prev_quat)
			loss_validation = custom_layer2D(output,targ)

			val_loss += loss_validation.item()
			validation_iteration+=1
	
	if (scheduler is not None):
		scheduler.step(val_loss/validation_iteration)
	print("Amount of time spent to validate dataset: {}s\n".format(int(time.time() - start_time)))
	print('-----------------------------------------------------------------------------------------------------------------------')
	
	return tr_loss/training_iteration , val_loss/validation_iteration

