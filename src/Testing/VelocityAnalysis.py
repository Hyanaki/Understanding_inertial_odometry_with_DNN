import os
import time
import math
import torch
import matplotlib.pyplot as plt
import numpy as np


from utils.utils import prepare_data

import tensorboardX as tbx
import torchvision
import torchvision.transforms as transforms

from utils.quaternions import hamilton_product

from utils.utils_quaternion import *

from models.VeloNet import*

torch.set_printoptions(precision=10)

def get_Relative_Kinematic():
	batch_size =256
	return Relative_Kinematic(batch_size,training=False)

#From training and validation dataset, predict the relative quaternion and relative velocity with a trained NN
def simple_test_Velonet(args,dataset):
	
	#---------------------Load the NN model--------------------------------------------
	model = torch.load('./saved_NN/neural_network.p')
	windows_size = 11
	device = torch.device('cpu')
	network = get_Relative_Kinematic().to(device)
	network.load_state_dict(model.get('model_state_dict'))

	fig_x = 4
	fig_y = 4
	fig, axes = plt.subplots(fig_x, ncols=fig_y)
		
	compute_mean = torch.zeros((windows_size,1))

	network.eval()
	for i in range(0,len(dataset)):
		dataset_name = dataset.dataset_name(i)
		print("Test filter on sequence: " + dataset_name)
		

		#---------------------Create a multi display--------------------------------------------
		if(i%(fig_x*fig_y)==0 and i!=0):
			plt.subplots_adjust(left=0.03, bottom=0.03, right=0.97, top=0.97, wspace = 0.12,hspace = 0.34)
			plt.show()
			fig, axes = plt.subplots(fig_x, ncols=fig_y)
        
		#---------------------Load the data--------------------------------------------
		t, ang_gt, _, v_gt, u = prepare_data(args, dataset, dataset_name)
		
		dt = (t[1:]-t[:-1]).float()
		d_vgt = v_gt[1:]-v_gt[:-1]
		ang_gt = correct_ang_gt(ang_gt)

		#---------------------Smoothing ground truth quaternion--------------------------------------------
		quat_tensor = torch.zeros(u.shape[0],4)

		for j in range(0,quat_tensor.shape[0]):
			quat_tensor[j,:] = euler_to_quaternion(ang_gt[j,0], ang_gt[j,1], ang_gt[j,2])

		quat_tensor[2:-2,:] = (quat_tensor[0:-4,:]+quat_tensor[1:-3,:]+quat_tensor[2:-2,:]+quat_tensor[3:-1,:]+quat_tensor[4:,:])/5
		quat_tensor = torch.div(quat_tensor,torch.norm(quat_tensor,dim=1).unsqueeze(1).repeat(1,4))
		relative_quaternion = relative_rotation(quat_tensor)
		relative_quaternion = torch.div(relative_quaternion,torch.norm(relative_quaternion,dim=1).unsqueeze(1).repeat(1,4))

		
		test_dataset = torch.zeros(u.shape[0],windows_size,14)
		label_dataset = torch.zeros(u.shape[0],windows_size,7)
		data_index = 0
		#---------------------Prepare the testing data--------------------------------------------
		for j in range(windows_size,u.shape[0]):
			index = j-windows_size 
					
			test_dataset[data_index,:,0:3] = u[index:j,0:3]
			test_dataset[data_index,:,3:6] = u[index:j,3:6]
			test_dataset[data_index,:,6:9] = v_gt[index:j,0:3]
			test_dataset[data_index,:,9:13] = quat_tensor[index:j,0:4]
			test_dataset[data_index,:,13] = dt[index:j]

			label_dataset[data_index,:,0:3]= d_vgt[index:j]
			label_dataset[data_index,:,3:7]= relative_quaternion[index:j,0:4]
			data_index +=1

		test_dataset = test_dataset[:data_index]
		label_dataset = label_dataset[:data_index]
		
		#test_dataset = quick_norm(test_dataset)
		test_dataset = quick_std(test_dataset)
		
		feat_gyr = (test_dataset[:,:,0:3]).to(device)
		feat_acc = (test_dataset[:,:,3:6]).to(device)
		feat_dt = (test_dataset[:,:,13].unsqueeze(2)).to(device)
		feat_prev_quat = (test_dataset[:,:,9:13]).to(device)

		#---------------------Use the NN to predict relative quaternion and relative velocity--------------------------------------------
		with torch.no_grad():
			measurements_vel, measurements_ori = network(feat_gyr,feat_acc,feat_dt,feat_prev_quat)

			measurements_vel = measurements_vel[:,5,:]
			measurements_ori = measurements_ori[:,5,:]
			label_dataset = label_dataset[:,5,:]

			measurements_ori = torch.div(measurements_ori,torch.norm(measurements_ori,dim=1).unsqueeze(1).repeat(1,4))

			#---------------------Compute the prediction errors--------------------------------------------
			conj_target = conjugate(label_dataset[:,3:7])
			qprod = hamilton_product(measurements_ori,conj_target)

			w,x,y,z = torch.chunk(qprod, 4, dim=-1)
	
			quat_error = 2*torch.sum(torch.abs(torch.cat((x,y,z),axis=-1)),dim=1)

			mse_vel = torch.mean(torch.pow((label_dataset[:,0:3]-measurements_vel),2),dim=0)
			mse_ori = torch.mean(torch.pow((label_dataset[:,3:7]-measurements_ori),2),dim=0)

			print('Error prediction')
			print(mse_vel,mse_ori,torch.mean(quat_error,dim=0))
	
		measurements_vel=measurements_vel.detach().numpy()
		measurements_ori=measurements_ori.detach().numpy()

		#---------------------Multi display--------------------------------------------
		x_ax = int((i%(fig_x*fig_y))/fig_x)
		y_ax = i%fig_y
		axes[x_ax,y_ax].grid(axis='y')
		
		if dataset_name in dataset.datasets_train_filter:
			axes[x_ax,y_ax].set_title(dataset_name, color = 'red')
		else:
			axes[x_ax,y_ax].set_title(dataset_name)
			
		axes[x_ax,y_ax].plot(label_dataset[:,0], color = 'orange' )
		axes[x_ax,y_ax].plot(label_dataset[:,1], color = 'blue' )
		axes[x_ax,y_ax].plot(label_dataset[:,2], color = 'green' )


		axes[x_ax,y_ax].plot(measurements_vel[:,0], color = 'red')
		axes[x_ax,y_ax].plot(measurements_vel[:,1], color = 'c')
		axes[x_ax,y_ax].plot(measurements_vel[:,2], color = 'purple')   
		axes[x_ax,y_ax].set_xlabel('frame')
		axes[x_ax,y_ax].set_ylabel('Relative velocity')
		
	plt.subplots_adjust(left=0.03, bottom=0.03, right=0.97, top=0.97, wspace = 0.12,hspace = 0.34)
	plt.show()

		

def quick_std(tensor):

	dictionary_std = torch.load('../temp/standardization_factor.p')

	input_loc = dictionary_std['input_loc']
	input_std = dictionary_std['input_std']
	
	tensor[:,:,0:6] = (tensor[:,:,0:6]-input_loc[0:6])/input_std[0:6]
	#tensor[:,:,6:9] = (tensor[:,:,6:9]-input_loc[6:9])/input_std[6:9]
	tensor[:,:,9:13] = (tensor[:,:,9:13]-input_loc[9:13])/input_std[9:13]
	#tensor[:,:,13] = (tensor[:,:,13]-input_loc[13])/input_std[13]

	return tensor

def quick_norm(tensor):
	dictionary_norm = torch.load('../../normalization_factor.p')

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

def launch_analysis_velocity(args,dataset):
	simple_test_Velonet(args,dataset)
