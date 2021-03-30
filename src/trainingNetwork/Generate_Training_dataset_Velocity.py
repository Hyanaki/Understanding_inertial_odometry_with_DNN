import torch
from utils.utils import prepare_data
import random
import numpy as np
import os
import pickle
import math

from utils.utils_quaternion import *

import pickle

np.set_printoptions(formatter={'float': lambda x: "{0:0.6f}".format(x)})
torch.set_printoptions(precision=10)


def create_dataset_Relative_Kinematic(args,dataset,windows_size):

	#---------------------Initialization of the tensors--------------------------------------------
	number_input = 14
	training_data = torch.zeros((dataset.training_dataset_length,windows_size,number_input))
	training_label_data = torch.zeros((dataset.training_dataset_length,windows_size,7))

	valid_data = torch.zeros((dataset.valid_dataset_length,windows_size,number_input))
	valid_label_data = torch.zeros((dataset.valid_dataset_length,windows_size,7))
	
	training_index = 0
	valid_index = 0

	training_dataset = dataset.datasets_train_filter

	
	for i in range(0,len(dataset)):
		dataset_name = dataset.dataset_name(i)
		print(dataset_name)
		#---------------------load the track--------------------------------------------
		t, ang_gt, _ , v_gt, u = prepare_data(args,dataset, dataset_name)

		if(not os.path.exists(args.path_normalization_factor)):
			normalization_all_data(args,dataset)
		if(not os.path.exists(args.path_standardization_factor)):
			standardization_all_data(args,dataset)


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

		#---------------------Attribution of the values for the training tensor and testing tensor--------------------------------------------
		if(dataset_name in training_dataset):
			for j in range(windows_size,u.shape[0]):
					
					index = j-windows_size 
					
					training_data[training_index,:,0:3] = u[index:j,0:3]
					training_data[training_index,:,3:6] = u[index:j,3:6]
					training_data[training_index,:,6:9] = v_gt[index:j,0:3]
					training_data[training_index,:,9:13] = quat_tensor[index:j,0:4]
					training_data[training_index,:,13] = dt[index:j]

					training_label_data[training_index,:,0:3]= d_vgt[index:j]
					training_label_data[training_index,:,3:7]= relative_quaternion[index:j,0:4]
					training_index +=1
		else :
			for j in range(windows_size,u.shape[0]):
					
					index = j-windows_size 
					
					valid_data[valid_index,:,0:3] = u[index:j,0:3]
					valid_data[valid_index,:,3:6] = u[index:j,3:6]
					valid_data[valid_index,:,6:9] = v_gt[index:j,0:3]
					valid_data[valid_index,:,9:13] = quat_tensor[index:j,0:4]
					valid_data[valid_index,:,13] = dt[index:j]

					valid_label_data[valid_index,:,0:3]= d_vgt[index:j]
					valid_label_data[valid_index,:,3:7]= relative_quaternion[index:j,0:4]
					valid_index +=1
	
	#---------------------Adjusting the size of the tensor and shuffling--------------------------------------------
	training_data = training_data[:training_index]
	training_label_data = training_label_data[:training_index]
	valid_data = valid_data[:valid_index]
	valid_label_data = valid_label_data[:valid_index]

	randomize = shuffle_tensor(training_data.shape[0])
	training_data = training_data[randomize]
	training_label_data = training_label_data[randomize]

	randomize = shuffle_tensor(valid_data.shape[0])
	valid_data = valid_data[randomize]
	valid_label_data = valid_label_data[randomize]

	mondict = {
	    'training_data': training_data, 'training_label_data': training_label_data, 'valid_data': valid_data, 'valid_label_data': valid_label_data
	    }
	path_data = '../Relative_kinematic_training.p'          
	torch.save(mondict,path_data)

	return training_data, training_label_data,valid_data,valid_label_data 

#shuffle the tensor
def shuffle_tensor(tensor_shape):
	seeds = 11
	np.random.seed(seeds)
	randomize = np.arange(tensor_shape)
	np.random.shuffle(randomize)
	return randomize

#Compute the normalization factors for each input
def normalization_all_data(args,dataset):



	#---------------------Initialization of the normalization factors----------------------------------------------------------------
	nb_data = 0

	max_quat = [0,0,0,0] #We initialize with a random low values
	min_quat = [20,20,20,20] #We initialize with a random high values

	max_gyr = [0,0,0]
	min_gyr = [20,20,20]

	max_acc = [0,0,0]
	min_acc = [20,20,20]
	
	max_gyr_dt = [0,0,0]
	min_gyr_dt = [20,20,20]

	max_acc_dt = [0,0,0]
	min_acc_dt = [20,20,20]

	max_acc_rot = [0,0,0]
	min_acc_rot = [20,20,20]

	max_prev_vel = 0
	min_prev_vel = 20

	max_velocity = [0,0,0]
	min_velocity = [20,20,20]

	for i in range(0,len(dataset)):

		#---------------------Load the track----------------------------------------------------------------
		dataset_name = dataset.dataset_name(i)
		print(dataset_name)
		t, ang_gt, p_gt, v_gt, u = prepare_data(args,dataset, dataset_name)
		dt = (t[1:]-t[:-1]).unsqueeze(1)

		nb_data+= (u.shape[0]-1)
		u_dt = u[1:,0:6].mul(dt[:])
		norm_v_gt = torch.norm(v_gt,dim=1)
		ang_gt = correct_ang_gt(ang_gt)
		

		#---------------------Smoothing ground truth quaternion--------------------------------------------
		quat_tensor = torch.zeros(u.shape[0],4)

		for j in range(0,quat_tensor.shape[0]):
			quat_tensor[j,:] = euler_to_quaternion(ang_gt[j,0], ang_gt[j,1], ang_gt[j,2])

		quat_tensor[2:-2,:] = (quat_tensor[0:-4,:]+quat_tensor[1:-3,:]+quat_tensor[2:-2,:]+quat_tensor[3:-1,:]+quat_tensor[4:,:])/5
		quat_tensor = torch.div(quat_tensor,torch.norm(quat_tensor,dim=1).unsqueeze(1).repeat(1,4))

		rot_matrix = as_rotation_matrix(quat_tensor[:-1]) 
		macc = mult_acc(rot_matrix,u[1:,3:6]) 

		#---------------------Get min-max values--------------------------------------------
		velocity_max = [v_gt[:,0].max(),v_gt[:,1].max(),v_gt[:,2].max()]
		velocity_min = [v_gt[:,0].min(),v_gt[:,1].min(),v_gt[:,2].min()]
		print(velocity_max)
		print(velocity_min)
		u_values_max = [u[:,0].max(),u[:,1].max(),u[:,2].max(),u[:,3].max(),u[:,4].max(),u[:,5].max()]
		u_values_min = [u[:,0].min(),u[:,1].min(),u[:,2].min(),u[:,3].min(),u[:,4].min(),u[:,5].min()]

		quat_tensor_max = [quat_tensor[:,0].max(),quat_tensor[:,1].max(),quat_tensor[:,2].max(),quat_tensor[:,3].max()]
		quat_tensor_min = [quat_tensor[:,0].min(),quat_tensor[:,1].min(),quat_tensor[:,2].min(),quat_tensor[:,3].min()]

		u_dt_values_max = [u_dt[:,0].max(),u_dt[:,1].max(),u_dt[:,2].max(),u_dt[:,3].max(),u_dt[:,4].max(),u_dt[:,5].max()]
		u_dt_values_min = [u_dt[:,0].min(),u_dt[:,1].min(),u_dt[:,2].min(),u_dt[:,3].min(),u_dt[:,4].min(),u_dt[:,5].min()]

		vel_max = norm_v_gt.max()
		vel_min = norm_v_gt.min()

		rotacc_values_max = [macc[:,0].max(),macc[:,1].max(),macc[:,2].max()]
		rotacc_values_min = [macc[:,0].min(),macc[:,1].min(),macc[:,2].min()]
		
		#print(vel_min,vel_max)
		#---------------------Update the min and max values--------------------------------------------
		if vel_min<min_prev_vel:
			min_prev_vel = vel_min
		if vel_max>max_prev_vel:
			max_prev_vel = vel_max

		if quat_tensor_min[3]<min_quat[3]:
			min_quat[3] = quat_tensor_min[3]			
		if quat_tensor_max[3]>max_quat[3]:
			max_quat[3] = quat_tensor_max[3]

		for i in range(3):
			if u_values_min[i]<min_gyr[i]:
				min_gyr[i] = u_values_min[i]
			if u_values_max[i]>max_gyr[i]:
				max_gyr[i] = u_values_max[i]

			if u_values_min[3+i]<min_acc[i]:
				min_acc[i] = u_values_min[3+i]
			if u_values_max[3+i]>max_acc[i]:
				max_acc[i] = u_values_max[3+i]

			if rotacc_values_min[i]<min_acc_rot[i]:
				min_acc_rot[i] = rotacc_values_min[i]
			if rotacc_values_max[i]>max_acc_rot[i]:
				max_acc_rot[i] = rotacc_values_max[i]

			if u_dt_values_min[i]<min_gyr_dt[i]:
				min_gyr_dt[i] = u_dt_values_min[i]
			if u_dt_values_max[i]>max_gyr_dt[i]:
				max_gyr_dt[i] = u_dt_values_max[i]

			if u_dt_values_min[3+i]<min_acc_dt[i]:
				min_acc_dt[i] = u_dt_values_min[3+i]
			if u_dt_values_max[3+i]>max_acc_dt[i]:
				max_acc_dt[i] = u_dt_values_max[3+i]

			if quat_tensor_min[i]<min_quat[i]:
				min_quat[i] = quat_tensor_min[i]			
			if quat_tensor_max[i]>max_quat[i]:
				max_quat[i] = quat_tensor_max[i]
		
			if velocity_min[i]<min_velocity[i]:
				min_velocity[i] = velocity_min[i]
			if velocity_max[i]>max_velocity[i]:
				max_velocity[i] = velocity_max[i]

	#---------------------Saving the factors in dictionary--------------------------------------------
	mondict = {
	    'max_quat' : max_quat, 'min_quat' : min_quat, 'max_gyr' : max_gyr, 'min_gyr' : min_gyr, 'max_acc' : max_acc, 'min_acc' : min_acc, 'max_gyr_dt' : max_gyr_dt,'min_gyr_dt' : min_gyr_dt,
	'max_prev_vel' : max_prev_vel, 'min_prev_vel' : min_prev_vel, 'max_acc_rot' : max_acc_rot, 'min_acc_rot' : min_acc_rot, 'max_acc_dt' : max_acc_dt, 'min_acc_dt' : min_acc_dt, 'min_velocity' : min_velocity, 'max_velocity' : max_velocity }
	
	path_data = args.path_normalization_factor

	torch.save(mondict,path_data)

	print("Test")
	print(max_velocity,min_velocity)
	return

#Compute the standardization factors for each input
def standardization_all_data(args,dataset):
	nb_data = 0
	nb_data_dt = 0
	#---------------------Compute X - mean  factor----------------------------------------------------------------
	for i in range(0,len(dataset)):
		#---------------------Load the track----------------------------------------------------------------
		dataset_name = dataset.dataset_name(i)
		print(dataset_name)

		t, ang_gt, _, v_gt, u = prepare_data(args,dataset, dataset_name)
		dt = (t[1:]-t[:-1]).unsqueeze(1)

		nb_data += u.shape[0]
		nb_data_dt += dt.shape[0]

		ang_gt = correct_ang_gt(ang_gt)
		#---------------------Smoothing ground truth quaternion--------------------------------------------
		quat_tensor = torch.zeros(u.shape[0],4)

		for j in range(0,quat_tensor.shape[0]):
			quat_tensor[j,:] = euler_to_quaternion(ang_gt[j,0], ang_gt[j,1], ang_gt[j,2])

		quat_tensor[2:-2,:] = (quat_tensor[0:-4,:]+quat_tensor[1:-3,:]+quat_tensor[2:-2,:]+quat_tensor[3:-1,:]+quat_tensor[4:,:])/5
		quat_tensor = torch.div(quat_tensor,torch.norm(quat_tensor,dim=1).unsqueeze(1).repeat(1,4))
		

		if i==0:
			u_input_loc = torch.sum(u[:],dim=0)
			v_gt_input_loc = torch.sum(v_gt[:],dim=0)
			quat_tensor_loc = torch.sum(quat_tensor[:,0:4],dim=0)
			dt_loc = torch.sum(dt,dim=0)
			
		else:
			u_input_loc += torch.sum(u[:],dim=0)
			v_gt_input_loc += torch.sum(v_gt[:],dim=0)
			quat_tensor_loc += torch.sum(quat_tensor[:,0:4],dim=0)
			dt_loc += torch.sum(dt,dim=0)

	
	u_input_loc = u_input_loc/nb_data
	v_gt_input_loc = v_gt_input_loc/nb_data
	quat_tensor_loc = quat_tensor_loc/nb_data
	dt_loc = dt_loc/nb_data_dt
	print(u_input_loc.shape,v_gt_input_loc.shape,quat_tensor_loc.shape,dt_loc.shape)

	input_loc = torch.cat((u_input_loc,v_gt_input_loc,quat_tensor_loc,dt_loc),dim=0)



	#---------------------Compute the standard deviation factor----------------------------------------------------------------
	for i in range(0,len(dataset)):
		#---------------------Load the track----------------------------------------------------------------
		dataset_name = dataset.dataset_name(i)
		print(dataset_name)
		t, ang_gt, p_gt, v_gt, u = prepare_data(args,dataset, dataset_name)

		dt = (t[1:]-t[:-1]).unsqueeze(1)
		
		ang_gt = correct_ang_gt(ang_gt)
		
		#---------------------Smoothing ground truth quaternion--------------------------------------------
		quat_tensor = torch.zeros(u.shape[0],4)

		for j in range(0,quat_tensor.shape[0]):
			quat_tensor[j,:] = euler_to_quaternion(ang_gt[j,0], ang_gt[j,1], ang_gt[j,2])
			
		quat_tensor[2:-2,:] = (quat_tensor[0:-4,:]+quat_tensor[1:-3,:]+quat_tensor[2:-2,:]+quat_tensor[3:-1,:]+quat_tensor[4:,:])/5
		quat_tensor = torch.div(quat_tensor,torch.norm(quat_tensor,dim=1).unsqueeze(1).repeat(1,4))


		if i == 0:
			u_input_std = ((u[:] - input_loc[0:6]) ** 2).sum(dim=0)
			v_gt_std = ((v_gt[:] - input_loc[6:9]) ** 2).sum(dim=0)
			quat_tensor_std = ((quat_tensor[:,0:4] - input_loc[9:13]) ** 2).sum(dim=0)
			dt_std = ((dt - input_loc[13]) ** 2).sum(dim=0)
			
			
		else:
			u_input_std += ((u[:] - input_loc[0:6]) ** 2).sum(dim=0)
			v_gt_std += ((v_gt[:] - input_loc[6:9]) ** 2).sum(dim=0)
			quat_tensor_std += ((quat_tensor[:,0:4] - input_loc[9:13]) ** 2).sum(dim=0)
			dt_std += ((dt - input_loc[13]) ** 2).sum(dim=0)
		
	u_input_std = (u_input_std/nb_data).sqrt()
	v_gt_std = (v_gt_std/nb_data).sqrt()
	quat_tensor_std = (quat_tensor_std/nb_data).sqrt()
	dt_std = (dt_std/nb_data_dt).sqrt()

	#---------------------Saving the factors in dictionary--------------------------------------------
	input_std = torch.cat((u_input_std,v_gt_std,quat_tensor_std,dt_std),dim=0)

	mondict = {
	    'input_loc': input_loc, 'input_std': input_std
	    }

	path_data = args.path_standardization_factor        
	torch.save(mondict,path_data)

	return

#compute the multiplication between the acceleration and the rotation matrix
def mult_acc(rot_mat,acc):

	out_acc = torch.zeros((acc.shape[1],acc.shape[0]))
	acc = acc.transpose(0,1)
	for i in range(0,acc.shape[1]):
		out_acc[:,i] = torch.mm(rot_mat[i,:,:],acc[:,i].unsqueeze(1)).squeeze(1)		
	return out_acc.transpose(0,1)



