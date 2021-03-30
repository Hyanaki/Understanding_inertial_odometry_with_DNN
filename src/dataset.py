from torch.utils.data.dataset import Dataset
import torch
import numpy as np
from termcolor import cprint
import pickle
import matplotlib.pyplot as plt
import os
from collections import OrderedDict


class BaseDataset(Dataset):
	pickle_extension = ".p"
	"""extension of the file saved in pickle format"""
	file_normalize_factor = "normalize_factors.p"
	"""name of file for normalizing input"""

	def __init__(self, args):
		# paths
		self.path_data_save = args.path_data_save

		self.training_path = self.path_data_save + '/train_data'
		self.validation_path = self.path_data_save + '/valid_data'


		self.normalization_path = args.path_normalization_factor
		self.standardization_path = args.path_standardization_factor

		# names of the sequences
		self.datasets = []
		"""dataset names"""
		self.datasets_train = []
		"""train datasets"""

		self.datasets_validatation_filter = OrderedDict()
		"""Validation dataset with index for starting/ending"""
		self.datasets_train_filter = OrderedDict()
		"""Validation dataset with index for starting/ending"""

		self.training_dataset_length = 0
		self.valid_dataset_length = 0
		# noise added to the data
		self.sigma_gyro = 1.e-4
		self.sigma_acc = 1.e-4
		self.sigma_b_gyro = 1.e-5
		self.sigma_b_acc = 1.e-4

		# number of training data points
		self.num_data = 0

		# factors for normalizing inputs
		#self.normalize_factors = None
		self.get_datasets()
		#self.set_normalize_factors()

	def __getitem__(self, i):
		if(self.datasets[i] in self.datasets_validatation_filter):
			mondict = self.load(self.validation_path, self.datasets[i])
		else:
			mondict = self.load(self.training_path, self.datasets[i])
		return mondict

	def __len__(self):
		return len(self.datasets)

	def get_datasets(self):
		for dataset in os.listdir(self.training_path):
			self.datasets += [dataset[:-2]]  # take just name, remove the ".p"
		for dataset in os.listdir(self.validation_path):
			self.datasets += [dataset[:-2]]




	def dataset_name(self, i):
		return self.datasets[i]

	def get_data(self, i):
		pickle_dict = self[self.datasets.index(i) if type(i) != int else i]
		
		return pickle_dict['t'], pickle_dict['ang_gt'], pickle_dict['p_gt'], pickle_dict['v_gt'],\
		       pickle_dict['u']



	def add_noise(self, u):
	#return a tensor with the same size as the input that is filled with random numbers from a normal distribution with mean 0 and variance 1
		w = torch.randn_like(u[:, :6]) # noise
		w_b = torch.randn_like(u[0, :6])  # bias
		w[:, :3] *= self.sigma_gyro
		w[:, 3:6] *= self.sigma_acc
		w_b[:3] *= self.sigma_b_gyro
		w_b[3:6] *= self.sigma_b_acc
		u[:, :6] += w + w_b
		return u


	@staticmethod
	def read_data(args):
		raise NotImplementedError

	@classmethod
	#Load pickle file
	def load(cls, *_file_name):
		file_name = os.path.join(*_file_name)
		if not file_name.endswith(cls.pickle_extension):
		    file_name += cls.pickle_extension
		with open(file_name, "rb") as file_pi:
		    pickle_dict = pickle.load(file_pi)
		return pickle_dict

	@classmethod
	#This method is used to put an object in an opened file 
	def dump(cls, mondict, *_file_name):
		file_name = os.path.join(*_file_name)
		if not file_name.endswith(cls.pickle_extension):
		    file_name += cls.pickle_extension
		with open(file_name, "wb") as file_pi:
		    pickle.dump(mondict, file_pi)


