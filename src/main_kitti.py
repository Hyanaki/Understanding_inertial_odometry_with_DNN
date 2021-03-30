import os
import numpy as np
from collections import namedtuple
import glob
import time
import datetime
import pickle
import torch
import matplotlib.pyplot as plt
from termcolor import cprint
from navpy import lla2ned
from collections import OrderedDict

from dataset import BaseDataset

from utils.utils import prepare_data

from trainingNetwork.train_Velonet import train_velonet


from Testing.VelocityAnalysis import launch_analysis_velocity


def launch(args):
	#if args.read_data:
	#	args.dataset_class.read_data(args)
	dataset = args.dataset_class(args)

	if args.train_veloNet:
		train_velonet(args,dataset)

	if args.test_veloNet:
		launch_analysis_velocity(args,dataset)


class KITTIDataset(BaseDataset):
	OxtsPacket = namedtuple('OxtsPacket',
			                'lat, lon, alt, ' + 'roll, pitch, yaw, ' + 'vn, ve, vf, vl, vu, '
			                                                           '' + 'ax, ay, az, af, al, '
			                                                                'au, ' + 'wx, wy, wz, '
			                                                                         'wf, wl, wu, '
			                                                                         '' +
			                'pos_accuracy, vel_accuracy, ' + 'navstat, numsats, ' + 'posmode, '
			                                                                      'velmode, '
			                                                                      'orimode')

	# Bundle into an easy-to-access structure
	OxtsData = namedtuple('OxtsData', 'packet, T_w_imu')
	min_seq_dim = 25 * 100  # 60 s
	datasets_fake = ['2011_09_26_drive_0093_extract', '2011_09_28_drive_0039_extract',
			         '2011_09_28_drive_0002_extract']
	


	# training set to the raw data of the KITTI dataset.
	# The following dict lists the name and end frame of each sequence that
	# has been used to extract the visual odometry / SLAM training set
	odometry_benchmark = OrderedDict()
	odometry_benchmark["2011_10_03_drive_0027_extract"] = [0, 45692]
	odometry_benchmark["2011_10_03_drive_0042_extract"] = [0, 12180]
	odometry_benchmark["2011_10_03_drive_0047_extract"] = [0, 8000]
	odometry_benchmark["2011_10_03_drive_0034_extract"] = [0, 47935]
	odometry_benchmark["2011_09_26_drive_0067_extract"] = [0, 8000]
	odometry_benchmark["2011_09_30_drive_0016_extract"] = [0, 2950]
	odometry_benchmark["2011_09_30_drive_0018_extract"] = [0, 28659]
	odometry_benchmark["2011_09_30_drive_0020_extract"] = [0, 11347]
	odometry_benchmark["2011_09_30_drive_0027_extract"] = [0, 11545]
	odometry_benchmark["2011_09_30_drive_0028_extract"] = [11231, 53650]
	odometry_benchmark["2011_09_30_drive_0033_extract"] = [0, 16589]
	odometry_benchmark["2011_09_30_drive_0034_extract"] = [0, 12744]

	odometry_benchmark_img = OrderedDict()
	odometry_benchmark_img["2011_10_03_drive_0027_extract"] = [0, 45400]
	odometry_benchmark_img["2011_10_03_drive_0042_extract"] = [0, 11000]
	odometry_benchmark_img["2011_10_03_drive_0047_extract"] = [0, 8000]
	odometry_benchmark_img["2011_10_03_drive_0034_extract"] = [0, 46600]
	odometry_benchmark_img["2011_09_26_drive_0067_extract"] = [0, 8000]
	odometry_benchmark_img["2011_09_30_drive_0016_extract"] = [0, 2700]
	odometry_benchmark_img["2011_09_30_drive_0018_extract"] = [0, 27600]
	odometry_benchmark_img["2011_09_30_drive_0020_extract"] = [0, 11000]
	odometry_benchmark_img["2011_09_30_drive_0027_extract"] = [0, 11000]
	odometry_benchmark_img["2011_09_30_drive_0028_extract"] = [11000, 51700]
	odometry_benchmark_img["2011_09_30_drive_0033_extract"] = [0, 15900]
	odometry_benchmark_img["2011_09_30_drive_0034_extract"] = [0, 12000]


	#data_border =  pickle.load( open( "Split_data.p", "rb" ) )
	

	def __init__(self, args):
		super(KITTIDataset, self).__init__(args)

		directory_train = '../kitti_data/train_data'
		directory_valid = '../kitti_data/valid_data'

		for filename in os.listdir(directory_train):
			if filename.endswith(".p"):
				self.datasets_train_filter[filename[:-2]] = [0, None]
				t,_,_,_,_ = prepare_data(args,self, filename[:-2])
				self.training_dataset_length += t.shape[0]
			else:
				continue

		for filename in os.listdir(directory_valid):
			if filename.endswith(".p"):
				self.datasets_validatation_filter[filename[:-2]] = [0, None]
				t,_,_,_,_ = prepare_data(args,self, filename[:-2])
				self.valid_dataset_length += t.shape[0]

			else:
				continue
		
		
		print(self.valid_dataset_length,self.training_dataset_length)

	@staticmethod
	def read_data(args):
		"""
		Read the data from the KITTI dataset

		:param args:
		:return:
		"""

		print("Start read_data")
		t_tot = 0  # sum of times for the all dataset
		date_dirs = os.listdir(args.path_data_base)
		for n_iter, date_dir in enumerate(date_dirs):
		    # get access to each sequence
		    path1 = os.path.join(args.path_data_base, date_dir)
		    if not os.path.isdir(path1):
		        continue
		    date_dirs2 = os.listdir(path1)

		    for date_dir2 in date_dirs2:
		        path2 = os.path.join(path1, date_dir2)
		        if not os.path.isdir(path2):
		            continue
		        # read data
		        oxts_files = sorted(glob.glob(os.path.join(path2, 'oxts', 'data', '*.txt')))
		        oxts = KITTIDataset.load_oxts_packets_and_poses(oxts_files)

		        """ Note on difference between ground truth and oxts solution:
		            - orientation is the same
		            - north and east axis are inverted
		            - position are closed to but different
		            => oxts solution is not loaded
		        """

		        print("\n Sequence name : " + date_dir2)
		        past_data = 200
		        if len(oxts) < KITTIDataset.min_seq_dim:  #  sequence shorter than 30 s are rejected
		            cprint("Dataset is too short ({:.2f} s)".format(len(oxts) / 100), 'yellow')
		            continue
		        lat_oxts = np.zeros(len(oxts))
		        lon_oxts = np.zeros(len(oxts))
		        alt_oxts = np.zeros(len(oxts))
		        roll_oxts = np.zeros(len(oxts))
		        pitch_oxts = np.zeros(len(oxts))
		        yaw_oxts = np.zeros(len(oxts))
		        roll_gt = np.zeros(len(oxts))
		        pitch_gt = np.zeros(len(oxts))
		        yaw_gt = np.zeros(len(oxts))
		        t = KITTIDataset.load_timestamps(path2)
		        acc = np.zeros((len(oxts), 3))
		        acc_bis = np.zeros((len(oxts), 3))
		        gyro = np.zeros((len(oxts), 3))
		        gyro_bis = np.zeros((len(oxts), 3))
		        p_gt = np.zeros((len(oxts), 3))
		        v_gt = np.zeros((len(oxts), 3))
		        v_rob_gt = np.zeros((len(oxts), 3))

		        k_max = len(oxts)
		        for k in range(k_max):
		            oxts_k = oxts[k]
		            t[k] = 3600 * t[k].hour + 60 * t[k].minute + t[k].second + t[
		                k].microsecond / 1e6
		            lat_oxts[k] = oxts_k[0].lat
		            lon_oxts[k] = oxts_k[0].lon
		            alt_oxts[k] = oxts_k[0].alt
		            acc[k, 0] = oxts_k[0].af
		            acc[k, 1] = oxts_k[0].al
		            acc[k, 2] = oxts_k[0].au
		            acc_bis[k, 0] = oxts_k[0].ax
		            acc_bis[k, 1] = oxts_k[0].ay
		            acc_bis[k, 2] = oxts_k[0].az
		            gyro[k, 0] = oxts_k[0].wf
		            gyro[k, 1] = oxts_k[0].wl
		            gyro[k, 2] = oxts_k[0].wu
		            gyro_bis[k, 0] = oxts_k[0].wx
		            gyro_bis[k, 1] = oxts_k[0].wy
		            gyro_bis[k, 2] = oxts_k[0].wz
		            roll_oxts[k] = oxts_k[0].roll
		            pitch_oxts[k] = oxts_k[0].pitch
		            yaw_oxts[k] = oxts_k[0].yaw
		            v_gt[k, 0] = oxts_k[0].ve
		            v_gt[k, 1] = oxts_k[0].vn
		            v_gt[k, 2] = oxts_k[0].vu
		            v_rob_gt[k, 0] = oxts_k[0].vf
		            v_rob_gt[k, 1] = oxts_k[0].vl
		            v_rob_gt[k, 2] = oxts_k[0].vu
		            p_gt[k] = oxts_k[1][:3, 3]
		            Rot_gt_k = oxts_k[1][:3, :3]
		            roll_gt[k], pitch_gt[k], yaw_gt[k] = to_rpy(Rot_gt_k)

		        t0 = t[0]
		        t = np.array(t) - t[0]
		        # some data can have gps out
		        if np.max(t[:-1] - t[1:]) > 0.1:
		            cprint(date_dir2 + " has time problem", 'yellow')
		        ang_gt = np.zeros((roll_gt.shape[0], 3))
		        ang_gt[:, 0] = roll_gt
		        ang_gt[:, 1] = pitch_gt
		        ang_gt[:, 2] = yaw_gt

		        p_oxts = lla2ned(lat_oxts, lon_oxts, alt_oxts, lat_oxts[0], lon_oxts[0],
		                         alt_oxts[0], latlon_unit='deg', alt_unit='m', model='wgs84')
		        p_oxts[:, [0, 1]] = p_oxts[:, [1, 0]]  # see note

		        # take correct imu measurements
		        u = np.concatenate((gyro_bis, acc_bis), -1)
		        # convert from numpy
		        t = torch.from_numpy(t)
		        p_gt = torch.from_numpy(p_gt)
		        v_gt = torch.from_numpy(v_gt)
		        ang_gt = torch.from_numpy(ang_gt)
		        u = torch.from_numpy(u)
		        
		        t_input = torch.zeros(t.shape[0]-past_data,past_data)
		        p_gt_input = torch.zeros(p_gt.shape[0]-past_data,past_data,p_gt.shape[1])
		        v_gt_input = torch.zeros(v_gt.shape[0]-past_data,past_data,v_gt.shape[1])
		        ang_gt_input = torch.zeros(ang_gt.shape[0]-past_data,past_data,ang_gt.shape[1])
		        u_input = torch.zeros(u.shape[0]-past_data,past_data,u.shape[1])
		        
		        for j in range(200,u.shape[0]):
		            index = j-200
		            u_input[index,:,:] = u[index:j,:]
		            p_gt_input[index,:,:] = p_gt[index:j,:]
		            v_gt_input[index,:,:] = v_gt[index:j,:]
		            ang_gt_input[index,:,:] = ang_gt[index:j,:]
		            t_input[index,:] = t[index:j]
		        # convert to float
		        t = t.float()
		        u = u.float()
		        p_gt = p_gt.float()
		        ang_gt = ang_gt.float()
		        v_gt = v_gt.float()
		        u_input = u_input.float()
		        v_gt_input = v_gt_input.float()

		        #print(u_input.shape)
		        mondict = {
		            't': t, 'p_gt': p_gt, 'ang_gt': ang_gt, 'v_gt': v_gt,
		            'u': u, 'name': date_dir2, 't0': t0
		            }

		        t_tot += t[-1] - t[0]
		        KITTIDataset.dump(mondict, args.path_data_save, date_dir2)
		print("\n Total dataset duration : {:.2f} s".format(t_tot))

	@staticmethod
	def rotx(t):
		"""Rotation about the x-axis."""
		c = np.cos(t)
		s = np.sin(t)
		return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

	@staticmethod
	def roty(t):
		"""Rotation about the y-axis."""
		c = np.cos(t)
		s = np.sin(t)
		return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

	@staticmethod
	def rotz(t):
		"""Rotation about the z-axis."""
		c = np.cos(t)
		s = np.sin(t)
		return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

	@staticmethod
	def to_rpy(Rot):
		pitch = np.arctan2(-Rot[2, 0], np.sqrt(Rot[0, 0]**2 + Rot[1, 0]**2))

		if np.isclose(pitch, np.pi / 2.):
			yaw = 0.
			roll = np.arctan2(Rot[0, 1], Rot[1, 1])
		elif np.isclose(pitch, -np.pi / 2.):
			yaw = 0.
			roll = -np.arctan2(Rot[0, 1], Rot[1, 1])
		else:
			sec_pitch = 1. / np.cos(pitch)
			yaw = np.arctan2(Rot[1, 0] * sec_pitch,
				             Rot[0, 0] * sec_pitch)
			roll = np.arctan2(Rot[2, 1] * sec_pitch,
				              Rot[2, 2] * sec_pitch)
		return roll, pitch, yaw



	@staticmethod
	def pose_from_oxts_packet(packet, scale):
		"""Helper method to compute a SE(3) pose matrix from an OXTS packet.
		"""
		er = 6378137.  # earth radius (approx.) in meters

		# Use a Mercator projection to get the translation vector
		tx = scale * packet.lon * np.pi * er / 180.
		ty = scale * er * np.log(np.tan((90. + packet.lat) * np.pi / 360.))
		tz = packet.alt
		t = np.array([tx, ty, tz])

		# Use the Euler angles to get the rotation matrix
		Rx = KITTIDataset.rotx(packet.roll)
		Ry = KITTIDataset.roty(packet.pitch)
		Rz = KITTIDataset.rotz(packet.yaw)
		R = Rz.dot(Ry.dot(Rx))

		# Combine the translation and rotation into a homogeneous transform
		return R, t

	@staticmethod
	def transform_from_rot_trans(R, t):
		"""Transformation matrix from rotation matrix and translation vector."""
		R = R.reshape(3, 3)
		t = t.reshape(3, 1)
		return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))

	@staticmethod
	def load_oxts_packets_and_poses(oxts_files):
		"""Generator to read OXTS ground truth data.
		   Poses are given in an East-North-Up coordinate system
		   whose origin is the first GPS position.
		"""
		# Scale for Mercator projection (from first lat value)
		scale = None
		# Origin of the global coordinate system (first GPS position)
		origin = None

		oxts = []

		for filename in oxts_files:
		    with open(filename, 'r') as f:
		        for line in f.readlines():
		            line = line.split()
		            # Last five entries are flags and counts
		            line[:-5] = [float(x) for x in line[:-5]]
		            line[-5:] = [int(float(x)) for x in line[-5:]]

		            packet = KITTIDataset.OxtsPacket(*line)

		            if scale is None:
		                scale = np.cos(packet.lat * np.pi / 180.)

		            R, t = KITTIDataset.pose_from_oxts_packet(packet, scale)

		            if origin is None:
		                origin = t

		            T_w_imu = KITTIDataset.transform_from_rot_trans(R, t - origin)

		            oxts.append(KITTIDataset.OxtsData(packet, T_w_imu))
		return oxts

	@staticmethod
	def load_timestamps(data_path):
		"""Load timestamps from file."""
		timestamp_file = os.path.join(data_path, 'oxts', 'timestamps.txt')

		# Read and parse the timestamps
		timestamps = []
		with open(timestamp_file, 'r') as f:
		    for line in f.readlines():
		        # NB: datetime only supports microseconds, but KITTI timestamps
		        # give nanoseconds, so need to truncate last 4 characters to
		        # get rid of \n (counts as 1) and extra 3 digits
		        t = datetime.datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
		        timestamps.append(t)
		return timestamps

class KITTIArgs():


	path_data_save = '../kitti_data'

	cpu = False

	path_normalization_factor  =  '../temp/normalization_factor.p'
	path_standardization_factor  =  '../temp/standardization_factor.p'
	continue_training = True

	# choose what to do
	#read_data = 0

	train_veloNet =0
	test_veloNet =1
	dataset_class = KITTIDataset

if __name__ == '__main__':
	args = KITTIArgs()
	dataset = KITTIDataset(args)
	launch(KITTIArgs)

