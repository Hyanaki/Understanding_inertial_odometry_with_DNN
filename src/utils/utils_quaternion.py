import torch
import numpy as np
from utils.quaternions import hamilton_product
import math



def euler_to_quaternion(roll, pitch, yaw):

	qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
	qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
	qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
	qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

	return torch.tensor([qw,qx, qy, qz])


def quaternion_to_euler(quaternion):

	euler_angle = np.zeros((quaternion.shape[0],3))
	quaternion = quaternion.numpy()

	w = quaternion[:,0]
	x = quaternion[:,1]
	y = quaternion[:,2]
	z = quaternion[:,3]

	"""Converts quaternions with components w, x, y, z into a tuple (roll, pitch, yaw)"""
	sinr_cosp = 2 * (w * x + y * z)
	cosr_cosp = 1 - 2 * (x**2 + y**2)
	euler_angle[:,0] = (np.arctan2(sinr_cosp, cosr_cosp))
	sinp = 2 * (w * y - z * x)
	euler_angle[:,1] = (np.where(np.abs(sinp) >= 1,
		     np.sign(sinp) * np.pi / 2,
		     np.arcsin(sinp)))

	siny_cosp = 2 * (w * z + x * y)
	cosy_cosp = 1 - 2 * (y**2 + z**2)
	euler_angle[:,2] = (np.arctan2(siny_cosp, cosy_cosp))

	return torch.from_numpy(euler_angle)

#Compute the relative rotation between 2 frames at frequency=100Hz
def relative_rotation(rotation,frequencie=100):
	initial_frequencie = 100
	frequencie = int(initial_frequencie/frequencie)

	relative_q = torch.zeros((rotation.shape[0]-frequencie,rotation.shape[1]))
	for i in range(rotation.shape[0]-frequencie):
		q_a = rotation[i].unsqueeze(0)
		q_b = rotation[i+frequencie].unsqueeze(0)

		hamilton_value = hamilton_product(conjugate(q_a),q_b)
		
		if hamilton_value[0,0]<0:
			relative_q[i] = -1*hamilton_value
		else:
			relative_q[i] = hamilton_value
	
	return relative_q

   
#Compute the conjugate of a quaternion
def conjugate(quat,device=None):
    if(device):
        mul_tensor = torch.tensor(([1.0,-1.0,-1.0,-1.0])).to(device)
    else:
        mul_tensor = torch.tensor(([1.0,-1.0,-1.0,-1.0]))
    return torch.mul(quat,mul_tensor)



def force_uniqueness(q):
    if(abs(q[0,0].item())> 1e-05 and q[0,0].item()<0):
        return -q
        
    elif (abs(q[0,1].item())> 1e-05 and q[0,1].item()<0):
        return -q
    elif (abs(q[0,2].item())> 1e-05 and q[0,2].item()<0):
        return -q
    elif (abs(q[0,3].item())> 1e-05 and q[0,3].item()<0):
        return -q
    else:
        return q


def isclose(mat1, mat2, tol=1e-10):
    return (mat1 - mat2).abs().lt(tol)

def to_rpy(Rot):
        """Convert a rotation matrix to RPY Euler angles."""

        pitch = torch.atan2(-Rot[2, 0], torch.sqrt(Rot[0, 0]**2 + Rot[1, 0]**2))

        if isclose(pitch, np.pi / 2.):
            yaw = pitch.new_zeros(1)
            roll = torch.atan2(Rot[0, 1], Rot[1, 1])
        elif isclose(pitch, -np.pi / 2.):
            yaw = pitch.new_zeros(1)
            roll = -torch.atan2(Rot[0, 1],  Rot[1, 1])
        else:
            sec_pitch = 1. / pitch.cos()
            yaw = torch.atan2(Rot[1, 0] * sec_pitch, Rot[0, 0] * sec_pitch)
            roll = torch.atan2(Rot[2, 1] * sec_pitch, Rot[2, 2] * sec_pitch)
        return [roll, pitch, yaw]

def approximRelativeRotation(gyroscope,dt):
	RelativeRotation = []
	
	Rot = np.identity(3)
	for i in range (0,gyroscope.shape[0]-1):
		#print(gyroscope[i])
		Rot[0,1] = -1*gyroscope[i,2]*dt[i]
		Rot[0,2] = gyroscope[i,1]*dt[i]
		Rot[1,0] = gyroscope[i,2]*dt[i]
		Rot[1,2] = -1*gyroscope[i,0]*dt[i]
		Rot[2,0] = -1*gyroscope[i,1]*dt[i]
		Rot[2,1] = gyroscope[i,0]*dt[i]
		RelativeRotation.append(Rot.copy())
		#print(Rot)
	#print(RelativeRotation)
	return RelativeRotation


def as_rotation_matrix(q):
	rot_matrix = torch.zeros((q.shape[0],3,3))
	rot_matrix[:,0,0] = 1.0 - 2*(q[:, 2]**2 + q[:, 3]**2)
	rot_matrix[:,0,1] = 2*(q[:, 1]*q[:, 2] - q[:, 3]*q[:, 0])
	rot_matrix[:,0,2] = 2*(q[:, 1]*q[:, 3] + q[:, 2]*q[:, 0])
	rot_matrix[:,1,0] = 2*(q[:, 1]*q[:, 2] + q[:, 3]*q[:, 0])
	rot_matrix[:,1,1] = 1.0 - 2*(q[:, 1]**2 + q[:, 3]**2)
	rot_matrix[:,1,2] = 2*(q[:, 2]*q[:, 3] - q[:, 1]*q[:, 0])
	rot_matrix[:,2,0] = 2*(q[:, 1]*q[:, 3] - q[:, 2]*q[:, 0])
	rot_matrix[:,2,1] = 2*(q[:, 2]*q[:, 3] + q[:, 1]*q[:, 0])
	rot_matrix[:,2,2] = 1.0 - 2*(q[:, 1]**2 + q[:, 2]**2)
	return rot_matrix


#Extend the range of euler angles
def correct_ang_gt(ang_gt):
	for j in range(1,ang_gt.shape[0]):
		sign_j = np.sign(ang_gt[j,2].item())
		sign_prev_j = np.sign(ang_gt[j-1,2].item())
		
		if(torch.abs(ang_gt[j,2]-ang_gt[j-1,2])>5):
			if(sign_prev_j==1):
				ang_gt[j,2] += 2*math.pi
			if(sign_prev_j==-1):
				ang_gt[j,2] -= 2*math.pi
		sign_j = np.sign(ang_gt[j,0].item())
		sign_prev_j = np.sign(ang_gt[j-1,0].item())
		if(torch.abs(ang_gt[j,0]-ang_gt[j-1,0])>5):
			if(sign_prev_j==1):
				ang_gt[j,0] += 2*math.pi
			if(sign_prev_j==-1):
				ang_gt[j,0] -= 2*math.pi
	return ang_gt

'''
def correct_ang_gt_euroc(ang_gt):
	turn_number = 1
	ang_gt_copy = ang_gt.clone().detach()
	for j in range(1,ang_gt.shape[0]):
		sign_j = np.sign(ang_gt[j,2].item())
		sign_prev_j = np.sign(ang_gt[j-1,2].item())

		difference = torch.abs(ang_gt[j,2]-ang_gt[j-1,2])
		if(difference<math.pi):
			if (ang_gt[j,2]>ang_gt[j-1,2]):
				ang_gt_copy[j,2] = ang_gt_copy[j-1,2]+difference
			else:
				ang_gt_copy[j,2] = ang_gt_copy[j-1,2]-difference

		else:
			if sign_prev_j==1:
				ang_gt_copy[j,2] = ang_gt_copy[j-1,2]  + torch.abs(math.pi - ang_gt[j-1,2]) + torch.abs(ang_gt[j,2] - (-math.pi))
			else:
				ang_gt_copy[j,2] = ang_gt_copy[j-1,2]  - (torch.abs(math.pi - ang_gt[j,2]) + torch.abs(ang_gt[j-1,2] -(-math.pi)))

		sign_j = np.sign(ang_gt[j,0].item())
		sign_prev_j = np.sign(ang_gt[j-1,0].item())
		difference = torch.abs(ang_gt[j,0]-ang_gt[j-1,0])
		if(difference<math.pi):
			
			if (ang_gt[j,0]>ang_gt[j-1,0]):
				ang_gt_copy[j,0] = ang_gt_copy[j-1,0]+difference
			else:
				ang_gt_copy[j,0] = ang_gt_copy[j-1,0]-difference

		else:
			
			if sign_prev_j==1:
				ang_gt_copy[j,0] = ang_gt_copy[j-1,0]  + torch.abs(math.pi - ang_gt[j-1,0]) + torch.abs(ang_gt[j,0] - (-math.pi))
			else:
				ang_gt_copy[j,0] = ang_gt_copy[j-1,0]  - (torch.abs(math.pi - ang_gt[j,0]) + torch.abs(ang_gt[j-1,0] -(-math.pi)))

		sign_j = np.sign(ang_gt[j,1].item())
		sign_prev_j = np.sign(ang_gt[j-1,1].item())
		difference = torch.abs(ang_gt[j,1]-ang_gt[j-1,1])
		if(difference<math.pi):
			if (ang_gt[j,1]>ang_gt[j-1,1]):
				ang_gt_copy[j,1] = ang_gt_copy[j-1,1]+difference
			else:
				ang_gt_copy[j,1] = ang_gt_copy[j-1,1]-difference

		else:
			if sign_prev_j==1:
				ang_gt_copy[j,1] = ang_gt_copy[j-1,1]  + torch.abs(math.pi - ang_gt[j-1,1]) + torch.abs(ang_gt[j,1] - (-math.pi))
			else:
				ang_gt_copy[j,1] = ang_gt_copy[j-1,1]  - (torch.abs(math.pi - ang_gt[j,1]) + torch.abs(ang_gt[j-1,1] -(-math.pi)))
	print(ang_gt_copy)
	return ang_gt_copy
'''
