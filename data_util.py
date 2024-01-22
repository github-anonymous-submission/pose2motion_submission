import torch 
import pytorch3d.transforms
from scipy.spatial.transform import Rotation as R
import numpy as np

def remove_t_dim(pose):
    if len(pose.shape)==4:
        pose = pose.permute(0,3,1,2).reshape([-1,pose.shape[1],pose.shape[2]])

    return pose



def format_rotation(input, rotation):
        if input.shape[-1]==9:
            if rotation=="quaternion":
                input = pytorch3d.transforms.matrix_to_quaternion(input.reshape([-1,input.shape[1],3,3]))
            elif rotation=="rotation_6d":
                input = pytorch3d.transforms.matrix_to_rotation_6d(input.reshape([-1,input.shape[1],3,3]))
        elif input.shape[-1]==6:
            if rotation=="quaternion":
                input = pytorch3d.transforms.rotation_6d_to_matrix(input)
                input = pytorch3d.transforms.matrix_to_quaternion(input.reshape([-1,input.shape[1],3,3]))
            elif rotation=="matrix":
                input = pytorch3d.transforms.rotation_6d_to_matrix(input).reshape([-1,input.shape[1],9])
        elif input.shape[-1]==4:
            if rotation=="rotation_6d":
                input = pytorch3d.transforms.quaternion_to_matrix(input)
                input = pytorch3d.transforms.matrix_to_rotation_6d(input.reshape([-1,input.shape[1],3,3]))
            elif rotation=="matrix":
                input = pytorch3d.transforms.quaternion_to_matrix(input).reshape([-1,input.shape[1],9])
        return input

def update_parent_list(parent_list,virtual_nodes):
    '''
    :param parent_list: a list of parent index
    :param virtual_nodes: a list of virtual node index
    :return: a list of parent index after removing virtual nodes
    '''
    parent_list_new = []
    for i,p in enumerate(parent_list):
        new_p = p
        for node in list(virtual_nodes):
            if p>node:
                new_p=new_p-1
        if i not in virtual_nodes:
            parent_list_new.append(new_p)

    parent_list_new[0] = -1
    return parent_list_new

def get_remain_index(num_joints,virtual_nodes):
    '''
    write a summary of the function here
    :param num_joints: number of joints
    :param virtual_nodes: a list of virtual node index
    :return: a list of remain index after removing virtual nodes
    '''
    all_index = torch.arange(num_joints)
    remain_index = []
    for i in all_index:
        if i.item() not in virtual_nodes:
            remain_index.append(int(i.cpu().numpy()))
    return remain_index


def to_matrix(pose):
    '''
    write a summary of the function here
    :param pose: a tensor of shape (batch_size, num_joints, 4) or (batch_size, num_joints, 6)
    :return: a tensor of shape (batch_size, num_joints, 9)
    '''
    b,j,d = pose.shape
    if d==4:
        pose = pytorch3d.transforms.quaternion_to_matrix(pose).reshape(b,j,9)
    if d==6:
        pose = pytorch3d.transforms.rotation_6d_to_matrix(pose).reshape(b,j,9)
    return pose


def to_quat_scipy(pose):
    '''
    write a summary of the function here
    :param pose: a tensor of shape (batch_size, num_joints, 4) or (batch_size, num_joints, 9)
    :return: a tensor of shape (batch_size, num_joints, 4)
    '''
    b,j,d = pose.shape
    if d==4:
        return pose
    if d==6:
        pose = to_matrix(pose)
    pose = R.from_matrix(pose.reshape(b*j,3,3)).as_quat().reshape(b,j,4)
    return pose


def to_full_joint(pose,remain_index,virtual_index,virtual_val):
    '''
    write a summary of the function here
    :param pose: a tensor of shape (batch_size, num_joints, 4) or (batch_size, num_joints, 9)
    :param remain_index: a list of remain index after removing virtual nodes
    :param virtual_index: a list of virtual node index
    :param virtual_val: a tensor of shape (batch_size, num_virtual_nodes, 4) or (batch_size, num_virtual_nodes, 9)
    '''
    total_num_joints = len(list(virtual_index)) + len(list(remain_index))
    b,s,d = pose.shape
    full_pose = torch.zeros([b,total_num_joints,d]).to(pose.device)
    full_pose[:,remain_index,:] = pose
    full_pose += virtual_val.to(pose.device)
    return full_pose

def check_equal_quat(pose1,pose2):
    '''
    write a summary of the function here
    :param pose1: a tensor of shape (batch_size, num_joints, 4)
    :param pose2: a tensor of shape (batch_size, num_joints, 4)
    :return: a boolean
    '''
    # turn to numpy if not already
    if not isinstance(pose1,np.ndarray):
        pose1 = pose1.detach().cpu().numpy()
    if not isinstance(pose2,np.ndarray):
        pose2 = pose2.detach().cpu().numpy()
    pose1 = R.from_quat(pose1.reshape(-1,4)).as_euler('xyz',degrees=True)
    pose2 = R.from_quat(pose2.reshape(-1,4)).as_euler('xyz',degrees=True)
    return np.allclose(pose1,pose2,atol=1e-5)