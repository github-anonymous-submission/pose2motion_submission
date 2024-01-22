import torch
import numpy as np

from os import path as osp
import random
import glob as glob
import h5py
from scipy.spatial.transform import Rotation as R 
import pytorch3d.transforms
from data_util import *

class Unaligned_dataset_biped_new(torch.utils.data.Dataset):
    def __init__(self, 
                num_joints_A=27,
                num_joints_B=27,
                use_reroot=False,
                remove_virtual_node=False,
                type="matrix",
                window_size=-1,
                with_root=False,
                root_translation=False,
                velocity_virtual_node=False,
                mean_A=None,
                mean_B=None,
                normalize_all=False,
                var_A=None,
                var_B=None,
                use_global_y=False,
                use_gt_stat=False,
                split=0.8,
                mode="train", # 'train', 'val', 'test' - as in VPoser training 
                partial_A = 1,
                index=0, # index of the dataset to use
                identity_test=0,
                 ):

        # get all datasets if input datasets is empty
        self.dataset_subsample = 1.
        self.animal = "Biped"
        if mode=="train":
            file_num_A = len(glob.glob("./data/Mixamo_ordered/Mousey_m/{}withroot.h5".format("*")))
            file_num_B = len(glob.glob("./data/Mixamo_ordered/Mousey_m/Mousey_m{}withroot.h5".format("*")))
            assert file_num_A == file_num_B
            print(file_num_A, file_num_B)
            train_num = int(file_num_A*split)
            if partial_A<1:
                train_num_A = int(file_num_A*split*partial_A)
            else:
                train_num_A = train_num
            self.datasets_A = sorted(glob.glob("./data/Mixamo_ordered/Mousey_m/{}withroot.h5".format("*")))[:train_num_A]
            self.datasets_B = sorted(glob.glob("./data/Mixamo_ordered/Aj/Aj{}withroot.h5".format("*")))[:train_num]
            print("train_num_A", train_num_A, "train_num_B", train_num)
            if identity_test==1:
                self.datasets_B  = self.datasets_A
        elif mode=="val":
            file_num_A = len(glob.glob("./data/Mixamo_ordered/Mousey_m/{}withroot.h5".format("*")))
            file_num_B = len(glob.glob("./data/Mixamo_ordered/Aj/Aj{}withroot.h5".format("*")))
            assert file_num_A == file_num_B
            train_num = int(file_num_A*split)
            self.datasets_A = sorted(glob.glob("./data/Mixamo_ordered/Mousey_m/{}withroot.h5".format("*")))[:train_num]
            self.datasets_B = sorted(glob.glob("./data/Mixamo_ordered/Aj/Aj{}withroot.h5".format("*")))[:train_num]
            if identity_test==1:
                self.datasets_B  = self.datasets_A
        elif mode=="test":
            file_num_A = len(glob.glob("./data/Mixamo_ordered/Mousey_m/{}withroot.h5".format("*")))
            file_num_B = len(glob.glob("./data/Mixamo_ordered/Aj/Aj{}withroot.h5".format("*")))
            assert file_num_A == file_num_B
            train_num = int(file_num_A*split)
            self.datasets_A = [sorted(glob.glob("./data/Mixamo_ordered/Mousey_m/*withroot.h5"))[train_num:][index]]
            self.datasets_B = [sorted(glob.glob("./data/Mixamo_ordered/Aj/Aj*withroot.h5"))[train_num:][index]]
            if identity_test==1:
                self.datasets_B  = self.datasets_A
        self.mode = mode
        self.use_gt_stat = use_gt_stat
        self.use_global_y = use_global_y
        self.normalize_all = normalize_all
        self.mean_A_ = mean_A
        self.mean_B_ = mean_B
        self.var_A_ = var_A
        self.var_B_ = var_B
        self.use_reroot = use_reroot
        self.with_root = with_root
        self.path_to_dataset_A = self.datasets_A # path to the folder that contains AMASS subdatasets
        self.path_to_dataset_B = self.datasets_B # path to the folder that contains AMASS subdatasets
        self.threshold = 0.003
        self.foot_index=[0,1]
  
        self.velocity_virtual_node = velocity_virtual_node
        self.pose_A, self.infoA = self.load_h5(self.path_to_dataset_A,type='A')
        self.pose_B, self.infoB = self.load_h5(self.path_to_dataset_B,type='B')
        self.num_joints_A = self.pose_A.shape[1]
        self.num_joints_B = self.pose_B.shape[1]

        self.pose_A_quat = self.pose_A.copy()
        self.pose_B_quat = self.pose_B.copy()
        self.root_translation = root_translation
        
            
        self.joint_parents_idx_A = self.infoA['joint_parents_idx']
        self.joint_parents_idx_B = self.infoB['joint_parents_idx']
        self.joint_parents_idx_A_full = self.infoA['joint_parents_idx_full']
        self.joint_parents_idx_B_full = self.infoB['joint_parents_idx_full']
        self.pose_virtual_index_A = self.infoA['pose_virtual_index']
        self.pose_virtual_index_B = self.infoB['pose_virtual_index']
        self.height_A = self.infoA['height']
        self.height_B = self.infoB['height']
        
        self.A_size = self.pose_A.shape[0]
        self.B_size = self.pose_B.shape[0]
        
        if type=="matrix":
            #TODO: add velocity virtual node
            self.pose_A = self._to_matrix(self.pose_A)
            self.pose_B = self._to_matrix(self.pose_B)
        elif type=="euler":
            #TODO: add velocity virtual node
            self.pose_A = self._to_euler(self.pose_A)
            self.pose_B = self._to_euler(self.pose_B)
            self.normalize()
        elif type=="rotation_6d":
            if self.velocity_virtual_node:
                self.pose_A_rotation = self._to_rotation_6d(self._to_matrix(self.pose_A[:,:-1]))
                self.pose_B_rotation = self._to_rotation_6d(self._to_matrix(self.pose_B[:,:-1]))
                self.pose_A = np.concatenate([self.pose_A[:,:],np.zeros([self.pose_A.shape[0],self.pose_A.shape[1],2])],axis=-1)
                self.pose_B = np.concatenate([self.pose_B[:,:],np.zeros([self.pose_B.shape[0],self.pose_B.shape[1],2])],axis=-1)
                self.pose_A = torch.from_numpy(self.pose_A).float()
                self.pose_B = torch.from_numpy(self.pose_B).float()
                self.pose_A = torch.cat([self.pose_A_rotation,self.pose_A[:,[-1]]],dim=1)
                self.pose_B = torch.cat([self.pose_B_rotation,self.pose_B[:,[-1]]],dim=1)
            else:
                self.pose_A = self._to_rotation_6d(self._to_matrix(self.pose_A))
                self.pose_B = self._to_rotation_6d(self._to_matrix(self.pose_B))

        
        self.infoA['pose_virtual_val'] = torch.zeros(self.pose_A.shape[1],self.pose_A.shape[2])
        self.infoA['pose_virtual_val'][self.infoA['pose_virtual_index']] = torch.tensor(self.pose_A[0,self.infoA['pose_virtual_index']]).float()
        self.infoB['pose_virtual_val'] = torch.zeros(self.pose_B.shape[1],self.pose_B.shape[2])
        self.infoB['pose_virtual_val'][self.infoB['pose_virtual_index']] = torch.tensor(self.pose_B[0,self.infoB['pose_virtual_index']]).float()
        self.infoA['virtual_mask'] = torch.ones(self.pose_A.shape[1],self.pose_A.shape[2])
        self.infoA['virtual_mask'][self.infoA['pose_virtual_index']] = 0
        self.infoB['virtual_mask'] = torch.ones(self.pose_B.shape[1],self.pose_B.shape[2])
        self.infoB['virtual_mask'][self.infoB['pose_virtual_index']] = 0
        
        self._to_tensor()

        if remove_virtual_node:
            # print(self.with_root,"remove virtual node",self.infoA['pose_virtual_index'][0],self.infoB['pose_virtual_index'][0])
            remain_index_A = get_remain_index(self.num_joints_A,self.infoA['pose_virtual_index'][0])
            remain_index_B = get_remain_index(self.num_joints_B,self.infoB['pose_virtual_index'][0])
            self.pose_A = self.pose_A[:,remain_index_A]
            self.pose_B = self.pose_B[:,remain_index_B]
            self.joint_parents_idx_A_new = update_parent_list(self.infoA['joint_parents_idx_full'],self.infoA['pose_virtual_index'][0])
            self.joint_parents_idx_B_new  = update_parent_list(self.infoB['joint_parents_idx_full'],self.infoB['pose_virtual_index'][0])
            self.infoA['joint_parents_idx'] = self.joint_parents_idx_A_new
            self.infoA['joint_parents_idx_full'] = self.joint_parents_idx_A
            self.infoB['joint_parents_idx'] = self.joint_parents_idx_A_new
            self.infoB['joint_parents_idx_full'] = self.joint_parents_idx_B
            self.infoA['remain_index'] = np.array(remain_index_A)
            self.infoB['remain_index'] = np.array(remain_index_B)
            self.num_joints_A = len(remain_index_A)
            self.num_joints_B = len(remain_index_B)
            self.joint_parents_idx_A = self.joint_parents_idx_A_new
            self.joint_parents_idx_B = self.joint_parents_idx_B_new

        if self.normalize_all and mode!="test":
            self.get_normalize()
            if not self.use_gt_stat and self.velocity_virtual_node:
                self.var_A[0,-1,:3] = self.var_B[0,-1,:3]/self.infoB['height_list'][0]*self.infoA['height_list'][0]
                self.mean_A[0,-1,:3] = self.mean_B[0,-1,:3]/self.infoB['height_list'][0]*self.infoA['height_list'][0]
            self.normalize()
        elif self.normalize_all and mode=="test":
            self.mean_A = self.mean_A_
            self.mean_B = self.mean_B_
            self.var_A = self.var_A_
            self.var_B = self.var_B_
            self.normalize()
        else:
            self.mean_A,self.var_A,self.mean_B,self.var_B = None,None,None,None


        self.window_size = window_size
        if self.window_size > 0:
            if self.dataset_subsample==1:
                self.pose_A = self._window(self.pose_A,self.window_size)
            self.pose_B = self._window(self.pose_B,self.window_size,type="B")
            self.A_size = len(self.pose_A)
            self.B_size = len(self.pose_B)
            
    
    def _window(self,pose,window_size,type="A"):
        window_list = []
        if type=="A":
            length = self.A_length
        elif type=="B":
            length = self.B_length

        length = np.array(length).cumsum()
        assert length[-1] == pose.shape[0]
        for i in range(length.shape[0]-1):
            data = pose[length[i]:length[i+1]]
            for j in range(data.shape[0]-window_size+1):
                window_list.append(data[j:j+window_size].permute(1,2,0))
        return window_list

    def _to_tensor(self):
        if type(self.pose_A) == np.ndarray:
            self.pose_A = torch.from_numpy(self.pose_A).float()
            self.pose_B = torch.from_numpy(self.pose_B).float()

    def _to_matrix(self,pose):
        b = pose.shape[0]
        pose = R.from_quat(pose.reshape(-1,4)).as_matrix().reshape([b,-1,9])
        return pose
    
    def _to_euler(self,pose):
        b = pose.shape[0]
        pose = R.from_quat(pose.reshape(-1,4)).as_euler('ZXY', degrees=False).reshape([b,-1,3])
        return pose

    def _to_rotation_6d(self,pose):
        b = pose.shape[0]
        if type(pose) == np.ndarray:
            pose = torch.from_numpy(pose).float()
        pose = pytorch3d.transforms.matrix_to_rotation_6d(pose.reshape(b,-1,3,3))
        return pose 

    def normalize(self):
        self.pose_A = (self.pose_A - self.mean_A) / self.var_A
        self.pose_B = (self.pose_B - self.mean_B) / self.var_B

    def get_normalize(self,eps=1e-5):
        self.mean_A = torch.mean(self.pose_A, (0,), keepdim=True)
        self.mean_B = torch.mean(self.pose_B, (0,), keepdim=True)
        self.var_A = torch.var(self.pose_A, (0), keepdim=True)
        self.var_A = self.var_A ** (1/2)
        idx = self.var_A < eps
        self.var_A[idx] = 1
        self.var_B = torch.var(self.pose_B, (0), keepdim=True)
        self.var_B = self.var_B ** (1/2)
        idx = self.var_B < eps
        self.var_B[idx] = 1
        if self.with_root:
            self.mean_A[:,0,:] = self.mean_B[:,0,:]
            self.var_A[:,0,:] = self.var_B[:,0,:]

    def denormalize(self,pose,type="A"):
        if not self.normalize_all:
            return pose
        if type == "A":
            return pose * self.var_A.cuda() + self.mean_A.cuda()
        elif type == "B":
            return pose * self.var_B.cuda() + self.mean_B.cuda()

    def load_h5(self, paths_to_dataset, type="A"):
        poses = []
        info = []
        root_velocity = []
        if type=="B":
            self.B_length = [0]
        if type=="A":
            self.A_length = [0]
        for path in paths_to_dataset:
            with h5py.File(path, 'r') as f:
                poses.append(f['rotat'][:])
                joint_parents_idx = f['joint_parents_idx'][:]
                joint_offsets = f['joint_offsets'][:]
                end_sites = f['end_sites'][:]
                
                T_pose = torch.tensor(f['T_pose'][:])
                joint_names = f['joint_names'][:]
                foot_height = T_pose[end_sites[3],1]
                root_translation = f['joint_translation'][:][:,0,:] #T*3
                v = np.zeros_like(root_translation)
                v[:-1,:] = root_translation[1:,:]-root_translation[:-1,:]
                if self.use_global_y:
                    v[:,1] = root_translation[:,1]
                root_velocity.append(v)
                if type=="B":
                    self.B_length.append(poses[-1].shape[0])
                if type=="A":
                    self.A_length.append(poses[-1].shape[0])
                if self.mode=="test":
                    actual_root_translation = f['joint_translation'][:][:,0,:]
                    actual_root_rotation = self._to_rotation_6d(self._to_matrix(f['rotat'][:][:,0,:]))
                else:
                    actual_root_translation = 1
                    actual_root_rotation = 1

        joint_names = [name.decode('utf-8') for name in joint_names]
        root_velocity = np.concatenate(root_velocity, axis=0)
        poses = np.concatenate(poses, axis=0)
        if type=="A" and self.mode!="test":
            poses = poses[np.random.permutation(poses.shape[0])]

    
        if self.velocity_virtual_node:
            root_velocity_ = np.zeros([poses.shape[0],1,poses.shape[2]])
            if type=="B" or self.mode=="test" or self.use_gt_stat:
                root_velocity_[:,0,:3] = root_velocity
            poses = np.concatenate([poses,root_velocity_],axis=1)
            joint_parents_idx = np.concatenate([joint_parents_idx,[0]],axis=0)
            joint_offsets = np.concatenate([joint_offsets,np.zeros([1,3])],axis=0)
            T_pose = torch.cat([T_pose,torch.zeros([1,3])],dim=0)
            joint_names.append("root_velocity_node")
            
            
        pose_virtual_index = np.where(((np.abs(poses[0,:]-poses[20,:]))<1e-8).sum(axis=1)==4)
        if self.velocity_virtual_node and type=="A":
            pose_virtual_index = [pose_virtual_index[0][:-1]]

        height_list = self.get_height(end_sites, joint_parents_idx, joint_offsets)
        height = height_list[0] + height_list[2]
        info = {'joint_parents_idx': joint_parents_idx,
                'joint_parents_idx_full': joint_parents_idx,
                'joint_offsets': joint_offsets,
                'end_sites': end_sites,
                'T_pose':T_pose,
                "foot_height":foot_height,
                "root_translation":root_translation,
                "actual_root_translation":actual_root_translation,
                "actual_root_rotation":actual_root_rotation,
                'joint_names': joint_names,
                'pose_virtual_index':pose_virtual_index,
                'height_list':np.array(height_list),
                'height':height,
                "foot_index":[0,1],
                'z_range':[np.min(T_pose[:,1].numpy())*1, np.max(T_pose[:,1].numpy()*1)],
                'x_range':[-1*(np.max(T_pose[:,:].numpy()*1)-np.min(T_pose[:,:].numpy()*1))/2,(np.max(T_pose[:,:].numpy()*1)-np.min(T_pose[:,:].numpy()*1))/2],
                'y_range':[np.min(T_pose[:,2].numpy())*1, np.max(T_pose[:,2].numpy()*1)],}

        return poses,  info



    def get_height(self, end_list,parent_list,offset):
        height_list = []
        for i in end_list:
            p = parent_list[i]
            height = 0
            while p!=0:
                height += np.dot(offset[p],offset[p])**0.5
                p = parent_list[p]
            height_list.append(height)
        return height_list

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index (int)      -- a random integer for data indexing
        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        index_A = index % self.A_size  # make sure index is within then range
        index_B = random.randint(0, self.B_size - 1)
        if self.dataset_subsample==1:
            pose_A = self.pose_A[index_A]
        else:
            pose_A = self.pose_A[torch.randperm(self.pose_A.shape[0])[:self.window_size]]
        pose_B = self.pose_B[index_B]

        return {'A': pose_A, 'B': pose_B, 'A_paths': index_A, 'B_paths': index_B, 'infoA': self.infoA, 'infoB': self.infoB}
    
    def __len__(self):
        """Return the total number of images in the dataset.
        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)


