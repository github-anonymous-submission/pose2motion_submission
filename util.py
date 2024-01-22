
"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import sys
sys.path.append("../")
sys.path.append("../dataset/")
from Kinemtics import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from rodrigues_utils import matrot2aa
import wandb as WWandb
from scipy.spatial.transform import Rotation as R 
import time 

from eval.bvh_writer import *

import argparse
def function_with_args_and_default_kwargs(optional_args=None, **kwargs):
    parser = argparse.ArgumentParser()
    # add some arguments
    # add the other arguments
    for k, v in kwargs.items():
        parser.add_argument('--' + k, default=v)
    agg = parser.parse_args(optional_args)
    return agg

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def _recompute_joint_global_info(_joint_rotation, _skeleton_joint_parents, _skeleton_joint_offsets, root_translation=None,root_rotation=None):
    # root_translation: (num_frames, 3)
    # root_rotation: (num_frames, 4)

    _num_frames = _joint_rotation.shape[0]
    _num_joints = _joint_rotation.shape[1]
    _joint_position = None
    _joint_orientation = None
    _joint_rotation = _joint_rotation
    _skeleton_joint_offsets = _skeleton_joint_offsets


    #########
    # now pre-compute joint global positions and orientations
    if _joint_position is None:
        _joint_position = np.zeros((_num_frames, _num_joints, 3))
    else:
        _joint_position.fill(0)
    
    if _joint_orientation is None:
        _joint_orientation = np.zeros((_num_frames, _num_joints, 4))
    else:
        _joint_orientation.fill(0)

    for i, pi in enumerate(_skeleton_joint_parents):
        if root_translation is not None and i==0:
            _joint_position[:,i,:] = root_translation[:,:]+_skeleton_joint_offsets[i,:]
        else:
            _joint_position[:,i,:] = _skeleton_joint_offsets[i,:]
            
        if root_rotation is not None and i==0:
            _joint_orientation[:,i,:] = root_rotation[:,:]
        else:
            _joint_orientation[:,i,:] = _joint_rotation[:,i,:]

        if pi < 0:
            assert (i == 0)
            continue
       
        parent_orient = R(_joint_orientation[:,pi,:], copy=False)
        _joint_position[:,i,:] = parent_orient.apply(_joint_position[:,i,:]) + _joint_position[:,pi,:]
        _joint_orientation[:,i,:] = (parent_orient * R(_joint_orientation[:,i,:], copy=False)).as_quat()
        _joint_orientation[:,i,:] /= np.linalg.norm(_joint_orientation[:,i,:], axis=-1, keepdims=True)
    return _joint_position, _joint_orientation

def plot_skeletons(info,_joint_position,type="men"):
    joint_names = info['joint_names']
    end_sites = info['end_sites']
    _skeleton_joint_parents = info['joint_parents_idx_full']
    _t_pose = info['T_pose']
    x_range = info['x_range']
    y_range = info['y_range']
    z_range = info['z_range']

    parent_idx = np.array(_skeleton_joint_parents)
    parent_idx[0] = 0

    joint_colors=['r' if 'Left' in x else 'b' if 'Right' in x else 'y' for x in joint_names]
    for i in range(len(joint_colors)):
        if i in end_sites:
            joint_colors[i] = 'k'

    fig = plt.figure(figsize=(20,10)) 
    columns = 6
        
    for i in range(columns):
        __joint_position = _joint_position[i]
        ax = fig.add_subplot(3,columns,columns*0+i+1, projection='3d')
        strokes = [plt.plot(xs=__joint_position[(i,p),0], zs=__joint_position[(i,p),1], ys=-__joint_position[(i,p),2], c=joint_colors[i], marker='x', linestyle='solid') for (i,p) in enumerate(parent_idx)]
        strokes_t = [plt.plot(xs=_t_pose[(i,p),0], zs=_t_pose[(i,p),1], ys=-_t_pose[(i,p),2], c=joint_colors[i], linestyle='--',alpha=0.25) for (i,p) in enumerate(parent_idx)]
        marker = ["v","^","<",">","p","D"]
        color_list = ['y', 'c', 'm', 'w', 'tab:orange', 'tab:pink']
        for j, end_site in enumerate(end_sites[1:5]):
            ax.plot(xs=__joint_position[end_site,0], zs=__joint_position[end_site,1], ys=-1*__joint_position[end_site,2], c=color_list[j], marker=marker[j], linestyle='solid')
            ax.plot(xs=_t_pose[end_site,0], zs=_t_pose[end_site,1], ys=-1*_t_pose[end_site,2], c=color_list[j], marker=marker[j], linestyle='solid',alpha=0.25)

        ax.auto_scale_xyz(x_range, x_range, x_range)
        ax.set_title('v1')
        ax.view_init(elev=10., azim=0, roll=0)

        ax = fig.add_subplot(3,columns,columns*1+i+1, projection='3d')
        strokes = [plt.plot(xs=__joint_position[(i,p),0], zs=__joint_position[(i,p),1], ys=-__joint_position[(i,p),2], c=joint_colors[i], marker='x', linestyle='solid') for (i,p) in enumerate(parent_idx)]
        strokes_t = [plt.plot(xs=_t_pose[(i,p),0], zs=_t_pose[(i,p),1], ys=-_t_pose[(i,p),2], c=joint_colors[i], linestyle='--',alpha=0.25) for (i,p) in enumerate(parent_idx)]
        marker = ["v","^","<",">","p","D"]
        color_list = ['y', 'c', 'm', 'w', 'tab:orange', 'tab:pink']
        for j, end_site in enumerate(end_sites[1:5]):
            ax.plot(xs=__joint_position[end_site,0], zs=__joint_position[end_site,1], ys=-1*__joint_position[end_site,2], c=color_list[j], marker=marker[j], linestyle='solid')
            ax.plot(xs=_t_pose[end_site,0], zs=_t_pose[end_site,1], ys=-1*_t_pose[end_site,2], c=color_list[j], marker=marker[j], linestyle='solid',alpha=0.25)

        ax.auto_scale_xyz(x_range,x_range, x_range)
        ax.set_title('v2')
        ax.view_init(elev=10., azim=40, roll=0)
        
        ax = fig.add_subplot(3,columns,columns*2+i+1, projection='3d')
        strokes = [plt.plot(xs=__joint_position[(i,p),0], zs=__joint_position[(i,p),1], ys=-__joint_position[(i,p),2], c=joint_colors[i], marker='x', linestyle='solid') for (i,p) in enumerate(parent_idx)]
        strokes_t = [plt.plot(xs=_t_pose[(i,p),0], zs=_t_pose[(i,p),1], ys=-_t_pose[(i,p),2], c=joint_colors[i], linestyle='--',alpha=0.25) for (i,p) in enumerate(parent_idx)]
        marker = ["v","^","<",">","p","D"]
        color_list = ['y', 'c', 'm', 'w', 'tab:orange', 'tab:pink']
        for j, end_site in enumerate(end_sites[1:5]):
            ax.plot(xs=__joint_position[end_site,0], zs=__joint_position[end_site,1], ys=-1*__joint_position[end_site,2], c=color_list[j], marker=marker[j], linestyle='solid')
            ax.plot(xs=_t_pose[end_site,0], zs=_t_pose[end_site,1], ys=-1*_t_pose[end_site,2], c=color_list[j], marker=marker[j], linestyle='solid',alpha=0.25)

        ax.auto_scale_xyz(x_range,x_range, x_range)
        ax.set_title('v3')
        ax.view_init(elev=10., azim=80, roll=0)

    return fig

def animte_skeleton_both(info_list,_joint_position_list,type="men",path=None):
    
    joint_names_0 = info_list[0]['joint_names']
    end_sites_0 = info_list[0]['end_sites']
    _skeleton_joint_parents_0 = info_list[0]['joint_parents_idx_full']
    _t_pose_0 = info_list[0]['T_pose']
    x_range_0 = [i/1.5 for i in info_list[0]['x_range']]

    joint_names_1 = info_list[1]['joint_names']
    end_sites_1 = info_list[1]['end_sites']
    _skeleton_joint_parents_1 = info_list[1]['joint_parents_idx_full']
    _t_pose_1 = info_list[1]['T_pose']
    x_range_1 = [i/1.5 for i in info_list[1]['x_range']]


    parent_idx_0 = np.array(_skeleton_joint_parents_0)
    parent_idx_0[0] = 0
    parent_idx_1 = np.array(_skeleton_joint_parents_1)
    parent_idx_1[0] = 0

    joint_colors_0=['r' if 'Left' in x else 'b' if 'Right' in x else 'y' for x in joint_names_0]
    for i in range(len(joint_colors_0)):
        if i in end_sites_0:
            joint_colors_0[i] = 'k'

    joint_colors_1=['r' if 'Left' in x else 'b' if 'Right' in x else 'y' for x in joint_names_1]
    for i in range(len(joint_colors_1)):
        if i in end_sites_1:
            joint_colors_1[i] = 'k'

    joint_colors = [joint_colors_0,joint_colors_1]
    parent_idx = [parent_idx_0,parent_idx_1]
    x_range = [x_range_0,x_range_1]
        
    fig = plt.figure(figsize=(15,10))
    columns = 2
    strokes = []
    
    for j in range(columns): 
        ax = fig.add_subplot(2,3,1+3*j, projection='3d')
        pos = _joint_position_list[j][0]
        strokes_0 = [plt.plot(xs=pos[(i,p),0], zs=pos[(i,p),1], ys=-pos[(i,p),2], c=joint_colors[j][i], marker='x', linestyle='solid') for (i,p) in enumerate(parent_idx[j])]
        ax.auto_scale_xyz(x_range[j], x_range[j], x_range[j])
        ax.set_title('clip0')
        ax.view_init(elev=10., azim=20, roll=0)
        
        ax = fig.add_subplot(2,3,2+3*j, projection='3d')
        pos = _joint_position_list[j][0]
        strokes_1 = [plt.plot(xs=pos[(i,p),0], zs=pos[(i,p),1], ys=-pos[(i,p),2], c=joint_colors[j][i], marker='x', linestyle='solid') for (i,p) in enumerate(parent_idx[j])]
        ax.auto_scale_xyz(x_range[j], x_range[j], x_range[j])
        ax.set_title('clip1')
        ax.view_init(elev=10., azim=40, roll=0)

        ax = fig.add_subplot(2,3,3+3*j, projection='3d')
        pos = _joint_position_list[j][0]
        strokes_2 = [plt.plot(xs=pos[(i,p),0], zs=pos[(i,p),1], ys=-pos[(i,p),2], c=joint_colors[j][i], marker='x', linestyle='solid') for (i,p) in enumerate(parent_idx[j])]
        ax.auto_scale_xyz(x_range[j], x_range[j], x_range[j])
        ax.set_title('clip1')
        ax.view_init(elev=10., azim=60, roll=0)
        strokes.append([strokes_0,strokes_1,strokes_2])

    def update_lines(num):
        for j in range(2):
            for (i,p) in enumerate(parent_idx[j]):
                strokes[j][0][i][0].set_data(_joint_position_list[j][num][(i,p),0], -_joint_position_list[j][num][(i,p),2])
                strokes[j][0][i][0].set_3d_properties(_joint_position_list[j][num][(i,p),1])
                
            for (i,p) in enumerate(parent_idx[j]):
                strokes[j][1][i][0].set_data(_joint_position_list[j][num][(i,p),0], -_joint_position_list[j][num][(i,p),2])
                strokes[j][1][i][0].set_3d_properties(_joint_position_list[j][num][(i,p),1])

            for (i,p) in enumerate(parent_idx[j]):
                strokes[j][2][i][0].set_data(_joint_position_list[j][num][(i,p),0], -_joint_position_list[j][num][(i,p),2])
                strokes[j][2][i][0].set_3d_properties(_joint_position_list[j][num][(i,p),1])

    print(min(_joint_position_list[j].shape[0],50))
    line_ani = animation.FuncAnimation(
        fig, update_lines, min(_joint_position_list[0].shape[0],50),
        interval=1, blit=False)
    line_ani.save(path, writer='imagemagick', fps=1)





def animte_skeleton(info,_joint_position,type="men",path=None):
    
    joint_names = info['joint_names']
    end_sites = info['end_sites']
    _skeleton_joint_parents = info['joint_parents_idx_full']
    _t_pose = info['T_pose']
    x_range = [i/1.5 for i in info['x_range']]
    y_range = info['y_range']
    z_range = info['z_range']

    parent_idx = np.array(_skeleton_joint_parents)
    parent_idx[0] = 0

    joint_colors=['r' if 'Left' in x else 'b' if 'Right' in x else 'y' for x in joint_names]
    for i in range(len(joint_colors)):
        if i in end_sites:
            joint_colors[i] = 'k'

    fig = plt.figure(figsize=(15,5)) 
   
    ax = fig.add_subplot(131, projection='3d')
    pos = _joint_position[0]
    strokes_0 = [plt.plot(xs=pos[(i,p),0], zs=pos[(i,p),1], ys=-pos[(i,p),2], c=joint_colors[i], marker='x', linestyle='solid') for (i,p) in enumerate(parent_idx)]
    ax.auto_scale_xyz(x_range, x_range, x_range)
    ax.set_title('clip0')
    ax.view_init(elev=10., azim=20, roll=0)
    
    ax = fig.add_subplot(132, projection='3d')
    pos = _joint_position[0]
    strokes_1 = [plt.plot(xs=pos[(i,p),0], zs=pos[(i,p),1], ys=-pos[(i,p),2], c=joint_colors[i], marker='x', linestyle='solid') for (i,p) in enumerate(parent_idx)]
    ax.auto_scale_xyz(x_range, x_range, x_range)
    ax.set_title('clip1')
    ax.view_init(elev=10., azim=40, roll=0)

    ax = fig.add_subplot(133, projection='3d')
    pos = _joint_position[0]
    strokes_2 = [plt.plot(xs=pos[(i,p),0], zs=pos[(i,p),1], ys=-pos[(i,p),2], c=joint_colors[i], marker='x', linestyle='solid') for (i,p) in enumerate(parent_idx)]
    ax.auto_scale_xyz(x_range, x_range, x_range)
    ax.set_title('clip1')
    ax.view_init(elev=10., azim=60, roll=0)

    def update_lines(num):
        for (i,p) in enumerate(parent_idx):
            strokes_0[i][0].set_data(_joint_position[num][(i,p),0], -_joint_position[num][(i,p),2])
            strokes_0[i][0].set_3d_properties(_joint_position[num][(i,p),1])
            
        for (i,p) in enumerate(parent_idx):
            strokes_1[i][0].set_data(_joint_position[num][(i,p),0], -_joint_position[num][(i,p),2])
            strokes_1[i][0].set_3d_properties(_joint_position[num][(i,p),1])

        for (i,p) in enumerate(parent_idx):
            strokes_2[i][0].set_data(_joint_position[num][(i,p),0], -_joint_position[num][(i,p),2])
            strokes_2[i][0].set_3d_properties(_joint_position[num][(i,p),1])

    print(min(_joint_position.shape[0],100))
    line_ani = animation.FuncAnimation(
        fig, update_lines, min(_joint_position.shape[0],100),
        interval=1, blit=False)
    line_ani.save(path, writer='imagemagick', fps=1)



def animate(type="Dog",rotation=None,wandb=None,info=None,gpu_ids=8,prior_shape=None,dino_pred=None, model=None,prefix=None,rec="",save_pth="."):
    rotation_copy = rotation.clone()
    
    rotation = R.from_matrix((rotation).reshape(-1,3,3).cpu().numpy()).as_quat().reshape(rotation.shape[0],-1,4)
    joint_offsets = info['joint_offsets']
    joint_parents_idx = info['joint_parents_idx_full']
    _joint_position, _ = _recompute_joint_global_info(rotation, joint_parents_idx, joint_offsets)
    path = os.path.join(save_pth,"{}.gif".format(prefix))
    print(path)
    animte_skeleton(info,_joint_position,path=path)



def animate_both(rotation_list=None,wandb=None,info=None,gpu_ids=8,prior_shape=None,dino_pred=None, model=None,prefix=None,rec="",save_pth="."):
    
    _joint_position_list = []
    for i,rotation in enumerate(rotation_list):
        rotation = R.from_matrix((rotation).reshape(-1,3,3).cpu().numpy()).as_quat().reshape(rotation.shape[0],-1,4)
        joint_offsets = info[i]['joint_offsets']
        joint_parents_idx = info[i]['joint_parents_idx_full']
        _joint_position, _ = _recompute_joint_global_info(rotation, joint_parents_idx, joint_offsets)
        _joint_position_list.append(_joint_position)
    path = os.path.join(save_pth,"{}.gif".format(prefix))
    print(path)
    animte_skeleton_both(info,_joint_position_list,path=path)


def save_bvh(rotation,info=None,filename=None,translation=None,normalize=True,names=None):
    offset = info['joint_offsets']
    parent = info['joint_parents_idx_full']
    names = names
    t_poseA = info['T_pose']
    if translation is not None:
        position = translation[:,:]
    else:
        position = np.zeros([rotation.shape[0], 3])
    # import pdb;pdb.set_trace()
    if normalize:
        scale = torch.max(t_poseA,dim=0)[0]-torch.min(t_poseA,dim=0)[0].numpy()
        offset = offset/(scale.mean())
        position = position/(scale.mean())
        
    writer = WriterWrapper(parent, offset)
    writer.write(filename,rotation,position,offset,names,"matrix")


def visualize(type="Dog",rotation=None,wandb=None,info=None,gpu_ids=8,prior_shape=None,dino_pred=None, model=None,prefix=None,rec="",writer=None,steps=0,with_root=False,noplot=False):
    rotation_copy = rotation.clone()
    if with_root:
        root_rotation = rotation[:,0,:].clone()
        # root_rotation = (torch.eye(3).cuda().reshape(1,-1)).repeat(rotation.shape[0],1)
        rotation[:,0,:] = torch.eye(3).cuda().reshape(-1)
        root_rotation = R.from_matrix((root_rotation).reshape(-1,3,3).cpu().numpy()).as_quat().reshape(rotation.shape[0],4)

            
    rotation = R.from_matrix((rotation).reshape(-1,3,3).cpu().numpy()).as_quat().reshape(rotation.shape[0],-1,4)
    joint_offsets = info['joint_offsets']
    joint_parents_idx = info['joint_parents_idx_full']
    if not with_root:
        _joint_position, _ = _recompute_joint_global_info(rotation, joint_parents_idx, joint_offsets)
    else:
        _joint_position, _ = _recompute_joint_global_info(rotation, joint_parents_idx, joint_offsets,root_rotation=root_rotation)
    if noplot:
        return _joint_position
    fig = plot_skeletons(info,_joint_position)
    plt.show()
    name = "{}_{}.png".format(type,time.time())
    plt.savefig(name)
    import pdb;pdb.set_trace()

    if wandb is not None:
        wandb.log({"{}/{}{}".format(prefix,type,rec): WWandb.Image(name)})
    if writer is not None:
        fig.canvas.draw()
        # Convert the figure to numpy array, read the pixel values and reshape the array
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = img / 255.0
        writer.add_image("{}/{}{}".format(prefix,type,rec), img.transpose(2,0,1), steps)
    try:
        os.remove(name)
    except:
        pass
    return _joint_position


def plt_3d(ax, joint_locations, topology,root_id=0):
    joints = joint_locations
    for idx in range(joints.shape[0]):
            parent_x = joints[idx,0]
            parent_y = joints[idx,2]
            parent_z = joints[idx,1]
            ax.scatter(xs=parent_x, 
                    ys=parent_y,  
                    zs=parent_z,  
                    alpha=0.6, c='b', marker='o')
            if idx==root_id:
                parent_x = joints[idx,0]
                parent_y = joints[idx,2]
                parent_z = joints[idx,1]
                ax.scatter(xs=parent_x, 
                        ys=parent_y,  
                        zs=parent_z,  
                        alpha=0.6, c='r', marker='o')
    for edge in topology:
        child, parent_chain = edge
        parent_x = joints[child,0]
        parent_y = joints[child,2]
        parent_z = joints[child,1]
        
        child_x = joints[parent_chain[0],0]
        child_y = joints[parent_chain[0],2]
        child_z = joints[parent_chain[0],1]
        # ^ In mocaps, Y is the up-right axis
        ax.plot([parent_x, child_x], [parent_y, child_y], [parent_z, child_z], 'k-', lw=2, c='black')
    ax.view_init(elev=20., azim=0, roll=0)

def mkdirs(paths):
    """create empty directories if they don't exist
    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


