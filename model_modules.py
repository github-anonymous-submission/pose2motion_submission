import torch
from torch import nn
from torch.nn import functional as F
from rodrigues_utils import batch_rodrigues, matrot2aa, ContinousRotReprDecoder
import pytorch3d 
from data_util import to_full_joint, to_matrix
from post_processing.models.contact import *

def _recompute_joint_global_info(_joint_rotation, _skeleton_joint_parents, _skeleton_joint_offsets,root_rotation=None,root_translation=None):
    _skeleton_joint_offsets = _skeleton_joint_offsets.cuda()
    _num_frames = _joint_rotation.shape[0]
    _num_joints = _joint_rotation.shape[1]
    _joint_rotation = _joint_rotation.view(_num_frames, _num_joints, 3, 3)
    _joint_position = torch.zeros((_num_frames, _num_joints, 3), dtype=torch.float32).cuda()
    _joint_orientation = torch.zeros((_num_frames, _num_joints, 3, 3), dtype=torch.float32).cuda()

    for i, pi in enumerate(_skeleton_joint_parents):
        if root_translation is not None and i==0:
            _joint_position[:,i,:] = root_translation[:,:]+_skeleton_joint_offsets[:,i,:]
        else:
            _joint_position[:,i,:] = _skeleton_joint_offsets[:,i,:]
            
        if root_rotation is not None and i==0:
            _joint_orientation[:,i,:] = root_rotation[:,:].reshape([-1,3,3])
        else:
            _joint_orientation[:,i,:] = _joint_rotation[:,i,:]

        if pi < 0:
            assert (i == 0)
            continue

        parent_orient = _joint_orientation[:, pi, :].clone()
        _joint_position[:, i, :] = torch.bmm(parent_orient,_joint_position[:, i, :][:,:,None].clone()).squeeze() + _joint_position[:, pi, :].clone()
        _joint_orientation[:, i, :] = parent_orient@_joint_orientation[:, i, :].clone()
    return _joint_position, _joint_orientation

class EndEffectorLoss(nn.Module):
    def __init__(self,remove_virtual_node,window_size=-1,loss_type="bbox",with_root=False,dataset=None,\
                                velocity_virtual_node=None,consistency_loss=False,use_global_y=False,\
                                foot_height_A=None,foot_height_B=None,use_mean_height=False,
                                ee_reweight=False):
        super(EndEffectorLoss, self).__init__()
        self.remove_virtual_node = remove_virtual_node
        self.window_size = window_size
        self.loss_type = loss_type
        self.with_root = with_root
        self.dataset = dataset
        self.velocity_virtual_node = velocity_virtual_node
        self.consistency_loss = consistency_loss
        self.use_global_y = use_global_y
        self.foot_height_A = foot_height_A
        self.foot_height_B = foot_height_B
        self.l1 = torch.nn.L1Loss()
        self.use_mean_height = use_mean_height
        self.ee_reweight = ee_reweight

    def __call__(self,poseA, poseB, infoA, infoB,temporal=False,root_translation=False,mean_A=None,var_A=None):
        
        poseA = poseA.clone()
        poseB = poseB.clone()

        if self.window_size>0:
            self.window_size = poseA.shape[3]
            poseA = poseA.permute(0,3,1,2).reshape([-1,poseA.shape[1],poseA.shape[2]])
            poseB = poseB.permute(0,3,1,2).reshape([-1,poseB.shape[1],poseB.shape[2]])
        if self.dataset.normalize_all:
            poseA = self.dataset.denormalize(poseA,type="A")
            poseB = self.dataset.denormalize(poseB,type="B")
        if self.velocity_virtual_node:
            root_v_A = poseA[:,-1,:3].clone()
            root_v_B = poseB[:,-1,:3].clone()
            poseA[:,-1,:] = torch.eye(3).cuda().reshape(-1)[:6]
            poseB[:,-1,:] = torch.eye(3).cuda().reshape(-1)[:6]
            if self.use_global_y:
                root_v_A[:,[0,2]] = root_v_A[:,[0,2]].cumsum(dim=0)
                root_v_B[:,[0,2]] = root_v_B[:,[0,2]].cumsum(dim=0)  
            else:
                root_v_A = root_v_A.cumsum(dim=0)
                root_v_B = root_v_B.cumsum(dim=0)
  
        if self.remove_virtual_node:
            poseA = to_full_joint(poseA,infoA['remain_index'][0],infoA['pose_virtual_index'][0][0],infoA['pose_virtual_val'][0])
            poseB = to_full_joint(poseB,infoB['remain_index'][0],infoB['pose_virtual_index'][0][0],infoB['pose_virtual_val'][0])
        if root_translation:
            poseA = to_matrix(poseA[:,:,:-3])
            poseB = to_matrix(poseB[:,:,:-3])
        else:
            poseA = to_matrix(poseA)
            poseB = to_matrix(poseB)
        if self.with_root:
            # set root rotation be identity
            poseA_root = poseA[:,0,:].clone()
            poseB_root = poseB[:,0,:].clone()
            poseA[:,0,:] = torch.eye(3).cuda().reshape(-1) #time x num_joints x 9
            poseB[:,0,:] = torch.eye(3).cuda().reshape(-1)
        
            
        bs = poseA.shape[0]
        joint_offsets = infoA['joint_offsets'].cuda()
        joint_parents_idx = infoA['joint_parents_idx_full'][0]
        end_sites = infoA['end_sites'][0]
        t_poseA = infoA['T_pose'][0].cuda()
        if self.loss_type=="bbox":
            scale_A = torch.max(t_poseA,dim=0)[0]-torch.min(t_poseA,dim=0)[0]
        elif self.loss_type.startswith("height"):
            scale_A = infoA['height_list'][0][:,None].cuda()

        if self.window_size>0:
            joint_offsets = joint_offsets.unsqueeze(1).repeat([1,self.window_size,1,1])
            joint_offsets = joint_offsets.reshape([-1,joint_offsets.shape[2],joint_offsets.shape[3]])

        _joint_position_A, _ = _recompute_joint_global_info(poseA.clone(), joint_parents_idx, joint_offsets)
        joint_end_site_A = _joint_position_A[:, end_sites, :].clone()
        t_pose_end_site_A = t_poseA[end_sites, :].clone()
        if self.with_root:
            joint_end_site_A_withroot = torch.bmm(poseA_root.repeat(len(end_sites),1).reshape(-1,3,3),joint_end_site_A.reshape(-1,3,1)).reshape(bs,-1,3).clone()

        joint_offsets = infoB['joint_offsets']
        if self.window_size>0:
            joint_offsets = joint_offsets.unsqueeze(1).repeat([1,self.window_size,1,1])
            joint_offsets = joint_offsets.reshape([-1,joint_offsets.shape[2],joint_offsets.shape[3]])

        joint_parents_idx = infoB['joint_parents_idx_full'][0].cuda()
        end_sites = infoB['end_sites'][0]
        t_poseB = infoB['T_pose'][0].cuda()
        if self.loss_type=="bbox":
            scale_B = torch.max(t_poseB,dim=0)[0]-torch.min(t_poseB,dim=0)[0]
        elif self.loss_type.startswith("height"):
            scale_B = infoB['height_list'][0][:,None].cuda()
        

        _joint_position_B, _ = _recompute_joint_global_info(poseB.clone(), joint_parents_idx, joint_offsets)
        joint_end_site_B = _joint_position_B[:, end_sites, :].clone()
        t_pose_end_site_B = t_poseB[end_sites, :].clone()
        if self.with_root:
            joint_end_site_B_withroot = torch.bmm(poseB_root.repeat(len(end_sites),1).reshape(-1,3,3),joint_end_site_B.reshape(-1,3,1)).reshape(bs,-1,3).clone()
        #loss is l1 norm of end site different each normalized by scale
        if self.loss_type == "height_reweight":
            scale_A[1:-1] = scale_A[1:-1].mean()
            scale_B[1:-1] = scale_B[1:-1].mean()
        if self.ee_reweight:
            scale_A = scale_A/self.dataset.ee_reweight
            scale_B = scale_B/self.dataset.ee_reweight
        
        
        loss_relative_to_T = (((joint_end_site_A-t_pose_end_site_A)/scale_A - (joint_end_site_B-t_pose_end_site_B)/scale_B)**2).mean()
        if temporal:
            if self.with_root:
                if self.velocity_virtual_node:

                    '''
                    xx= (joint_end_site_B_withroot+root_v_B[:,None,:]).reshape([-1,self.window_size,joint_end_site_B_withroot.shape[1],joint_end_site_B_withroot.shape[2]])
                    xx = joint_end_site_B_withroot
                    contact = foot_contact(xx[:,:,1:-1,:].clone(),threshold=0.4)
                    a1 = torch.where(contact[:,:,0]==1)
                    min1 = (xx[a1[0],a1[1],1,1].min())
                    a2 = torch.where(contact[:,:,1]==1)
                    min2 = (xx[a1[0],a1[1],2,1].min())
                    a3 = torch.where(contact[:,:,2]==1)
                    min3 = (xx[a1[0],a1[1],3,1].min())
                    a4 = torch.where(contact[:,:,3]==1)
                    min4 = (xx[a1[0],a1[1],4,1].min())
                    print(min1,min2,min3,min4)
        
                    '''
                    joint_end_site_A_withroot_local = joint_end_site_A_withroot.clone().reshape([-1,self.window_size,joint_end_site_A_withroot.shape[1],joint_end_site_A_withroot.shape[2]])
                    joint_end_site_A_withroot = (joint_end_site_A_withroot+root_v_A[:,None,:]).reshape([-1,self.window_size,joint_end_site_A_withroot.shape[1],joint_end_site_A_withroot.shape[2]])
                    joint_end_site_B_withroot_local = joint_end_site_B_withroot.clone().reshape([-1,self.window_size,joint_end_site_B_withroot.shape[1],joint_end_site_B_withroot.shape[2]])
                    joint_end_site_B_withroot = (joint_end_site_B_withroot+root_v_B[:,None,:]).reshape([-1,self.window_size,joint_end_site_B_withroot.shape[1],joint_end_site_B_withroot.shape[2]])
                    loss_relative_temporal = (((joint_end_site_A_withroot[:,:-1,:]-joint_end_site_A_withroot[:,1:,:])/scale_A - (joint_end_site_B_withroot[:,:-1,:]-joint_end_site_B_withroot[:,1:,:])/scale_B)**2).mean()
                    if self.dataset.threshold==0.003 or self.dataset.threshold==0.02:
                        joint_end_site_B_withroot_norm = (joint_end_site_B_withroot[:,:,self.dataset.foot_index,:].clone())/self.dataset.height_B
                        contact = foot_contact(joint_end_site_B_withroot_norm,threshold=self.dataset.threshold)
                        velo_A = velocity(joint_end_site_A_withroot[:,:,self.dataset.foot_index,:].clone(),padding=True)
                    else:
                        contact = foot_contact(joint_end_site_B_withroot[:,:,self.dataset.foot_index,:].clone(),threshold=self.dataset.threshold)
                        velo_A = velocity(joint_end_site_A_withroot[:,:,self.dataset.foot_index,:].clone(),padding=True)
                    if contact.sum()>0:
                        # in some version of dog training. no devided by contact
                        # loss_contact = (contact.clone().detach()*velo_A/(scale_A[self.dataset.foot_index,:].mean())).sum()/contact.sum()
                        if self.dataset.animal=="horse":
                            loss_contact = (contact.clone().detach()*velo_A).sum()/contact.sum()
                        else:
                            loss_contact = (contact.clone().detach()*velo_A/scale_A[None,None,self.dataset.foot_index,0]).sum()/contact.sum()
                    else:
                        loss_contact = 0
                    contact_index = torch.where(contact==1)
                    height_contact_B = joint_end_site_B_withroot_local[:,:,self.dataset.foot_index,:][contact_index[0],contact_index[1],contact_index[2],1].clone().detach()
                    weight = ((height_contact_B<self.foot_height_B)*1).detach() # weight = (loss_height_contact_B<5)*1
                    height_contact = joint_end_site_A_withroot_local[:,:,self.dataset.foot_index,:][contact_index[0],contact_index[1],contact_index[2],1].clone()
                    if self.use_mean_height:
                        self.foot_height_A = height_contact[weight==1].mean()
                    if weight.sum()>0:
                        if self.dataset.animal=="horse":
                            loss_height_contact = (weight*(torch.nn.functional.relu(height_contact-(torch.ones_like(height_contact,device=height_contact.device)*self.foot_height_A).detach()))**2).sum()/weight.sum()
                        else:
                            loss_height_contact = ((weight*(torch.nn.functional.relu(height_contact-(torch.ones_like(height_contact,device=height_contact.device)*self.foot_height_A).detach()))**2)/scale_A[None,None,self.dataset.foot_index,0].mean()).sum()/weight.sum()
                    else:
                        loss_height_contact = 0
                 
                else:
                    joint_end_site_A_withroot = joint_end_site_A_withroot.reshape([-1,self.window_size,joint_end_site_A_withroot.shape[1],joint_end_site_A_withroot.shape[2]])
                    joint_end_site_B_withroot = joint_end_site_B_withroot.reshape([-1,self.window_size,joint_end_site_B_withroot.shape[1],joint_end_site_B_withroot.shape[2]])
                    loss_relative_temporal = (((joint_end_site_A_withroot[:,:-1,:]-joint_end_site_A_withroot[:,1:,:])/scale_A - (joint_end_site_B_withroot[:,:-1,:]-joint_end_site_B_withroot[:,1:,:])/scale_B)**2).mean()
                    loss_contact = 0
                    loss_height_contact = 0
            else:
                loss_relative_temporal = 0
                loss_contact = 0
                loss_height_contact = 0
        else:
            loss_relative_temporal = 0
            loss_contact = 0
            loss_height_contact = 0

        return {"loss_relative_to_T":loss_relative_to_T,'loss_relative_temporal':loss_relative_temporal,\
                'loss_contact':loss_contact,'loss_relative_to_T_': (((joint_end_site_A-t_pose_end_site_A)/scale_A - (joint_end_site_B-t_pose_end_site_B)/scale_B)**2),\
                'loss_height_contact':loss_height_contact}


class PoseEachDiscriminator(nn.Module):
    def __init__(self, num_joints, hidden_poseeach=32):
        super(PoseEachDiscriminator, self).__init__()
        self.num_joints = num_joints
        
        self.fc_layer = nn.ModuleList()
        for idx in range(self.num_joints):
            self.fc_layer.append(nn.Linear(in_features=hidden_poseeach, out_features=1))

    def forward(self, comm_features):
        # getting individual pose outputs
        # common features is of shape [N x hidden_comm x num_joints]
        d_each = []
        for idx in range(self.num_joints):
            d_each.append(self.fc_layer[idx](comm_features[:,:,idx]))
        d_each_out = torch.cat(d_each, 1) # N x 23
        return d_each_out


class PoseAllDiscriminator(nn.Module):
    def __init__(self, num_joints, hidden_feature=32,hidden_poseall=1024, num_layers=2):
        super(PoseAllDiscriminator, self).__init__()
        self.num_joints = num_joints
        self.hidden_poseall = hidden_poseall
        self.num_layers = num_layers
        self.hidden_feature = hidden_feature

        fc_all_pose = [
            nn.Linear(in_features=self.hidden_feature*self.num_joints, out_features=hidden_poseall), 
            nn.LeakyReLU(0.2),
            ]
        for _ in range(num_layers - 1):
            fc_all_pose += [
                    nn.Linear(in_features=hidden_poseall, out_features=hidden_poseall), 
                    nn.LeakyReLU(0.2),
                    ]
        fc_all_pose += [nn.Linear(in_features=hidden_poseall, out_features=1)]
        self.fc_all_pose = nn.Sequential(*fc_all_pose)

    def forward(self, comm_features):
        # getting pose-all output
        # common features is of shape [N x hidden_comm x num_joints]
        d_all_pose = self.fc_all_pose(comm_features.contiguous().view(comm_features.size(0), -1))
        return d_all_pose


class geodesic_loss_R(nn.Module):
    def __init__(self, reduction='batchmean'):
        super(geodesic_loss_R, self).__init__()

        self.reduction = reduction
        self.eps = 1e-6

    # batch geodesic loss for rotation matrices
    def bgdR(self,m1,m2):
        batch = m1.shape[0]
        m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3

        cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
        cos = torch.min(cos, m1.new(torch.ones(batch)-self.eps))
        cos = torch.max(cos, (m1.new(torch.ones(batch)) * -1)+self.eps)

        return torch.acos(cos)

    def forward(self, ypred, ytrue):
        theta = self.bgdR(ypred,ytrue)
        if self.reduction == 'mean':
            return torch.mean(theta)
        if self.reduction == 'nomean':
            return theta
        if self.reduction == 'batchmean':
            breakpoint()
            return torch.mean(torch.sum(theta, dim=theta.shape[1:]))

        else:
            return theta
        
class Discriminator(nn.Module):
    def __init__(self,
                num_joints=21,
                hidden_poseeach=32, 
                hidden_poseall=1024,
                poseall_num_layers=2,
                hidden_comm=32,
                d_poseeach_weight=10,
                input_dim=9,
                velocity_virtual_node=False,
                with_root=False,):
        super(Discriminator, self).__init__() 
        self.num_joints = num_joints
        self.velocity_virtual_node = velocity_virtual_node
        if self.velocity_virtual_node:
            self.num_joints = self.num_joints-1
        self.with_root = with_root
        if self.with_root:
            self.num_joints = self.num_joints-1
        self.hidden_comm = hidden_comm
        self.comm_conv = nn.Sequential(
            nn.Conv2d(input_dim, self.hidden_comm, kernel_size=1, stride=1, padding=0, bias=True), 
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.hidden_comm, self.hidden_comm, kernel_size=1, stride=1, padding=0, bias=True), 
            nn.LeakyReLU(0.2),
            )
        
        self.disc_poseeach = PoseEachDiscriminator(num_joints=self.num_joints, hidden_poseeach=hidden_poseeach)
        self.disc_poseall = PoseAllDiscriminator(num_joints=self.num_joints, hidden_feature=self.hidden_comm, hidden_poseall=hidden_poseall, num_layers=poseall_num_layers)
        self.d_poseeach_weight = d_poseeach_weight
        
        self.velocity_virtual_node = velocity_virtual_node
        
    def forward(self, pose, input_type='matrot'):
        if len(pose.shape) == 4:
            pose = pose.permute(0,3,1,2).reshape([-1,pose.shape[1],pose.shape[2]])
        # input has a shape of SMPL pose, N x num_joints x 9
        if input_type == 'aa':
            pose = batch_rodrigues(pose.contiguous().view(-1,3))
            pose = pose.view(-1, self.num_joints, 9)
        inputs = pose.transpose(1, 2).unsqueeze(2) # to N x 9 x 1 x num_joints
        if self.with_root:
            # discriminator is applied on the cananical pose, so we set the root joint rotation to be 0
            inputs = inputs[:,:,:,1:]
        if self.velocity_virtual_node:
            inputs = inputs[:,:,:,:-1]
        comm_features = self.comm_conv(inputs).view(-1, self.hidden_comm, self.num_joints) # to N x hidden_comm x num_joints
        d_poseeach = self.disc_poseeach(comm_features) # B x num_joints
        d_poseall = self.disc_poseall(comm_features) # B x 1
        d_out = {
            'poseeach' : d_poseeach,
            'poseall' : d_poseall,
        }
        return d_poseeach*self.d_poseeach_weight+d_poseall

    def forward_test(self, pose, input_type='matrot'):
        if len(pose.shape) == 4:
            pose = pose.permute(0,3,1,2).reshape([-1,pose.shape[1],pose.shape[2]])
        # input has a shape of SMPL pose, N x num_joints x 9
        if input_type == 'aa':
            pose = batch_rodrigues(pose.contiguous().view(-1,3))
            pose = pose.view(-1, self.num_joints, 9)
        inputs = pose.transpose(1, 2).unsqueeze(2) # to N x 9 x 1 x num_joints
        comm_features = self.comm_conv(inputs).view(-1, self.hidden_comm, self.num_joints) # to N x hidden_comm x num_joints
        d_poseeach = self.disc_poseeach(comm_features) # B x num_joints
        d_poseall = self.disc_poseall(comm_features) # B x 1
        d_out = {
            'poseeach' : d_poseeach,
            'poseall' : d_poseall,
        }
        return d_out


class NormalDistDecoder(nn.Module):
    def __init__(self, num_feat_in, latentD):
        super(NormalDistDecoder, self).__init__()

        self.mu = nn.Linear(num_feat_in, latentD)
        self.logvar = nn.Linear(num_feat_in, latentD)

    def forward(self, Xout):
        return torch.distributions.normal.Normal(self.mu(Xout), F.softplus(self.logvar(Xout)))


class BatchFlatten(nn.Module):
    def __init__(self):
        super(BatchFlatten, self).__init__()
        self._name = 'batch_flatten'

    def forward(self, x):
        return x.view(x.shape[0], -1)
    
class ContinousRotReprDecoder(nn.Module):
    def __init__(self):
        super(ContinousRotReprDecoder, self).__init__()

    def forward(self, module_input):
        reshaped_input = module_input.view(-1, 3, 2)

        b1 = F.normalize(reshaped_input[:, :, 0], dim=1)

        dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
        b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=1)

        return torch.stack([b1, b2, b3], dim=-1)


class Generator(nn.Module):
    def __init__(self,
                 num_joints_in=27,
                 num_joints_out=27,
                 hidden_feature=256,
                 vae = False,
                 latent_dim=32,
                 fix_virtual_bones=False,
                 ):
        super(Generator, self).__init__()

        self.num_joints_in = num_joints_in
        self.num_joints_out = num_joints_out
        self.hidden_feature = hidden_feature
        self.vae = vae
        self.latent_dim = latent_dim

        self.encoder_net = nn.Sequential(
            BatchFlatten(),
            nn.BatchNorm1d(self.num_joints_in*9),
            nn.Linear(self.num_joints_in*9, self.hidden_feature),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.hidden_feature),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_feature, self.hidden_feature),
            nn.Linear(self.hidden_feature, self.hidden_feature),
            NormalDistDecoder(self.hidden_feature, self.latent_dim)
        )
        self.decoder_net = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_feature),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_feature, self.hidden_feature),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_feature, self.num_joints_out * 6),
            ContinousRotReprDecoder(),
        )

        self.mask = None
        self.val = None
        self.fix_virtual_bones = fix_virtual_bones
    
    def update(self,info):
        if self.fix_virtual_bones:
            self.mask = torch.tensor(info['virtual_mask']).cuda()
            self.val = torch.tensor(info['pose_virtual_val']).cuda()
            if len(self.mask.shape) == 2:
                self.mask = self.mask.unsqueeze(0)
                self.val = self.val.unsqueeze(0)


    def encode(self, pose, input_type="matrot"):
        '''
        :param Pin: Nx(numjoints*3)
        :param rep_type: 'matrot'/'aa' for matrix rotations or axis-angle
        :return:
        '''
        if input_type=="aa":
            pose_gt = pose.to(device="cuda", non_blocking=True)
            batch_size = pose_gt.size(0)
            matrot_real = batch_rodrigues(pose_gt.contiguous().view(-1,3))
            matrot_real = matrot_real.view(batch_size, -1, 9)
        else:
            matrot_real = pose
        # print(matrot_real.shape)
        return self.encoder_net(matrot_real)

    def decode(self, Zin):
        bs = Zin.shape[0]
        prec = self.decoder_net(Zin)
        prec = prec.view(bs, -1, 9)
            
        if self.fix_virtual_bones:
            self.mask = self.mask[[0]].repeat(bs,1,1)
            self.val = self.val[[0]].repeat(bs,1,1)
            prec = prec*self.mask + self.val

        return {
            "Zin": Zin,
            'pose': matrot2aa(prec.view(-1, 3, 3)).view(bs, -1, 3),
            'pose_matrot': prec.view(bs, -1, 9)
        }

    def forward(self, pose,input_type="matrot"):
        # print(pose.shape)
        q_z = self.encode(pose,input_type=input_type)
        if not self.vae:
            q_z_sample = q_z.mean
        else:
            raise NotImplementedError
    
        decode_results = self.decode(q_z_sample)
        return decode_results['pose_matrot']
    

class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'wgan-gp':
            self.loss = self.wgan_loss
            real_label = 1
            fake_label = 0
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)
       
    @staticmethod
    def wgan_loss(prediction, target):
        lmbda = torch.ones_like(target)
        lmbda[target == 1] = -1
        return (prediction * lmbda).mean()


    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgan-gp':
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        return loss
