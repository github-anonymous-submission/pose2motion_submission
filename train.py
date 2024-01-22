# from vis_mgp import get_plot_file
import torch
from train_options import TrainOptions, get_skeleton_aware_options
from dataloader import *
import wandb
import util
from tqdm import tqdm
import time
import os
import matplotlib.pyplot as plt
import wandb as WWandb

#tensorboard
from torch.utils.tensorboard import SummaryWriter
from data_util import remove_t_dim

opt = TrainOptions().parse()   # get training options
torch.cuda.set_device(opt.gpu_ids[0])
torch.set_num_threads(opt.num_threads)
# prior_shape,dino_pred, model_mgp = get_plot_file(opt.gpu_ids[0])
prior_shape,dino_pred, model_mgp= None,None,None
from cycle_gan import *
util.seed_everything(24)

if opt.dataset =="biped":
    dataset = Unaligned_dataset_biped_new(use_reroot=opt.skeleton_aware,remove_virtual_node=opt.remove_virtual_node==1,type=opt.rotation_data,window_size=opt.window_size,\
                                                    with_root=opt.with_root,velocity_virtual_node=opt.velocity_virtual_node,normalize_all=opt.normalize_all,partial_A=opt.partial_A,\
                                                        use_gt_stat=opt.use_gt_stat,use_global_y=opt.use_global_y,identity_test=opt.identity_test)
    dataset_test = Unaligned_dataset_biped_new(type=opt.rotation_data,mode="test",use_reroot=opt.skeleton_aware,remove_virtual_node=opt.remove_virtual_node==1,index=9,\
                                                    with_root=opt.with_root,velocity_virtual_node=opt.velocity_virtual_node,normalize_all=opt.normalize_all,\
                                                    mean_A=dataset.mean_A.clone(),mean_B=dataset.mean_B.clone(),var_A=dataset.var_A.clone(),var_B=dataset.var_B.clone(),use_global_y=opt.use_global_y,
                                                    identity_test=opt.identity_test)

    opt.num_joints_A = dataset.num_joints_A
    opt.num_joints_B = dataset.num_joints_B
    opt.foot_height_A = dataset.infoA['foot_height']
    opt.foot_height_B = dataset.infoB['foot_height']
    type_B = "men"
    type_A = "men"

    opt.foot_height_A = dataset.infoA['foot_height']
    opt.foot_height_B = dataset.infoB['foot_height']
    animal="men"


if opt.skeleton_aware:
    args = get_skeleton_aware_options()
    args.topology_a = dataset.joint_parents_idx_A
    args.topology_b = dataset.joint_parents_idx_B
    args.fix_virtual_bones = opt.fix_virtual_bones
    args.rotation_data = opt.rotation_data
    args.rotation = opt.rotation
    args.num_layers = opt.skeleton_num_layers
    args.kernel_size = opt.skeleton_kernel_size
    args.window_size = opt.window_size
    args.with_root = opt.with_root
    args.connect_end_site = opt.connect_end_site
    args.velocity_virtual_node = opt.velocity_virtual_node
    args.acGAN = opt.acGAN
    args.v_connect_all_joints = opt.v_connect_all_joints
    args.last_sigmoid = opt.last_sigmoid
    args.scale_factor = opt.scale_factor
    args.skeleton_dist = opt.skeleton_dist
    if args.connect_end_site:
        args.end_sites_a = dataset.infoA['end_sites']
        args.end_sites_b = dataset.infoB['end_sites']
    else:
        args.end_sites_a = None
        args.end_sites_b = None
    opt.skeleton_aware_args = args

    

#setup tensorboard
# writer = SummaryWriter(log_dir=os.path.join(opt.checkpoints_dir,opt.name))
writer = None
opt.lr_G_A = opt.lr
opt.lr_G_B = opt.lr
if opt.load_pretrained:
    opt.lr_G_A = 0
model  = CycleGANModel(opt,dataset).cuda()
if opt.load_pretrained:
    print("load pretrain model")
    model_dict = torch.load(os.path.join(opt.pretrained_path,"latest_netG.pth"))
    model.netG_A.encoder.load_state_dict(model_dict['encoder'],strict=True)
    model.netG_B.decoder.load_state_dict(model_dict['decoder'],strict=True)

dataloaders = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True if opt.overfit==0 else False, num_workers=1)

wandb = wandb.init(project=opt.wandb_project_name,name=animal+opt.checkpoints_dir, config=opt,settings=wandb.Settings(code_dir="."))
wandb.log_code(".")
wandb.watch(model)
total_iters = 0
torch.cuda.set_device(opt.gpu_ids[0])
opt.path = os.path.join(opt.checkpoints_dir,opt.name)
dict_var = vars(opt)
np.save(os.path.join(opt.checkpoints_dir,opt.name,"opt.npy"),dict_var)
dict_args = vars(args)
np.save(os.path.join(opt.checkpoints_dir,opt.name,"args.npy"),dict_var)
# copy code "model_modules.py" to checkpoints folder
import shutil
shutil.copyfile("model_modules.py", os.path.join(opt.checkpoints_dir,opt.name,"model_modules.py"))
# dataloader 
shutil.copyfile("dataloader.py", os.path.join(opt.checkpoints_dir,opt.name,"dataloader.py"))

# model.save_networks('latest')
for epoch in range(opt.n_epochs):
    # model.update_learning_rate()  
    epoch_start_time = time.time()  # timer for entire epoch
    for i, data in enumerate(tqdm(dataloaders)):
        if opt.load_pretrained:
            model.netG_A.encoder.eval()
            model.netG_B.decoder.eval()
      
        model.set_input(data)         # unpack data from dataset and apply preprocessing
        if opt.gen_freq+opt.disc_freq>0:
            if total_iters%(opt.gen_freq+opt.disc_freq) < opt.disc_freq:
                model.optimize_parameters(wandb,opt_gen=False,opt_disc=True,writer=writer)     # calculate loss functions, get gradients, update network weights
            elif total_iters%(opt.gen_freq+opt.disc_freq) >= opt.disc_freq:
                model.optimize_parameters(wandb,opt_gen=True,opt_disc=False,writer=writer)     # calculate loss functions, get gradients, update network weights
        else:
            model.optimize_parameters(wandb,opt_gen=True,opt_disc=True,writer=writer)

        total_iters+=1
        
        if total_iters % opt.save_bvh == 0 or total_iters==1:
            model.save_networks('latest')
    
            with torch.no_grad():
                model.eval()
                if opt.window_size>0:
                    num_windows = dataset_test.pose_B.shape[0]//opt.window_size
                    skeleton_B = dataset_test.pose_B[:num_windows*opt.window_size].clone()
                    skeleton_B = skeleton_B.reshape([1,num_windows*opt.window_size,dataset_test.pose_B.shape[1],dataset_test.pose_B.shape[2]]).permute(0,2,3,1)
                else:
                    skeleton_B = dataset_test.pose_B.clone()
                skeleton_A = model.netG_B(skeleton_B.cuda().clone(),input_type="matrot").detach()
                skeleton_B = dataset_test.pose_B.clone()
                if opt.dataset.startswith("biped"):
                    skeleton_A_gt = dataset_test.pose_A.clone()
                # import pdb;pdb.set_trace()
                if opt.window_size>0:
                    skeleton_A = skeleton_A.permute(0,3,1,2).reshape([-1,dataset_test.pose_A.shape[1],dataset_test.pose_A.shape[2]])
                if opt.normalize_all:
                    skeleton_A = dataset_test.denormalize(skeleton_A,"A")
                    skeleton_B = dataset_test.denormalize(skeleton_B.cuda(),"B")
                    if opt.dataset.startswith("biped"):
                        skeleton_A_gt = dataset_test.denormalize(skeleton_A_gt.cuda(),"A")

                if opt.remove_virtual_node:
                    skeleton_A = to_full_joint(skeleton_A,dataset_test.infoA['remain_index'],dataset_test.infoA['pose_virtual_index'][0],dataset_test.infoA['pose_virtual_val'])
                    skeleton_B = to_full_joint(skeleton_B,dataset_test.infoB['remain_index'],dataset_test.infoB['pose_virtual_index'][0],dataset_test.infoB['pose_virtual_val'])
                
                    if opt.dataset.startswith("biped"):
                        skeleton_A_gt = to_full_joint(skeleton_A_gt,dataset_test.infoA['remain_index'],dataset_test.infoA['pose_virtual_index'][0],dataset_test.infoA['pose_virtual_val'])
                if opt.velocity_virtual_node:
                    root_A = skeleton_A[:,-1,:3].clone()
                    root_B = skeleton_B[:,-1,:3].clone()
                    skeleton_A[:,-1,:] = torch.eye(3).cuda().reshape(-1)[:6]
                    skeleton_B[:,-1,:] = torch.eye(3).cuda().reshape(-1)[:6]
                   
                    skeleton_A_rotation = to_matrix(skeleton_A)
                    skeleton_B_rotation = to_matrix(skeleton_B)
                   
                    if opt.dataset.startswith("biped"):
                        root_A_gt = skeleton_A_gt[:,-1,:3].clone()
                        skeleton_A_gt[:,-1,:] = torch.eye(3).cuda().reshape(-1)[:6]
                        skeleton_A_gt_rotation = to_matrix(skeleton_A_gt)

                    if opt.use_global_y:
                        root_A[:,[0,2]] = root_A[:,[0,2]].cumsum(dim=0)
                        root_B[:,[0,2]] = root_B[:,[0,2]].cumsum(dim=0)   
                        root_A = root_A.detach().cpu().numpy()
                        root_B = root_B.detach().cpu().numpy()

                        if opt.dataset.startswith("biped"):
                            root_A_gt[:,[0,2]] = root_A_gt[:,[0,2]].cumsum(dim=0)
                            root_A_gt = root_A_gt.detach().cpu().numpy()
                    else:
                        root_A = root_A.cumsum(dim=0).detach().cpu().numpy()
                        root_B = root_B.cumsum(dim=0).detach().cpu().numpy()
                        if opt.dataset.startswith("biped"):
                            root_A_gt = root_A_gt.cumsum(dim=0).detach().cpu().numpy()
                else:
                    skeleton_A_rotation = to_matrix(skeleton_A)
                    skeleton_B_rotation = to_matrix(skeleton_B)
                    root_A = None
                    root_B = None

                    if opt.dataset.startswith("biped"):
                        skeleton_A_gt_rotation = to_matrix(skeleton_A_gt)
                        root_A_gt = None

                #create path if not exist:
                path = os.path.join(opt.checkpoints_dir,opt.name)
                bvh_path = "bvh"

                if not os.path.exists(os.path.join(path,bvh_path)):
                    os.makedirs(os.path.join(path,bvh_path))
                
                if opt.dataset.startswith("biped"):
                    util.save_bvh(skeleton_A_gt_rotation[:].detach().cpu().numpy(),dataset_test.infoA,"{}/{}/B2A_gt_A.bvh".format(path,bvh_path),root_A_gt,normalize=True)
                util.save_bvh(skeleton_A_rotation[:].detach().cpu().numpy(),dataset_test.infoA,"{}/{}/B2A_2_{}.bvh".format(path,bvh_path,total_iters),root_A,normalize=True)
                util.save_bvh(skeleton_B_rotation[:].detach().cpu().numpy(),dataset_test.infoB,"{}/{}/B2A_gt_B.bvh".format(path,bvh_path),root_B,normalize=True)
                model.train()

    if epoch % opt.save_epoch_freq == 0 or epoch==0:              # cache our model every <save_epoch_freq> epochs
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        model.save_networks(str(epoch))

    model.save_networks('latest')
    print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
