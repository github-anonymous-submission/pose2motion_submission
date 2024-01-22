import subprocess
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

processes = set()
max_processes = 8

vals = [(lambda_A,lambda_B,lambda_End,lambda_gp,lambda_identity,lambda_GAN,
         gen_dis_freq,
         remove_virtual_node,fix_virtual_bones,
         skeleton_aware,
         hidden_poseall,gan_mode,rotation_data,
         rotation,dataset,checkpoints_dir,
         skeleton_par,pool_size,
         batchSize,
         window_size,ee_loss_type,d_poseeach_weight,hidden_poseeach,hidden_poseall,load_pretrained,
         with_root,lambda_End_temporal,velocity_virtual_node,normalize_all,use_global_y,
         connect_end_site, lambda_End_contact,asymmetric,acGAN,v_connect_all_joints,partial_A,use_gt_stat,
         temporal_discriminator) 
         #cycle consistency loss has to go down 
         #pool size too large may not be good for the discriminator
        for lambda_A in [1.,] 
        for lambda_B in [1.,] 
        for lambda_End in [5]
        for lambda_gp in [0.1]
        for lambda_identity in [0.1]
        for lambda_GAN in [0.1]

        for gen_dis_freq in [[0,0]]
        for remove_virtual_node in [0]
 
        for hidden_poseall in [64] 
        for gan_mode in ["wgan-gp"]
        for rotation_data in ['rotation_6d']
        for rotation in ['rotation_6d'] 
        for skeleton_aware in [1]

        for dataset in ['biped']
        for fix_virtual_bones in [0]
        for skeleton_par in [[2,15]]
        for pool_size in [20]
        for batchSize in [128]
        for window_size in [64]
        for checkpoints_dir in ["./code_release_test/mixamo/"]
        for ee_loss_type in ['height']
        for d_poseeach_weight in [20]
        for hidden_poseeach in [64]
        for hidden_poseall in [128]
        for load_pretrained in [0]
        for with_root in [1]
        for lambda_End_temporal in [100.]
        for velocity_virtual_node in [1]
        for normalize_all in [1]
        for use_global_y in [0]
        for connect_end_site in [0]
        for lambda_End_contact in [5]
        for asymmetric in [1]
        for acGAN in [0]
        for v_connect_all_joints in [0]
        for use_gt_stat in [0]
        for partial_A in [1]
        for temporal_discriminator in [1]]

gpu = [5]
gpu_idx = 0
process_idx = 0

for idx, val in enumerate(vals):
    lambda_A,lambda_B,lambda_End,lambda_gp,lambda_identity,lambda_GAN,gen_dis_freq,remove_virtual_node,fix_virtual_bones,skeleton_aware,hidden_poseall,gan_mode,\
                rotation_data,rotation,dataset, checkpoints_dir, skeleton_par, pool_size, batchSize, window_size, ee_loss_type, d_poseeach_weight, \
                hidden_poseeach, hidden_poseall, load_pretrained, with_root, lambda_End_temporal, velocity_virtual_node,\
                normalize_all, use_global_y, connect_end_site, lambda_End_contact, asymmetric, acGAN, v_connect_all_joints, \
                partial_A, use_gt_stat, temporal_discriminator= val
    os.system("cp run_all_wgan_human.py "+checkpoints_dir)
    if remove_virtual_node==0:
        fix_virtual_bones=1
    gen_freq = gen_dis_freq[0]
    disc_freq = gen_dis_freq[1]
    skeleton_num_layers = skeleton_par[0]
    skeleton_kernel_size = skeleton_par[1]

    lambda_B=lambda_A

    
    if len(processes) >= max_processes:
        os.wait()
        processes.difference_update([p for p in processes if p.poll() is not None])
    gpu_id = gpu[gpu_idx]
    process_idx += 1

    processes.add(subprocess.Popen(["python", "train.py", "--name", str(process_idx), 
                                                            "--gpu_ids", str(gpu_id), 
                                                            "--lambda_A", str(lambda_A),
                                                            "--lambda_B", str(lambda_B),
                                                            "--lambda_End", str(lambda_End),
                                                            "--lambda_gp", str(lambda_gp),
                                                            "--lambda_GAN", str(lambda_GAN),
                                                            "--lambda_identity", str(lambda_identity),
                                                            "--gen_freq", str(gen_freq),
                                                            "--disc_freq", str(disc_freq),
                                                            "--remove_virtual_node", str(remove_virtual_node),
                                                            "--fix_virtual_bones", str(fix_virtual_bones),
                                                            "--skeleton_aware", str(skeleton_aware),
                                                            "--hidden_poseall", str(hidden_poseall),
                                                            "--gan_mode", str(gan_mode),
                                                            "--rotation_data", str(rotation_data),
                                                            "--dataset", str(dataset),
                                                            "--rotation", str(rotation),
                                                            "--batchSize", str(batchSize),
                                                            "--partial_A", str(partial_A),
                                                            "--skeleton_num_layers", str(skeleton_num_layers),
                                                            "--skeleton_kernel_size", str(skeleton_kernel_size),
                                                            "--checkpoints_dir", str(checkpoints_dir),
                                                            "--window_size", str(window_size),
                                                            "--ee_loss_type", str(ee_loss_type),
                                                            "--d_poseeach_weight", str(d_poseeach_weight),
                                                            "--hidden_poseeach", str(hidden_poseeach),
                                                            "--hidden_poseall", str(hidden_poseall),
                                                            "--pool_size", str(pool_size),
                                                            "--load_pretrained", str(load_pretrained),
                                                            "--with_root", str(with_root),
                                                            "--lambda_End_temporal", str(lambda_End_temporal),
                                                            "--display_freq", str(1000),
                                                            "--save_bvh", str(1000),
                                                            "--asymmetric", str(asymmetric),
                                                            "--acGAN", str(acGAN),
                                                            "--temporal_discriminator", str(temporal_discriminator),
                                                            "--normalize_all", str(normalize_all),
                                                            "--use_gt_stat", str(use_gt_stat),
                                                            "--velocity_virtual_node", str(velocity_virtual_node),
                                                            "--use_global_y", str(use_global_y),
                                                            "--connect_end_site", str(connect_end_site),
                                                            "--lambda_End_contact", str(lambda_End_contact),
                                                            "--v_connect_all_joints", str(v_connect_all_joints),
                                                            "--n_epochs", str(20),
                                                            "--wandb_project_name", str("Aj2Mousy"),]))
                                                            
                                                            
   
    gpu_idx += 1
    while gpu_idx >= len(gpu):
        os.wait()
        processes.difference_update(
            [p for p in processes if p.poll() is not None])
        if len(processes) == 0:
            gpu_idx = 0


# Check if all the child processes were closed
for p in processes:
    if p.poll() is None:
        p.wait()

