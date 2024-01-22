import argparse
import os
import torch
from util import *

def get_skeleton_aware_options():
    args = argparse.Namespace()
    args.rotation = 'quaternion'
    args.kernel_size = 5
    args.skeleton_info = "none"
    args.num_layers = 2
    args.extra_conv = 0
    args.batch_norm = 0
    args.n_layers = -1
    args.base_channel = -1
    args.skeleton_dist = 2
    args.padding_mode = "reflection"
    args.skeleton_pool = "mean"
    args.pos_repr = "4d"
    args.upsampling = "linear"
    return args


class TrainOptions():
    """This class defines options used during both training and test time.
    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # model related parameters
        parser.add_argument('--dataroot', type=str,default="./data/", help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--name', type=str, default='endeffector_all', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
        parser.add_argument('--dataset', type=str, default='biped', help='chooses how datasets are loaded. [unaligned | aligned | single]')
        parser.add_argument('--model', type=str, default='cycle_gan', help='chooses which model to use. cycle_gan, pix2pix, test')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
        parser.add_argument('--num_joints_A', type=int, default=27)
        parser.add_argument('--num_joints_B', type=int, default=27)
        parser.add_argument('--hidden_feature', type=int, default=256)
        parser.add_argument('--latent_dim', type=int, default=32)
        parser.add_argument('--vae', action='store_true', help='use vae')
        parser.add_argument('--num_threads', default=3, type=int, help='# threads for loading data')
        parser.add_argument('--hidden_poseeach', type=int, default=32)
        parser.add_argument('--hidden_poseall', type=int, default=1024)
        parser.add_argument('--poseall_num_layers', type=int, default=2)
        
        parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
        parser.add_argument('--hidden_comm', type=int, default=32)
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--beta1', type=float, default=0.5)
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--pool_size', type=int, default=100, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        # visdom and HTML visualization parameters
        parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving the latest results')
        parser.add_argument('--display_freq', type=int, default=500, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_bvh', type=int, default=1000, help='frequency of showing training results on console')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--overfit', type=int, default=0, help='if we overfit')
       
        parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=0, help='number of epochs to linearly decay learning rate to zero')

        parser.add_argument('--skeleton_aware', type=int, default=1, help='if use the skeleton_aware model')
        parser.add_argument('--fix_virtual_bones', type=int, default=0, help='if use the skeleton_aware model')
        # wandb parameters
        parser.add_argument('--use_wandb', action='store_true', help='if specified, then init wandb logging')
        parser.add_argument('--wandb_project_name', type=str, default='MotionAdaptation_heightEE', help='specify wandb project name')

        # cycle consistency loss
        parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
        parser.add_argument('--lambda_End', type=float, default=1, help='weight for endeffectors loss')
        parser.add_argument('--lambda_GAN', type=float, default=1, help='weight for endeffectors loss')
        parser.add_argument('--lambda_identity', type=float, default=0, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
        parser.add_argument('--lambda_gp', type=float, default=0, help='weight for lambda_gp loss')
        parser.add_argument('--lambda_End_scheduling', type=int, default=0, )
        parser.add_argument('--lambda_End_temporal', type=float, default=1., )
        parser.add_argument('--lambda_End_contact', type=float, default=1., )
        parser.add_argument('--ee_loss_type', type=str, default="bbox")
        parser.add_argument('--d_poseeach_weight', type=int, default=10)
        parser.add_argument('--asymmetric', type=int, default=0)
        parser.add_argument('--acGAN', type=int, default=0)
        parser.add_argument('--v_connect_all_joints', type=int, default=0)
        parser.add_argument('--lambda_loss_height', type=float, default=0)
        parser.add_argument('--ee_reweight', type=int, default=0)
        
        parser.add_argument('--window_size', type=int, default=-1, )
        parser.add_argument('--with_root', type=int, default=0, )
        parser.add_argument('--root_translation', type=int, default=0, )
        parser.add_argument('--velocity_virtual_node', type=int, default=0, )
        parser.add_argument('--constant_A', type=int, default=0, )
        parser.add_argument('--vector', type=int, default=0, )
        parser.add_argument('--reweight', type=int, default=0, )
        self.initialized = True
        
        ### GAN related parameter
        parser.add_argument('--disc_freq', type=int, default=0,)
        parser.add_argument('--gen_freq', type=int, default=0,)
        parser.add_argument('--skeleton_num_layers', type=int, default=2,)
        parser.add_argument('--skeleton_kernel_size', type=int, default=5)
        parser.add_argument('--skeleton_dist', type=int, default=2)
        parser.add_argument('--optimizer', type=str, default="Adam")
        parser.add_argument('--load_pretrained', type=int, default=0)
        parser.add_argument('--pretrained_path', type=str, default="/")

        ### data related parameter
        parser.add_argument('--rotation_data', type=str, default='matrix')
        parser.add_argument('--rotation', type=str, default='quaternion')
        parser.add_argument('--remove_virtual_node', type=int, default=0)
        parser.add_argument('--normalize_all', type=int, default=0)
        parser.add_argument('--use_global_y', type=int, default=0)
        parser.add_argument('--connect_end_site', type=int, default=0)
        parser.add_argument('--partial_A', type=float, default=1)
        parser.add_argument('--use_gt_stat', type=int, default=0)
        parser.add_argument('--balance', type=int, default=0)
        parser.add_argument('--last_sigmoid', type=float, default=0)
        parser.add_argument('--scale_factor', type=int, default=2)
        parser.add_argument('--use_mean_height', type=int, default=0)
        parser.add_argument('--temporal_discriminator', type=int, default=0)
        parser.add_argument('--animal', type=str, default="none")
        parser.add_argument('--dataset_subsample', type=float, default=1.)
        parser.add_argument('--normalize_translation_only', type=int, default=0)

        # identity_test
        parser.add_argument('--identity_test', type=int, default=0)

        
        return parser

    def print_options(self, opt):
        """Print and save options
        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        mkdirs(expr_dir) 
        
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        self.parser = parser
        return parser.parse_args()
    
    def update(self,opt):
        fname_vars = ['gan_mode','rotation_data','rotation','optimizer','batchSize','remove_virtual_node','skeleton_num_layers','skeleton_kernel_size','pool_size', "disc_freq", "gen_freq", 'lambda_A', 'lambda_B', 'lambda_End', 'lambda_GAN','lambda_gp', 'lambda_identity','window_size','ee_loss_type']
        
        # opt.name = ''.join([f'{k[:8]}_{vars(opt)[k]}_|_'.replace('[', '(').replace(']',')') for k in fname_vars])[16:-1]
        # if len(opt.name) > 200:
        #     opt.name = ''.join([f'{k[:6]}_{vars(opt)[k]}_|_'.replace('[', '(').replace(']',')') for k in fname_vars])[14:-1] #14
        # if opt.load_pretrained==1:
        #     fname_vars = ['gan_mode','load_pretrained','rotation_data','rotation','optimizer','batchSize','remove_virtual_node','skeleton_num_layers','skeleton_kernel_size','pool_size', "disc_freq", "gen_freq", 'lambda_A', 'lambda_B', 'lambda_End', 'lambda_GAN','lambda_gp', 'lambda_identity','window_size','ee_loss_type','hidden_poseeach','hidden_poseall']
        #     opt.name = ''.join([f'{k[:8]}_{vars(opt)[k]}_|_'.replace('[', '(').replace(']',')') for k in fname_vars])[16:-1]
        #     if len(opt.name) > 200:
        #         opt.name = ''.join([f'{k[:6]}_{vars(opt)[k]}_|_'.replace('[', '(').replace(']',')') for k in fname_vars])[16:-1]
        #         if len(opt.name) > 200:
        #             opt.name = ''.join([f'{k[:4]}_{vars(opt)[k]}_|_'.replace('[', '(').replace(']',')') for k in fname_vars])

        # else:
        #     if opt.dataset.startswith("biped"):
        #         fname_vars = ['gan_mode','rotation_data','rotation','optimizer','batchSize','remove_virtual_node','skeleton_num_layers','skeleton_kernel_size','pool_size', "disc_freq", "gen_freq", 'lambda_A', 'lambda_B', 'lambda_End', 'lambda_GAN','lambda_gp', 'lambda_identity','window_size','ee_loss_type']
        #         opt.name = ''.join([f'{k[:8]}_{vars(opt)[k]}_|_'.replace('[', '(').replace(']',')') for k in fname_vars])[16:-1]
        #         if len(opt.name) > 200:
        #             opt.name = ''.join([f'{k[:6]}_{vars(opt)[k]}_|_'.replace('[', '(').replace(']',')') for k in fname_vars])[14:-1]
        #     else:
        #         fname_vars = ['gan_mode','rotation_data','rotation','optimizer','batchSize','remove_virtual_node','skeleton_num_layers','skeleton_kernel_size','pool_size', "disc_freq", "gen_freq", 'lambda_A', 'lambda_B', 'lambda_End', 'lambda_GAN','lambda_gp', 'lambda_identity','window_size','ee_loss_type','hidden_poseeach','hidden_poseall']
        #         opt.name = ''.join([f'{k[:8]}_{vars(opt)[k]}_|_'.replace('[', '(').replace(']',')') for k in fname_vars])[16:-1]
        #         if len(opt.name) > 200:
        #             opt.name = ''.join([f'{k[:6]}_{vars(opt)[k]}_|_'.replace('[', '(').replace(']',')') for k in fname_vars])[16:-1]
        #             if len(opt.name) > 200:
        #                 opt.name = ''.join([f'{k[:4]}_{vars(opt)[k]}_|_'.replace('[', '(').replace(']',')') for k in fname_vars])
        if opt.load_pretrained==1:
            fname_vars = ['gan_mode','load_pretrained','rotation_data','rotation','optimizer','batchSize','remove_virtual_node','skeleton_num_layers','skeleton_kernel_size','pool_size', "disc_freq", "gen_freq", 'lambda_A', 'lambda_B', 'lambda_End', 'lambda_GAN','lambda_gp', 'lambda_identity','window_size','ee_loss_type','hidden_poseeach','hidden_poseall']
            opt.name = ''.join([f'{k[:8]}_{vars(opt)[k]}_|_'.replace('[', '(').replace(']',')') for k in fname_vars])[16:-1]
            if len(opt.name) > 200:
                opt.name = ''.join([f'{k[:6]}_{vars(opt)[k]}_|_'.replace('[', '(').replace(']',')') for k in fname_vars])[16:-1]
                if len(opt.name) > 200:
                    opt.name = ''.join([f'{k[:4]}_{vars(opt)[k]}_|_'.replace('[', '(').replace(']',')') for k in fname_vars])

        elif opt.with_root==1:
            fname_vars = ['gan_mode','rotation_data','rotation','optimizer','batchSize','remove_virtual_node','skeleton_num_layers','skeleton_kernel_size','pool_size', "disc_freq", "gen_freq", 'lambda_A', 'lambda_B', 'lambda_End', 'lambda_GAN','lambda_gp', 'lambda_identity','window_size','ee_loss_type','hidden_poseeach','hidden_poseall','with_root','lambda_End_temporal']
            if opt.use_global_y:
                fname_vars.append('use_global_y')
            if opt.connect_end_site:
                fname_vars.append('connect_end_site')
            if opt.velocity_virtual_node:
                fname_vars.append('velocity_virtual_node')
            if opt.normalize_all:
                fname_vars.append('normalize_all')
            if opt.lambda_End_contact:
                fname_vars.append('lambda_End_contact')
            if opt.asymmetric:
                fname_vars.append('asymmetric')
            if opt.acGAN:
                fname_vars.append('acGAN')
            if opt.v_connect_all_joints:
                fname_vars.append('v_connect_all_joints')
            if opt.partial_A!=1.:
                fname_vars.append('partial_A')
            if opt.use_gt_stat:
                fname_vars.append('use_gt_stat')
                
            if opt.balance:
                fname_vars.append("balance")
            if opt.lambda_loss_height>0:
                fname_vars.append("lambda_loss_height")
            if opt.last_sigmoid:
                fname_vars.append("last_sigmoid")
            if opt.constant_A:
                fname_vars.append('constant_A')
            if opt.vector:
                fname_vars.append('vector')
            if opt.reweight:
                fname_vars.append('reweight')
            if opt.scale_factor!=2:
                fname_vars.append('scale_factor')
            if opt.skeleton_dist!=2:
                fname_vars.append('skeleton_dist')
            if opt.use_mean_height:
                fname_vars.append('use_mean_height')
            if opt.temporal_discriminator:
                fname_vars.append('temporal_discriminator')
            if opt.animal!="none":
                fname_vars.insert(0,"animal")
            if opt.ee_reweight==1:
                fname_vars.append('ee_reweight')
            if opt.normalize_translation_only==1:
                fname_vars.append('normalize_translation_only')
            fname_vars.append(opt.dataset_subsample)

            opt.name = ''.join([f'{k[:8]}_{vars(opt)[k]}_|_'.replace('[', '(').replace(']',')') for k in fname_vars])[16:-1]
            if len(opt.name) > 200:
                opt.name = ''.join([f'{k[:4]}_{vars(opt)[k]}_|_'.replace('[', '(').replace(']',')') for k in fname_vars])[16:-1]
                if len(opt.name) > 200:
                    opt.name = ''.join([f'{k[:2]}_{vars(opt)[k]}_|_'.replace('[', '(').replace(']',')') for k in fname_vars])
                    if len(opt.name)>200:
                        opt.name = ''.join([f'{k[:1]}_{vars(opt)[k]}_|_'.replace('[', '(').replace(']',')') for k in fname_vars])
                        if len(opt.name)>200:
                            opt.name = ''.join([f'{k[:1]}{vars(opt)[k]}' for k in fname_vars])


        else:
            fname_vars = ['gan_mode','rotation_data','rotation','optimizer','batchSize','remove_virtual_node','skeleton_num_layers','skeleton_kernel_size','pool_size', "disc_freq", "gen_freq", 'lambda_A', 'lambda_B', 'lambda_End', 'lambda_GAN','lambda_gp', 'lambda_identity','window_size','ee_loss_type','hidden_poseeach','hidden_poseall']
            if opt.scale_factor!=2:
                fname_vars.append('scale_factor')
            if opt.skeleton_dist!=2:
                fname_vars.append('skeleton_dist')
                
            if opt.use_gt_stat:
                fname_vars.append('use_gt_stat')
            opt.name = ''.join([f'{k[:8]}_{vars(opt)[k]}_|_'.replace('[', '(').replace(']',')') for k in fname_vars])[16:-1]
            if len(opt.name) > 200:
                opt.name = ''.join([f'{k[:6]}_{vars(opt)[k]}_|_'.replace('[', '(').replace(']',')') for k in fname_vars])[16:-1]
                if len(opt.name) > 200:
                    opt.name = ''.join([f'{k[:4]}_{vars(opt)[k]}_|_'.replace('[', '(').replace(']',')') for k in fname_vars])
                    if len(opt.name)>200:
                        opt.name = ''.join([f'{k[:1]}_{vars(opt)[k]}_|_'.replace('[', '(').replace(']',')') for k in fname_vars])
                        if len(opt.name)>200:
                            opt.name = ''.join([f'{k[:1]}{vars(opt)[k]}' for k in fname_vars])
        return opt
    
    def update_name(self,opt2,removed_name=[]):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = self.initialize(parser)
        self.parser = parser
        fname_vars = ['name', 'skeleton_aware', 'gan_mode','rotation_data','rotation','remove_virtual_node','dataset', "disc_freq", "gen_freq", 'lambda_A', 'lambda_B', 'lambda_End', 'lambda_GAN','lambda_gp', 'overfit','fix_virtual_bones','d_poseeach_weight']
        for removed_name_ in removed_name:
            fname_vars.remove(removed_name_)
        opt2.name = ''.join([f'{k}_{vars(opt2)[k]}_|_'.replace('[', '(').replace(']',')') for k in fname_vars])[16:-1]

        return opt2
    
    def parse_jupyter(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = self.initialize(parser)
        self.parser = parser
        opt, unknown = parser.parse_known_args()

        if opt.phase == 'train':
            self.isTrain = True
        else:
            self.isTrain = False
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        # fname_vars = ['name', 'skeleton_aware', 'gan_mode','rotation_data','rotation','remove_virtual_node', "disc_freq", "gen_freq", 'lambda_A', 'lambda_B', 'lambda_End', 'lambda_GAN','lambda_gp', 'overfit','fix_virtual_bones']
        fname_vars = ['gan_mode','rotation_data','rotation','remove_virtual_node', "disc_freq", "gen_freq", 'lambda_A', 'lambda_B', 'lambda_End', 'lambda_GAN','lambda_gp', 'skeleton_num_layers']
        
        opt.name = ''.join([f'{k}_{vars(opt)[k]}_|_'.replace('[', '(').replace(']',')') for k in fname_vars])[16:-1]
        

        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
    
    # def parse_npy(self):

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        
        if opt.phase == 'train':
            self.isTrain = True
        else:
            self.isTrain = False
        opt.isTrain = self.isTrain   # train or test
        opt.hidden_comm = opt.hidden_poseeach
        opt.lr_G_A = opt.lr
        opt.lr_G_B = opt.lr
        
        # process opt.suffix
        # fname_vars = ['name', 'skeleton_aware', 'gan_mode','rotation_data','rotation','remove_virtual_node','dataset', "disc_freq", "gen_freq", 'lambda_A', 'lambda_B', 'lambda_End', 'lambda_GAN','lambda_gp', 'overfit','fix_virtual_bones']
        if opt.load_pretrained==1:
            fname_vars = ['gan_mode','load_pretrained','rotation_data','rotation','optimizer','batchSize','remove_virtual_node','skeleton_num_layers','skeleton_kernel_size','pool_size', "disc_freq", "gen_freq", 'lambda_A', 'lambda_B', 'lambda_End', 'lambda_GAN','lambda_gp', 'lambda_identity','window_size','ee_loss_type','hidden_poseeach','hidden_poseall']
            opt.name = ''.join([f'{k[:8]}_{vars(opt)[k]}_|_'.replace('[', '(').replace(']',')') for k in fname_vars])[16:-1]
            if len(opt.name) > 200:
                opt.name = ''.join([f'{k[:6]}_{vars(opt)[k]}_|_'.replace('[', '(').replace(']',')') for k in fname_vars])[16:-1]
                if len(opt.name) > 200:
                    opt.name = ''.join([f'{k[:4]}_{vars(opt)[k]}_|_'.replace('[', '(').replace(']',')') for k in fname_vars])

        elif opt.with_root==1:
            fname_vars = ['gan_mode','rotation_data','rotation','optimizer','batchSize','remove_virtual_node','skeleton_num_layers','skeleton_kernel_size','pool_size', "disc_freq", "gen_freq", 'lambda_A', 'lambda_B', 'lambda_End', 'lambda_GAN','lambda_gp', 'lambda_identity','window_size','ee_loss_type','hidden_poseeach','hidden_poseall','with_root','lambda_End_temporal']
            if opt.use_global_y:
                fname_vars.append('use_global_y')
            if opt.connect_end_site:
                fname_vars.append('connect_end_site')
            if opt.velocity_virtual_node:
                fname_vars.append('velocity_virtual_node')
            if opt.normalize_all:
                fname_vars.append('normalize_all')
            if opt.lambda_End_contact:
                fname_vars.append('lambda_End_contact')
            if opt.asymmetric:
                fname_vars.append('asymmetric')
            if opt.acGAN:
                fname_vars.append('acGAN')
            if opt.v_connect_all_joints:
                fname_vars.append('v_connect_all_joints')
            if opt.partial_A!=1.:
                fname_vars.append('partial_A')
            if opt.use_gt_stat:
                fname_vars.append('use_gt_stat')
                
            if opt.balance:
                fname_vars.append("balance")
            if opt.lambda_loss_height>0:
                fname_vars.append("lambda_loss_height")
            if opt.last_sigmoid:
                fname_vars.append("last_sigmoid")
            if opt.constant_A:
                fname_vars.append('constant_A')
            if opt.vector:
                fname_vars.append('vector')
            if opt.reweight:
                fname_vars.append('reweight')
            if opt.scale_factor!=2:
                fname_vars.append('scale_factor')
            if opt.skeleton_dist!=2:
                fname_vars.append('skeleton_dist')
            if opt.use_mean_height:
                fname_vars.append('use_mean_height')
            if opt.temporal_discriminator:
                fname_vars.append('temporal_discriminator')
            if opt.animal!="none":
                fname_vars.insert(0,"animal")
            if opt.ee_reweight==1:
                fname_vars.append('ee_reweight')
            if opt.normalize_translation_only==1:
                fname_vars.append('normalize_translation_only')
            fname_vars.append('dataset_subsample')


            opt.name = ''.join([f'{k[:8]}_{vars(opt)[k]}_|_'.replace('[', '(').replace(']',')') for k in fname_vars])[16:-1]
            if len(opt.name) > 200:
                opt.name = ''.join([f'{k[:4]}_{vars(opt)[k]}_|_'.replace('[', '(').replace(']',')') for k in fname_vars])[16:-1]
                if len(opt.name) > 200:
                    opt.name = ''.join([f'{k[:2]}_{vars(opt)[k]}_|_'.replace('[', '(').replace(']',')') for k in fname_vars])
                    if len(opt.name)>200:
                        opt.name = ''.join([f'{k[:1]}_{vars(opt)[k]}_|_'.replace('[', '(').replace(']',')') for k in fname_vars])
                        if len(opt.name)>200:
                            opt.name = ''.join([f'{k[:1]}{vars(opt)[k]}' for k in fname_vars])

        else:
            fname_vars = ['gan_mode','rotation_data','rotation','optimizer','batchSize','remove_virtual_node','skeleton_num_layers','skeleton_kernel_size','pool_size', "disc_freq", "gen_freq", 'lambda_A', 'lambda_B', 'lambda_End', 'lambda_GAN','lambda_gp', 'lambda_identity','window_size','ee_loss_type','hidden_poseeach','hidden_poseall']
            if opt.scale_factor!=2:
                fname_vars.append('scale_factor')
            if opt.skeleton_dist!=2:
                fname_vars.append('skeleton_dist')

            if opt.use_gt_stat:
                fname_vars.append('use_gt_stat')
            opt.name = ''.join([f'{k[:8]}_{vars(opt)[k]}_|_'.replace('[', '(').replace(']',')') for k in fname_vars])[16:-1]
            if len(opt.name) > 200:
                opt.name = ''.join([f'{k[:6]}_{vars(opt)[k]}_|_'.replace('[', '(').replace(']',')') for k in fname_vars])[16:-1]
                if len(opt.name) > 200:
                    opt.name = ''.join([f'{k[:4]}_{vars(opt)[k]}_|_'.replace('[', '(').replace(']',')') for k in fname_vars])
                    if len(opt.name)>200:
                        opt.name = ''.join([f'{k[:1]}_{vars(opt)[k]}_|_'.replace('[', '(').replace(']',')') for k in fname_vars])
                        if len(opt.name)>200:
                            opt.name = ''.join([f'{k[:1]}{vars(opt)[k]}' for k in fname_vars])

        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt