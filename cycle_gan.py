from model_modules import *
import torch
from torch import nn
from torch.nn import functional as F
import itertools
from pose_pool import Pool
import sys
from skeleton_aware.integrate import *
import glob

class CycleGANModel(nn.Module):
    def __init__(self, opt, dataset=None):
        super(CycleGANModel, self).__init__()

        self.opt = opt
        #for denormalization
        self.dataset = dataset
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.total_step = 0
        if opt.skeleton_aware==1:
            self.netG_A, self.netG_B = get_skeleton_aware_generator(opt.skeleton_aware_args)
        else:
            self.netG_A = Generator(num_joints_out=opt.num_joints_B, num_joints_in=opt.num_joints_A, 
                                    hidden_feature=opt.hidden_feature, vae=opt.vae, latent_dim=opt.latent_dim,fix_virtual_bones=opt.fix_virtual_bones)
            self.netG_B = Generator(num_joints_out=opt.num_joints_A, num_joints_in=opt.num_joints_B,
                                    hidden_feature=opt.hidden_feature, vae=opt.vae, latent_dim=opt.latent_dim,fix_virtual_bones=opt.fix_virtual_bones)

        
        if opt.rotation_data=="matrix":
            self.d=9
        elif opt.rotation_data=="quaternion":
            self.d=4
        elif opt.rotation_data=="rotation_6d":
            self.d=6
        
        self.netD_A = Discriminator(input_dim=self.d, num_joints=opt.num_joints_B, hidden_poseeach=opt.hidden_poseeach,hidden_poseall=opt.hidden_poseall, \
                                    poseall_num_layers=opt.poseall_num_layers, hidden_comm=opt.hidden_comm,with_root=opt.with_root,\
                                    velocity_virtual_node=opt.velocity_virtual_node)
        self.netD_B = Discriminator(input_dim=self.d, num_joints=opt.num_joints_A, hidden_poseeach=opt.hidden_poseeach,hidden_poseall=opt.hidden_poseall, \
                                    poseall_num_layers=opt.poseall_num_layers, hidden_comm=opt.hidden_comm,with_root=opt.with_root,\
                                    velocity_virtual_node=opt.velocity_virtual_node)
        if opt.temporal_discriminator:
            self.netD_B_rec = get_skeleton_aware_disciminator(opt.skeleton_aware_args)
            
        self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        self.optimizers = []
        if self.isTrain:
            if opt.root_translation:
                self.mean_A = torch.nn.parameter.Parameter(torch.tensor(torch.zeros(3)).to(self.device),requires_grad=True)
                self.var_A = torch.nn.parameter.Parameter(torch.tensor(torch.ones(3)).to(self.device),requires_grad=True)
                self.optimizer_Scale = torch.optim.Adam([self.mean_A,self.var_A], lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_Scale)
            if opt.optimizer=="Adam":
                self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
                # self.optimizer_G_A = torch.optim.Adam(itertools.chain(self.netG_A.encoder.parameters(), self.netG_B.decoder.parameters()), lr=opt.lr_G_A, betas=(opt.beta1, 0.999))
                # self.optimizer_G_B = torch.optim.Adam(itertools.chain(self.netG_B.encoder.parameters(), self.netG_A.decoder.parameters()), lr=opt.lr_G_B, betas=(opt.beta1, 0.999))
                self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
                if self.opt.temporal_discriminator:
                    self.optimizer_D_rec = torch.optim.Adam(itertools.chain(self.netD_B_rec.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optimizer=="RMSprop":
                self.optimizer_G = torch.optim.RMSprop(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr)
                self.optimizer_D = torch.optim.RMSprop(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr)
            # self.optimizers.append(self.optimizer_G)
            
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            if self.opt.temporal_discriminator:
                self.optimizers.append(self.optimizer_D_rec)
            self.fake_A_pool = Pool(opt.pool_size)
            self.fake_B_pool = Pool(opt.pool_size)
            self.rec_B_pool = Pool(opt.pool_size)

            self.criterionGAN = GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionEndeffector = EndEffectorLoss(opt.remove_virtual_node==1,opt.window_size,loss_type=opt.ee_loss_type,\
                                                        with_root=self.opt.with_root,dataset=self.dataset,\
                                                        velocity_virtual_node=self.opt.velocity_virtual_node,\
                                                        foot_height_A=self.opt.foot_height_A,\
                                                        foot_height_B=self.opt.foot_height_B,\
                                                        use_global_y=self.opt.use_global_y,\
                                                        use_mean_height=self.opt.use_mean_height,ee_reweight=opt.ee_reweight).to(self.device)
        self.not_saved = True
        
    def update_lambda_endeffector(self, epoch):
        self.lambda_endeffector = self.opt.lambda_End * min(1, epoch / 100)

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        # old_lr = self.optimizers[0].param_groups[0]['lr']
        # for scheduler in self.schedulers:
        #     if self.opt.lr_policy == 'plateau':
        #         scheduler.step(self.metric)
        #     else:
        #         scheduler.step()

        # lr = self.optimizers[0].param_groups[0]['lr']
        # print('learning rate %.7f -> %.7f' % (old_lr, lr))

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


    def save_networks(self, epoch):
        """Save all the networks to the disk.
        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        import os
        if self.not_saved:
            n = glob.glob(os.path.join(self.opt.checkpoints_dir,self.opt.name))
            if len(n)>1:
                self.opt.name = self.opt.name + "_%d"%(len(n)+1)
            self.not_saved = False
            # create folder if it doesn't exist
            if not os.path.exists(os.path.join(self.opt.checkpoints_dir,self.opt.name)):
                os.makedirs(os.path.join(self.opt.checkpoints_dir,self.opt.name))
            
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.opt.checkpoints_dir,self.opt.name, save_filename)
                net = getattr(self, 'net' + name)
                torch.save(net.cpu().state_dict(), save_path)
                net.cuda()
            
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.infoA = input['infoA']
        self.infoB = input['infoB']

        if self.opt.window_size<0:
            self.real_A = self.real_A.view(-1, self.opt.num_joints_A,self.d)
            self.real_B = self.real_B.view(-1, self.opt.num_joints_B,self.d)
        else:
            self.real_A = self.real_A
            self.real_B = self.real_B

        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.netG_A.update(self.infoB)
        self.netG_B.update(self.infoA)
        if not self.opt.asymmetric:
            self.fake_B = self.netG_A(self.real_A,input_type="matrot")  # G_A(A) 
            self.rec_A = self.netG_B(self.fake_B,input_type="matrot")   # G_B(G_A(A))
     
        self.fake_A = self.netG_B(self.real_B,input_type="matrot")  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A,input_type="matrot")   # G_A(G_B(B))
        # if self.opt.asymmetric and self.isTrain:
        #     self.fake_B = self.rec_B.clone()
        
    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator
        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator
        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def gradient_penalty(self, netD, real, fake):
        alpha = torch.rand((1,)).to(self.device)
        interpolates = (alpha * real + ((1 - alpha) * fake.detach())).requires_grad_(True)
        d_interpolates = netD(interpolates)
        gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                        grad_outputs=torch.ones(d_interpolates.size()).to(self.device),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.opt.lambda_gp
        gradient_penalty.backward()
        return gradient_penalty
    
    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B.detach())
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
        if self.opt.gan_mode == "wgan-gp":
            self.loss_gp_A = self.gradient_penalty(self.netD_A, self.real_B, fake_B)
        else:
            self.loss_gp_A = 0

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A.detach())
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
        if self.opt.gan_mode == "wgan-gp":
            self.loss_gp_B = self.gradient_penalty(self.netD_B, self.real_A, fake_A)
        else:
            self.loss_gp_B = 0

    def backward_D_B_rec(self):
        """Calculate GAN loss for discriminator D_B"""
        rec_B = self.rec_B_pool.query(self.rec_B.detach())
        self.loss_D_B_rec = self.backward_D_basic(self.netD_B_rec, self.real_B, rec_B)
        if self.opt.gan_mode == "wgan-gp":
            self.loss_gp_B_rec = self.gradient_penalty(self.netD_B_rec, self.real_B, rec_B)
        else:
            self.loss_gp_B_rec = 0
        

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_End = self.opt.lambda_End
        lambda_GAN = self.opt.lambda_GAN
        lambda_End_temporal = self.opt.lambda_End_temporal
        lambda_End_contact = self.opt.lambda_End_contact
        lambda_loss_height = self.opt.lambda_loss_height

        # Identity loss
        if lambda_idt > 0 and (not self.opt.acGAN):
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            # self.idt_A = self.netG_A(self.real_B)
            if self.opt.with_root:
                root_rotation_A = self.real_A[:,0,:].clone()
                root_rotation_B = self.real_B[:,0,:].clone()
            else:
                root_rotation_A = None
                root_rotation_B = None
            if self.opt.velocity_virtual_node:
                root_translation_A = self.real_A[:,-1,:].clone()
                root_translation_B = self.real_B[:,-1,:].clone()
            else:
                root_translation_A = None
                root_translation_B = None
            if not self.opt.asymmetric:
                self.idt_A = self.netG_B.decoder(self.netG_A.encoder(self.real_A),root_rotation=root_rotation_A,root_translation=root_translation_A)
                self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_A)
            else:
                self.loss_idt_A = 0

            self.idt_B = self.netG_A.decoder(self.netG_B.encoder(self.real_B),root_rotation=root_rotation_B,root_translation=root_translation_B)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_B) 
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        if lambda_GAN > 0:
            if not self.opt.asymmetric:
                self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
            else:
                self.loss_G_A = 0

            # GAN loss D_B(G_B(B))
            self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        else:
            self.loss_G_A = 0
            self.loss_G_B = 0
            
        # Forward cycle loss || G_B(G_A(A)) - A||
        if not self.opt.asymmetric:
            if self.opt.reweight:
                weight = torch.ones_like(self.real_A,device=self.device)
                weight[:,[0,-1],...] = 5
                self.loss_cycle_A = self.criterionCycle(self.rec_A*weight, self.real_A*weight)
            else:
                self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A)
        else:
            self.loss_cycle_A = 0

        # Backward cycle loss || G_A(G_B(B)) - B||
        if self.opt.reweight:
            weight = torch.ones_like(self.real_B,device=self.device)
            weight[:,[0,-1],...] = 5
            self.loss_cycle_B = self.criterionCycle(self.rec_B*weight, self.real_B*weight)
        else:
            self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B)
           
        # combined loss and calculate gradients
        if not self.opt.asymmetric:
            self.endeff_loss_A_dict = self.criterionEndeffector(self.real_A, self.netG_A(self.real_A.clone(),input_type="matrot"), self.infoA, self.infoB)
            self.endeff_loss_A = self.endeff_loss_A_dict["loss_relative_to_T"]
        else:
            self.endeff_loss_A = 0

        self.endeff_loss_B_dict = self.criterionEndeffector(self.netG_B(self.real_B.clone(),input_type="matrot"), self.real_B, self.infoA, self.infoB,temporal=True)
        self.endeff_loss_B = self.endeff_loss_B_dict["loss_relative_to_T"]
        self.endeff_loss_B_temporal = self.endeff_loss_B_dict["loss_relative_temporal"]
        self.endeff_loss_B_contact = self.endeff_loss_B_dict["loss_contact"]
        self.endeff_loss_B_height = self.endeff_loss_B_dict["loss_height_contact"]  
     
        self.loss_G = self.loss_G_A * lambda_GAN + self.loss_G_B * lambda_GAN \
                                + self.loss_cycle_A * lambda_A + self.loss_cycle_B * lambda_B \
                                + self.loss_idt_A * lambda_A * lambda_idt \
                                + self.loss_idt_B * lambda_B * lambda_idt \
                                + self.endeff_loss_A * lambda_End \
                                + self.endeff_loss_B * lambda_End \
                                + self.endeff_loss_B_temporal * lambda_End_temporal \
                                + self.endeff_loss_B_contact * lambda_End_contact \
                                + self.endeff_loss_B_height * lambda_loss_height
                        
        self.loss_G.backward()
                                

       

    def optimize_parameters(self,wandb=None,opt_gen=False,opt_disc=False,writer=None,epoch=0):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        if self.opt.lambda_End_scheduling:
            self.update_lambda_endeffector()
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        if opt_gen:
            self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
            self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
            if self.opt.root_translation:
                self.optimizer_Scale.zero_grad()
            self.backward_G()             # calculate gradients for G_A and G_B
            self.optimizer_G.step()       # update G_A and G_B's weights
            if self.opt.root_translation:
                self.optimizer_Scale.step()
            dict = {"loss_G":self.loss_G,"loss_G_A":self.loss_G_A,"loss_G_B":self.loss_G_B,"loss_cycle_A":self.loss_cycle_A,"loss_cycle_B":self.loss_cycle_B,
                            "loss_idt_A":self.loss_idt_A,"loss_idt_B":self.loss_idt_B,"endeff_loss_A":self.endeff_loss_A,
                            "endeff_loss_B":self.endeff_loss_B,"endeff_loss_B_temporal":self.endeff_loss_B_temporal,"endeff_loss_B_contact":self.endeff_loss_B_contact,\
                            "endeff_loss_B_height":self.endeff_loss_B_height}
            if wandb!=None:
                wandb.log(dict)
                if self.opt.with_root:
                    wandb.log({"root mean":self.fake_A[:,0,:,:].abs().mean()})
                if self.opt.root_translation:
                    wandb.log({"mean":self.mean_A.mean().item()})
                    wandb.log({"var":self.var_A.mean().item()})
            if writer!=None:
                for key, value in dict.items():
                    writer.add_scalar(key, value, self.total_step)
                
        # D_A and D_B
        if opt_disc and self.opt.lambda_GAN > 0.0:
            self.set_requires_grad([self.netD_A, self.netD_B], True)
            self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
            if not self.opt.asymmetric:
                self.backward_D_A() 
            else:     # calculate gradients for D_A
                self.loss_D_A = 0
                self.loss_gp_A = 0
            self.backward_D_B()      # calculate graidents for D_B
            if self.opt.temporal_discriminator:
                self.optimizer_D_rec.zero_grad()
                self.backward_D_B_rec()
                self.optimizer_D_rec.step()
                dict_temporal = {"loss_D_B_rec":self.loss_D_B_rec,"loss_gp_B_rec":self.loss_gp_B_rec}
            else:
                dict_temporal = {}
            self.optimizer_D.step()  # update D_A and D_B's weights
            dict = {"loss_D_A":self.loss_D_A,"loss_D_B":self.loss_D_B,"loss_gp_A":self.loss_gp_A,"loss_gp_B":self.loss_gp_B}
            if wandb!=None:
                wandb.log(dict)
                wandb.log(dict_temporal)
            if writer!=None:
                for key, value in dict.items():
                    writer.add_scalar(key, value, self.total_step)  
                
        self.total_step+=1
      