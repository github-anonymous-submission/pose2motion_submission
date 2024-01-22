import torch
import torch.nn as nn
from skeleton_aware.skeleton import *
from skeleton_aware.enc_and_dec import *

class Skeleton_aware_Generator(nn.Module):
    def __init__(self, encoder, decoder,fix_virtual_bones=False):
        super(Skeleton_aware_Generator, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.fix_virtual_bones = fix_virtual_bones
        self.mask = None
        self.val = None


    def update(self,info):
        if self.fix_virtual_bones:
            self.mask = torch.tensor(info['virtual_mask']).cuda()
            self.val = torch.tensor(info['pose_virtual_val']).cuda()
            if len(self.mask.shape) == 2:
                self.mask = self.mask.unsqueeze(0)
                self.val = self.val.unsqueeze(0)
            if self.encoder.args.window_size>0:
                self.mask = self.mask.unsqueeze(-1)
                self.val = self.val.unsqueeze(-1)
                if len(self.mask.shape) == 3:
                    self.mask = self.mask.unsqueeze(-1)
                    self.val = self.val.unsqueeze(-1)


    def forward(self, input,input_type=None):
        shape_input = input.shape
        if self.encoder.args.with_root:
            root_rotation = input[:,0,:].clone()
        else:
            root_rotation = None

        if self.encoder.args.velocity_virtual_node:
            root_translation = input[:,-1,:].clone()
            window_size = input.shape[-1]
        else:
            root_translation = None
            window_size = None
        bs = input.shape[0]
        input = self.encoder(input)
        rotation = self.decoder(input,root_rotation=root_rotation,root_translation=root_translation,window_size=window_size,bs=bs)
        # window_size = rotation.shape[-1]
        if window_size is None:
            window_size = 0
        else:
            window_size = rotation.shape[-1]
        
        if self.fix_virtual_bones:
            if window_size>0:
                self.mask = self.mask[[0],:,:,:][:,:,:,[0]].repeat(bs,1,1,window_size)
                self.val = self.val[[0],:,:,:][:,:,:,[0]].repeat(bs,1,1,window_size)
            else:
                self.mask = self.mask[[0],:,:].repeat(bs,1,1)
                self.val = self.val[[0],:,:].repeat(bs,1,1)
            rotation = rotation * self.mask + self.val
        return rotation
    
# def get_channels_list(opt, topolgy):
#     if opt.rotation_data=="rotation_6d":
#         n_channels = 6
#     else:
#         raise NotImplementedError
#     neighbour_list = find_neighbor([topolgy], opt.skeleton_dist)
#     joint_num = len(neighbour_list)

#     base_channel = opt.base_channel if opt.base_channel != -1 else 128
#     n_layers = opt.n_layers if opt.n_layers != -1 else 4

#     channels_list = [n_channels]
#     for i in range(n_layers - 1):
#         channels_list.append(base_channel * (2 ** ((i+1) // 2)))
#     channels_list += [n_channels]
#     channels_list = [((n - 1) // joint_num + 1) * joint_num for n in channels_list]
#     return channels_list, neighbour_list

def get_skeleton_aware_disciminator(args):
    topology_b = args.topology_b
    # channels_list, neighbour_list= get_channels_list(args, topology_b)
    # disc = Conv1dModel(channels_list[:-1] + [1,], args.kernel_size, last_active=None,
    #                        padding_mode=args.padding_mode, batch_norm=args.batch_norm,
    #                        neighbour_list=neighbour_list, skeleton_aware=True)
    # return disc
    edges_b = build_edge_topology(topology_b, torch.zeros((len(topology_b), 3)),args.end_sites_b)
    encoder_b = Encoder(args,edges_b,disc=True)
    return encoder_b 




def get_skeleton_aware_generator(args):
    topology_a = args.topology_a
    topology_b = args.topology_b
    
    edges_a = build_edge_topology(topology_a, torch.zeros((len(topology_a), 3)),args.end_sites_a)
    edges_b = build_edge_topology(topology_b, torch.zeros((len(topology_b), 3)),args.end_sites_b)
    if args.acGAN:
        encoder_a_ = Encoder(args,edges_a)
        decoder_a = Decoder(args,encoder_a_)
        encoder_b = Encoder(args,edges_b)
        netG_B = Skeleton_aware_Generator(encoder_b,decoder_a,args.fix_virtual_bones)
        args.kernel_size = 5
        encoder_b_ = Encoder(args,edges_b,pose_wise=True)
        decoder_b = Decoder(args,encoder_b_)
        encoder_a = Encoder(args,edges_a,pose_wise=True)
        netG_A = Skeleton_aware_Generator(encoder_a,decoder_b,args.fix_virtual_bones)
        return netG_A, netG_B

    else:
        encoder_a = Encoder(args,edges_a)
        decoder_a = Decoder(args,encoder_a)

        encoder_b = Encoder(args,edges_b)
        decoder_b = Decoder(args,encoder_b)

        netG_A = Skeleton_aware_Generator(encoder_a,decoder_b,args.fix_virtual_bones)
        netG_B = Skeleton_aware_Generator(encoder_b,decoder_a,args.fix_virtual_bones)
        return netG_A, netG_B
        