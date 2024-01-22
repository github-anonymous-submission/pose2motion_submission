import torch
import torch.nn as nn
from skeleton_aware.skeleton import SkeletonUnpool, SkeletonPool, SkeletonConv, find_neighbor, SkeletonLinear
import pytorch3d.transforms
from data_util import *

class Encoder(nn.Module):
    def __init__(self, args, topology, use_skeleton_linear=False,pose_wise=False,disc=False):
        super(Encoder, self).__init__()
        self.topologies = [topology]
        if args.rotation == 'euler_angle': self.channel_base = [3]
        elif args.rotation == 'quaternion': self.channel_base = [4]
        elif args.rotation == 'rotation_6d': self.channel_base = [6]
        # if args.root_translation: self.channel_base[0] += 3
        self.rotation = args.rotation
        self.channel_list = []
        self.edge_num = [len(topology) + 1]
        self.pooling_list = []
        self.layers = nn.ModuleList()
        self.args = args
        self.convs = []
        self.use_skeleton_linear = use_skeleton_linear
        self.pose_wise = pose_wise

        kernel_size = args.kernel_size
        padding = (kernel_size - 1) // 2
        bias = True
        if args.skeleton_info == 'concat': add_offset = True
        else: add_offset = False

        for i in range(args.num_layers):
            if self.use_skeleton_linear:
                self.channel_base.append(self.channel_base[-1])
            else:
                self.channel_base.append(self.channel_base[-1] * self.args.scale_factor)
        
        for i in range(args.num_layers):
            seq = []
            neighbor_list = find_neighbor(self.topologies[i], args.skeleton_dist)
            if self.args.v_connect_all_joints:
                neighbor_list[-1] = [i for i in range(len(self.topologies[i]))]
            in_channels = self.channel_base[i] * self.edge_num[i]
            out_channels = self.channel_base[i+1] * self.edge_num[i]
            if i==args.num_layers-1 and disc:
                out_channels = 1*self.edge_num[i]
            if i == 0: self.channel_list.append(in_channels)
            self.channel_list.append(out_channels)
            # print('Layer {}: {} -> {}'.format(i, in_channels, out_channels))
            # print('Neighbor list: {}'.format(neighbor_list))
            # print("len of neighbor list: {}".format(len(neighbor_list)))
            for _ in range(args.extra_conv):
                if self.use_skeleton_linear:
                    seq.append(SkeletonLinear(neighbor_list, in_channels=in_channels, out_channels=in_channels))
                else:
                    seq.append(SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=in_channels,
                                            joint_num=self.edge_num[i], kernel_size=kernel_size, stride=1,
                                            padding=padding, padding_mode=args.padding_mode, bias=bias))
                # seq.append(SkeletonLinear(neighbor_list, in_channels=in_channels, out_channels=in_channels))
            if self.use_skeleton_linear:
                seq.append(SkeletonLinear(neighbor_list, in_channels=in_channels, out_channels=out_channels))
                print("use linear layer, in_channels, out_channels",in_channels,out_channels)
            else:
                seq.append(SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=out_channels,
                                        joint_num=self.edge_num[i], kernel_size=kernel_size, stride=2,
                                        padding=padding, padding_mode=args.padding_mode, bias=bias, add_offset=add_offset,
                                        in_offset_channel=3 * self.channel_base[i] // self.channel_base[0]))
                # print("neighbor_list",neighbor_list,)
                # print('edge_num',self.edge_num)
                # print('channel_base',self.channel_base)
                # print("use conv layer, in_channels, out_channels",in_channels,out_channels)
            self.convs.append(seq[-1])
            last_pool = True if i == args.num_layers - 1 else False
            # print("self.topologies[i]",len(self.topologies[i]),self.topologies[i])
            pool = SkeletonPool(edges=self.topologies[i], pooling_mode=args.skeleton_pool,
                                channels_per_edge=out_channels // len(neighbor_list), last_pool=last_pool)
            # print("pooling list",len(pool.new_edges),"pooling_mode", args.skeleton_pool,"pooling_list",pool.pooling_list)
            seq.append(pool)
            seq.append(nn.LeakyReLU(negative_slope=0.2))
            self.layers.append(nn.Sequential(*seq))

            self.topologies.append(pool.new_edges)
            self.pooling_list.append(pool.pooling_list)
            self.edge_num.append(len(self.topologies[-1]) + 1)
            if i == args.num_layers - 1:
                self.last_channel = self.edge_num[-1] * self.channel_base[i + 1]

    

    def forward(self, input, offset=None,input_type=None):
        '''
        input shape: (batch_size, joint_num, 6, window_size)
        '''
        
        input = format_rotation(input,self.rotation)
        if not self.use_skeleton_linear and self.args.window_size<0:
            input = input.reshape([input.shape[0],-1])[...,None].repeat([1,1,3])
        if self.pose_wise:
            bs,js,d,ws = input.shape
            input = input.permute([0,3,1,2]).reshape([-1,js,d]).reshape([-1,js*d])[...,None].repeat([1,1,8])
            # bs*ws js*d  * 8
            
        if len(input.shape) == 4:
            input = input.reshape([input.shape[0],-1,input.shape[-1]])

        # print("encoder,layer before input",input.shape)
        # padding the one zero row to global position, so each joint including global position has 4 channels as input
        if self.args.rotation == 'quaternion' and self.args.pos_repr != '4d':
            input = torch.cat((input, torch.zeros_like(input[:, [0], :])), dim=1)
        for i, layer in enumerate(self.layers):
            if self.args.skeleton_info == 'concat' and offset is not None:
                self.convs[i].set_offset(offset[i])
            input = layer(input)
            # print("encoder,layer, output",i,input.shape)
        return input


class Decoder(nn.Module):
    def __init__(self, args, enc: Encoder):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()
        self.unpools = nn.ModuleList()
        self.args = args
        self.enc = enc
        self.convs = []
        self.use_skeleton_linear = self.enc.use_skeleton_linear

        kernel_size = args.kernel_size
        padding = (kernel_size - 1) // 2
        
        if args.skeleton_info == 'concat': add_offset = True
        else: add_offset = False

        for i in range(args.num_layers):
            seq = []
            in_channels = enc.channel_list[args.num_layers - i]
            out_channels = in_channels // self.args.scale_factor
            neighbor_list = find_neighbor(enc.topologies[args.num_layers - i - 1], args.skeleton_dist)

            if i != 0 and i != args.num_layers - 1:
                bias = False
            else:
                bias = True

            self.unpools.append(SkeletonUnpool(enc.pooling_list[args.num_layers - i - 1], in_channels // len(neighbor_list)))

            seq.append(nn.Upsample(scale_factor=2, mode=args.upsampling, align_corners=False))
            seq.append(self.unpools[-1])
            for _ in range(args.extra_conv):
                if self.use_skeleton_linear:
                    seq.append(SkeletonLinear(neighbor_list, in_channels=in_channels, out_channels=in_channels))
                else:
                    seq.append(SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=in_channels,
                                        joint_num=enc.edge_num[args.num_layers - i - 1], kernel_size=kernel_size,
                                        stride=1,
                                        padding=padding, padding_mode=args.padding_mode, bias=bias))
            if self.use_skeleton_linear:
                seq.append(SkeletonLinear(neighbor_list, in_channels=in_channels, out_channels=out_channels))
                print("use linear layer, in_channels, out_channels",in_channels,out_channels)
            else:
                seq.append(SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=out_channels,
                                        joint_num=enc.edge_num[args.num_layers - i - 1], kernel_size=kernel_size, stride=1,
                                        padding=padding, padding_mode=args.padding_mode, bias=bias, add_offset=add_offset,
                                        in_offset_channel=3 * enc.channel_base[args.num_layers - i - 1] // enc.channel_base[0]))
                # print("use decoder conv layer, in_channels, out_channels",in_channels,out_channels)
            self.convs.append(seq[-1])
            if i != args.num_layers - 1: seq.append(nn.LeakyReLU(negative_slope=0.2))

            self.layers.append(nn.Sequential(*seq))
            self.rotation = enc.rotation

    def forward(self, input, offset=None,input_type=None,root_rotation=None,root_translation=None,window_size=None,bs=None):
        for i, layer in enumerate(self.layers):
            
            if self.args.skeleton_info == 'concat':
                self.convs[i].set_offset(offset[len(self.layers) - i - 1])
            input = layer(input)

        if self.args.last_sigmoid:
            if self.args.velocity_virtual_node:
                # apply relu to 
                input[:,0,:] = torch.tanh(input[:,0,:])
            else:
                input = torch.tanh(input)
                
        # throw the padded rwo for global position
        if self.args.rotation == 'quaternion' and self.args.pos_repr != '4d':
            input = input[:, :-1, :]
        # import pdb; pdb.set_trace()
        if not (self.use_skeleton_linear) and self.args.window_size<0:
            input = input[:,:,:].mean(dim=-1)
        if self.rotation=="quaternion":d = 4
        elif self.rotation=="matrix":d = 9
        elif self.rotation=="rotation_6d":d = 6
        if window_size is None:
            window_size = self.args.window_size
        if self.enc.pose_wise:
            input = input.mean(dim=-1)
            input = input.reshape([input.shape[0],-1,d]).reshape([bs,window_size,-1,d]).permute(0,2,3,1)
        else:
            if self.args.window_size>0:
                input = input.reshape([input.shape[0],-1,d,window_size])
            else:
                input = input.reshape([input.shape[0],-1,d])
        output_rotation = format_rotation(input, self.args.rotation_data)
        
        if self.args.with_root:
            #for skip connection
            root_rotation = format_rotation(root_rotation, self.args.rotation_data)
            output_rotation[:,0,:,:] = output_rotation[:,0,:]*1e-1 + root_rotation #0.1  for some eval 0.01 for residual
        if self.args.velocity_virtual_node:
            output_rotation[:,-1,:,:] = output_rotation[:,-1,:]*1e-1  + root_translation #1 for some eval 0.01 for residaul
        return output_rotation


class AE(nn.Module):
    def __init__(self, args, topology):
        super(AE, self).__init__()
        self.enc = Encoder(args, topology)
        self.dec = Decoder(args, self.enc)

    def forward(self, input, offset=None):
        latent = self.enc(input, offset)
        result = self.dec(latent, offset)
        return latent, result


# eoncoder for static part, i.e. offset part
class StaticEncoder(nn.Module):
    def __init__(self, args, edges):
        super(StaticEncoder, self).__init__()
        self.args = args
        self.layers = nn.ModuleList()
        activation = nn.LeakyReLU(negative_slope=0.2)
        channels = 3

        for i in range(args.num_layers):
            neighbor_list = find_neighbor(edges, args.skeleton_dist)
            
            seq = []
            seq.append(SkeletonLinear(neighbor_list, in_channels=channels * len(neighbor_list),
                                      out_channels=channels * 2 * len(neighbor_list), extra_dim1=True))
            if i < args.num_layers - 1:
                pool = SkeletonPool(edges, channels_per_edge=channels*2, pooling_mode='mean')
                seq.append(pool)
                edges = pool.new_edges
            seq.append(activation)
            channels *= 2
            self.layers.append(nn.Sequential(*seq))

    # input should have shape B * E * 3
    def forward(self, input: torch.Tensor):
        output = [input]
        for i, layer in enumerate(self.layers):
            input = layer(input)
            output.append(input.squeeze(-1))
        return output
