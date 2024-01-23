import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules import  uv_to_grid
from model.modules  import UnetNoCond5DS, GeomConvLayers, GeomConvBottleneckLayers, ShapeDecoder


class POP_no_unet(nn.Module):
    def __init__(
                self, 
                c_geom=64, # channels of the geometric features
                geom_layer_type='conv', # the type of architecture used for smoothing the geometric feature tensor
                nf=64, # num filters for the unet
                hsize=256, # hidden layer size of the ShapeDecoder MLP
                up_mode='upconv', # upconv or upsample for the upsampling layers in the pose feature UNet
                use_dropout=False, # whether use dropout in the pose feature UNet
                uv_feat_dim=2, # input dimension of the uv coordinates
                ):

        super().__init__()
        self.geom_layer_type = geom_layer_type 
        
        geom_proc_layers = {
            'unet': UnetNoCond5DS(c_geom, c_geom, nf, up_mode, use_dropout), # use a unet
            'conv': GeomConvLayers(c_geom, c_geom, c_geom, use_relu=False), # use 3 trainable conv layers
            'bottleneck': GeomConvBottleneckLayers(c_geom, c_geom, c_geom, use_relu=False), # use 3 trainable conv layers
        }

        # optional layer for spatially smoothing the geometric feature tensor
        if geom_layer_type is not None:
            self.geom_proc_layers = geom_proc_layers[geom_layer_type]
    
        # shared shape decoder across different outfit types
        self.decoder = ShapeDecoder(in_size=uv_feat_dim + c_geom,
                                    hsize=hsize, actv_fn='softplus')

            
    def forward(self, pose_featmap, geom_featmap, uv_loc):
        '''
        :param x: input posmap, [batch, 3, 256, 256]
        :param geom_featmap: a [B, C, H, W] tensor, spatially pixel-aligned with the pose features extracted by the UNet
        :param uv_loc: querying uv coordinates, ranging between 0 and 1, of shape [B, H*W, 2].
        :param pq_coords: the 'sub-UV-pixel' (p,q) coordinates, range [0,1), shape [B, H*W, 1, 2]. 
                        Note: It is the intra-patch coordinates in SCALE. Kept here for the backward compatibility with SCALE.
        :return:
            clothing offset vectors (residuals) and normals of the points
        '''
        # geometric feature tensor
        if self.geom_layer_type is not None:
                geom_featmap = self.geom_proc_layers(geom_featmap)

        if  pose_featmap is None:
            # pose and geom features are concatenated to form the feature for each point

            pix_feature = geom_featmap
        else:
            pix_feature = pose_featmap + geom_featmap

        
        feat_res = geom_featmap.shape[2] # spatial resolution of the input pose and geometric features
        uv_res = int(uv_loc.shape[1]**0.5) # spatial resolution of the query uv map

        # spatially bilinearly upsample the features to match the query resolution
        if feat_res != uv_res:
            query_grid = uv_to_grid(uv_loc, uv_res)
            pix_feature = F.grid_sample(pix_feature, query_grid, mode='bilinear', align_corners=False)#, align_corners=True)

        B, C, H, W = pix_feature.shape
        N_subsample = 1 # inherit the SCALE code custom, but now only sample one point per pixel

        uv_feat_dim = uv_loc.size()[-1]
        uv_loc = uv_loc.expand(N_subsample, -1, -1, -1).permute([1, 2, 0, 3])

        # uv and pix feature is shared for all points in each patch
        pix_feature = pix_feature.view(B, C, -1).expand(N_subsample, -1,-1,-1).permute([1,2,3,0]) # [B, C, N_pix, N_sample_perpix]
        pix_feature = pix_feature.reshape(B, C, -1)

        uv_loc = uv_loc.reshape(B, -1, uv_feat_dim).transpose(1, 2)  # [B, N_pix, N_subsample, 2] --> [B, 2, Num of all pq subpixels]

        residuals, scales, shs = self.decoder(torch.cat([pix_feature, uv_loc], 1))  # [B, 3, N all subpixels]

        return residuals, scales, shs 

