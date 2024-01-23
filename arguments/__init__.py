## partial code from origin 3D GS source
## https://github.com/graphdeco-inria/gaussian-splatting
from argparse import ArgumentParser, Namespace
import sys
import os
import math
import torch
from pytorch3d import transforms
import numpy as np

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

# here we define the cannonical pose for smplx or smpl model

leg_angle = 30
smplx_cpose_param = torch.zeros(1, 165)
smplx_cpose_param[:, 5] =  leg_angle / 180 * math.pi
smplx_cpose_param[:, 8] = -leg_angle / 180 * math.pi
oula_arm_l = transforms.euler_angles_to_matrix(torch.tensor(np.array([[-90, 0, 0]])) / 180 * np.pi, 'XYZ')
axis_arm_l = transforms.matrix_to_axis_angle(oula_arm_l)

smpl_cpose_param = torch.zeros(1, 72)
smpl_cpose_param[:, 5] =  leg_angle / 180 * math.pi
smpl_cpose_param[:, 8] = -leg_angle / 180 * math.pi

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self._source_path = ""
        self._model_path = ""
        self.project_path = os.getcwd()

        # smpl and smplx model path
        self.smpl_model_path = os.getcwd() + '/assets/smpl_files/smpl'
        self.smplx_model_path = os.getcwd() + '/assets/smpl_files/smplx'
        self.test_folder = os.getcwd() + '/assets/test_pose'

        # two stage training, stage one for pose optimization and stage two for adding dynamic appearances
        self.stage1_out_path =  ''
        self.save_epoch = 30
        self.train_stage = 1

        ########## here we need specific change ###############
        self.dataset_type = 'peeplesnapshot'   # dataset_type defined for preprocessing 
        self.smpl_gender = 'neutral'    # neutral, female, male, neutral as defaults
        self.smpl_type = 'smpl'      
        self.no_mask = 0  # our 0 x_avatar 1
        self.fixed_inp = 0
        self.train_mode = 0  #  [0, 1,] pop our
        self.cam_static = 1
        self._white_background = True
        ##########for mode 0##################################

        # for_test
        self.bullet_pose_list = [112, 217, 755]    # set the specific pose for novel view synthesize
        self.batch_size = 2

        # input uv size and query uv size, 
        self.query_posmap_size= 512
        self.inp_posmap_size= 128

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class NetworkParams(ParamGroup):
    def __init__(self, parser):
        
        # pose encoder 
        self.c_pose = 64  
        self.c_geom = 64
        self.hsize = 128
        self.nf = 32
        self.up_mode = 'upconv'
        self.use_dropout = 0
        self.pos_encoding = 0
        self.num_emb_freqs = 6

        self.posemb_incl_input = 0
        self.geom_layer_type = 'conv'
        self.gaussian_kernel_size = 5

        super().__init__(parser, "Network Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.epochs = 200     # total epochs for training 
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = self.epochs
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_scale = 3e-2
        self.lambda_lpips = 0.2
        self.lambda_aiap = 0.1
        self.lambda_color = 3e-2

        self.lambda_pose = 10
        self.lambda_rgl = 1e1
        self.log_iter = 2000
        self.lpips_start_iter = 150
        self.pose_op_start_iter = 1800  #define when to start pose optimization, >epochs means no optimization
        self.lr_net = 3e-3
        self.lr_geomfeat = 5e-4

        self.sched_milestones = [int(self.epochs/ 3), int(self.epochs *2/ 3)]

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)