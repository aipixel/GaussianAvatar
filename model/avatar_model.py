import torch
import numpy as np
import torch
import os
import numpy as np
import torch.nn as nn
from submodules import smplx
import trimesh
from scene.dataset_mono import MonoDataset_train, MonoDataset_test, MonoDataset_novel_pose, MonoDataset_novel_view
from utils.general_utils import worker_init_fn
from utils.system_utils import mkdir_p
from model.network import POP_no_unet
from utils.general_utils import load_masks
from gaussian_renderer import render_batch
from os.path import join
import torch.nn as nn
from model.modules  import UnetNoCond5DS

class AvatarModel:
    def __init__(self, model_parms, net_parms, opt_parms, load_iteration=None, train=True):

        self.model_parms = model_parms
        self.net_parms = net_parms
        self.opt_parms = opt_parms
        self.model_path = model_parms.model_path
        self.loaded_iter = None
        self.train = train
        self.train_mode = model_parms.train_mode
        self.gender = self.model_parms.smpl_gender

        if train:
            self.batch_size = self.model_parms.batch_size
        else:
            self.batch_size = 1

        if train:
            split = 'train'
        else:
            split = 'test'

        self.train_dataset  = MonoDataset_train(model_parms)
        self.smpl_data = self.train_dataset.smpl_data

        # partial code derive from POP (https://github.com/qianlim/POP)
        assert model_parms.smpl_type in ['smplx', 'smpl']
        if model_parms.smpl_type == 'smplx':
            self.smpl_model = smplx.SMPLX(model_path=self.model_parms.smplx_model_path, gender = self.gender, use_pca = False, num_pca_comps = 45, flat_hand_mean = True, batch_size = self.batch_size).cuda().eval()
            flist_uv, valid_idx, uv_coord_map = load_masks(model_parms.project_path, self.model_parms.query_posmap_size, body_model='smplx')
            query_map_path = join(model_parms.source_path, split, 'query_posemap_{}_cano_smplx.npz'.format(self.model_parms.query_posmap_size))
            inp_map_path = join(model_parms.source_path, split, 'query_posemap_{}_cano_smplx.npz'.format(self.model_parms.inp_posmap_size))

            query_lbs_path =join(model_parms.project_path, 'assets', 'lbs_map_smplx_{}.npy'.format(self.model_parms.query_posmap_size))
            mat_path = join(model_parms.source_path, split, 'smplx_cano_joint_mat.pth')
            joint_num = 55
        
        else:
            self.smpl_model = smplx.SMPL(model_path=self.model_parms.smpl_model_path, gender = self.gender, batch_size = self.batch_size).cuda().eval()
            flist_uv, valid_idx, uv_coord_map = load_masks(model_parms.project_path, self.model_parms.query_posmap_size, body_model='smpl')

            query_map_path = join(model_parms.source_path, split, 'query_posemap_{}_cano_smpl.npz'.format(self.model_parms.query_posmap_size))
            inp_map_path = join(model_parms.source_path, split, 'query_posemap_{}_cano_smpl.npz'.format(self.model_parms.inp_posmap_size))

            query_lbs_path =join(model_parms.project_path, 'assets', 'lbs_map_smpl_{}.npy'.format(self.model_parms.query_posmap_size))
            mat_path = join(model_parms.source_path,  split, 'smpl_cano_joint_mat.pth')
            joint_num = 24

        self.uv_coord_map = uv_coord_map
        self.valid_idx = valid_idx

        if model_parms.fixed_inp:
            fix_inp_map = torch.from_numpy(np.load(inp_map_path)['posmap' + str(self.model_parms.inp_posmap_size)].transpose(2,0,1)).cuda()
            self.fix_inp_map = fix_inp_map[None].expand(self.batch_size, -1, -1, -1)
        
        ## query_map store the sampled points from the cannonical smpl mesh, shape as [512. 512, 3] 
        query_map = torch.from_numpy(np.load(query_map_path)['posmap' + str(self.model_parms.query_posmap_size)]).reshape(-1,3)
        query_points = query_map[valid_idx, :].cuda().contiguous()
        self.query_points = query_points[None].expand(self.batch_size, -1, -1)
        
        # we fix the opacity and rots of 3d gs as described in paper 
        self.fix_opacity = torch.ones((self.query_points.shape[1], 1)).cuda()
        rots = torch.zeros((self.query_points.shape[1], 4), device="cuda")
        rots[:, 0] = 1
        self.fix_rotation = rots
        
        # we save the skinning weights from the cannonical mesh
        query_lbs = torch.from_numpy(np.load(query_lbs_path)).reshape(self.model_parms.query_posmap_size*self.model_parms.query_posmap_size, joint_num)
        self.query_lbs = query_lbs[valid_idx, :][None].expand(self.batch_size, -1, -1).cuda().contiguous()
        
        self.inv_mats = torch.linalg.inv(torch.load(mat_path)).expand(self.batch_size, -1, -1, -1).cuda()
        print('inv_mat shape: ', self.inv_mats.shape)

        num_training_frames = len(self.train_dataset)
        param = []

        if not torch.is_tensor(self.smpl_data['beta']):
            self.betas = torch.from_numpy(self.smpl_data['beta'][0])[None].expand(self.batch_size, -1).cuda()
        else:
            self.betas = self.smpl_data['beta'][0][None].expand(self.batch_size, -1).cuda()

        if model_parms.smpl_type == 'smplx':
            self.pose = torch.nn.Embedding(num_training_frames, 66, _weight=self.train_dataset.pose_data, sparse=True).cuda()
            param += list(self.pose.parameters())

            self.transl = torch.nn.Embedding(num_training_frames, 3, _weight=self.train_dataset.transl_data, sparse=True).cuda()
            param += list(self.transl.parameters())
        else:
            self.pose = torch.nn.Embedding(num_training_frames, 72, _weight=self.train_dataset.pose_data, sparse=True).cuda()
            param += list(self.pose.parameters())

            self.transl = torch.nn.Embedding(num_training_frames, 3, _weight=self.train_dataset.transl_data, sparse=True).cuda()
            param += list(self.transl.parameters())
        
        self.optimizer_pose = torch.optim.SparseAdam(param, 5.0e-3)
        
        bg_color = [1, 1, 1] if model_parms.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.rotation_activation = torch.nn.functional.normalize
        self.sigmoid_activation =  nn.Sigmoid()
        
        self.net_set(self.model_parms.train_stage)

    def net_set(self, mode):
        assert mode in [0, 1, 2]

        self.net = POP_no_unet(
            c_geom=self.net_parms.c_geom, # channels of the geometric features
            geom_layer_type=self.net_parms.geom_layer_type, # the type of architecture used for smoothing the geometric feature tensor
            nf=self.net_parms.nf, # num filters for the unet
            hsize=self.net_parms.hsize, # hidden layer size of the ShapeDecoder MLP
            up_mode=self.net_parms.up_mode,# upconv or upsample for the upsampling layers in the pose feature UNet
            use_dropout=bool(self.net_parms.use_dropout), # whether use dropout in the pose feature UNet
            uv_feat_dim=2, # input dimension of the uv coordinates
        ).cuda()
            
        geo_feature = torch.ones(1, self.net_parms.c_geom, self.model_parms.inp_posmap_size, self.model_parms.inp_posmap_size).normal_(mean=0., std=0.01).float().cuda()
        self.geo_feature = nn.Parameter(geo_feature.requires_grad_(True))
        
        if self.model_parms.train_stage == 2:
            self.pose_encoder = UnetNoCond5DS(
                input_nc=3,
                output_nc=self.net_parms.c_pose,
                nf=self.net_parms.nf,
                up_mode=self.net_parms.up_mode,
                use_dropout=False,
            ).cuda()

    def training_setup(self):
        if self.model_parms.train_stage  ==1:
            self.optimizer = torch.optim.Adam(
            [
                {"params": self.net.parameters(), "lr": self.opt_parms.lr_net},
                {"params": self.geo_feature, "lr": self.opt_parms.lr_geomfeat}
            ])
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.opt_parms.sched_milestones, gamma=0.1)
        else:
            self.optimizer = torch.optim.Adam(
            [   
                {"params": self.net.parameters(), "lr": self.opt_parms.lr_net * 0.1},
                {"params": self.pose_encoder.parameters(), "lr": self.opt_parms.lr_net},
            ])
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.opt_parms.sched_milestones, gamma=0.1)
    def save(self, iteration):
        net_save_path = os.path.join(self.model_path, "net/iteration_{}".format(iteration))
        mkdir_p(net_save_path)
        if self.model_parms.train_stage  == 1:
            torch.save(
                {
                "net": self.net.state_dict(),
                "geo_feature": self.geo_feature,
                "pose": self.pose.state_dict(),
                "transl": self.transl.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict()}, 
            os.path.join(net_save_path,  "net.pth"))
        else:
            torch.save(
                {
                "pose_encoder": self.pose_encoder.state_dict(),
                "geo_feature": self.geo_feature,
                "pose": self.pose.state_dict(),
                "transl": self.transl.state_dict(),
                "net": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict()}, 
            os.path.join(net_save_path,  "pose_encoder.pth"))

    def load(self, iteration, test=False):

        net_save_path = os.path.join(self.model_path, "net/iteration_{}".format(iteration))

        saved_model_state = torch.load(
            os.path.join(net_save_path, "net.pth"))
        print('load pth: ', os.path.join(net_save_path, "net.pth"))
        self.net.load_state_dict(saved_model_state["net"], strict=False)

        if self.model_parms.train_stage  ==1:
            if not test:
                self.pose.load_state_dict(saved_model_state["pose"], strict=False)
                self.transl.load_state_dict(saved_model_state["transl"], strict=False)
            # if self.train_mode == 0:
            self.geo_feature.data[...] = saved_model_state["geo_feature"].data[...]

        if self.optimizer is not None:
            self.optimizer.load_state_dict(saved_model_state["optimizer"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(saved_model_state["scheduler"])


    def stage_load(self, ckpt_path):

        net_save_path = ckpt_path
        print('load pth: ', os.path.join(net_save_path, "net.pth"))
        saved_model_state = torch.load(
            os.path.join(net_save_path, "net.pth"))
        
        self.net.load_state_dict(saved_model_state["net"], strict=False)
        self.pose.load_state_dict(saved_model_state["pose"], strict=False)
        self.transl.load_state_dict(saved_model_state["transl"], strict=False)
        # if self.train_mode == 0:
        self.geo_feature.data[...] = saved_model_state["geo_feature"].data[...]

    def stage2_load(self, epoch):
    
        pose_encoder_path = os.path.join(self.model_parms.project_path, self.model_path, "net/iteration_{}".format(epoch))

        pose_encoder_state = torch.load(
            os.path.join(pose_encoder_path, "pose_encoder.pth"))
        print('load pth: ', os.path.join(pose_encoder_path, "pose_encoder.pth"))

        self.net.load_state_dict(pose_encoder_state["net"], strict=False)
        self.pose.load_state_dict(pose_encoder_state["pose"], strict=False)
        self.transl.load_state_dict(pose_encoder_state["transl"], strict=False)
        # if self.train_mode == 0:
        self.geo_feature.data[...] = pose_encoder_state["geo_feature"].data[...]
        self.pose_encoder.load_state_dict(pose_encoder_state["pose_encoder"], strict=False)

    def getTrainDataloader(self,):
        return torch.utils.data.DataLoader(self.train_dataset,
                                            batch_size = self.batch_size,
                                            shuffle = True,
                                            num_workers = 4,
                                            worker_init_fn = worker_init_fn,
                                            drop_last = True)

    def getTestDataset(self,):
        self.test_dataset = MonoDataset_test(self.model_parms)
        return self.test_dataset
    
    def getNovelposeDataset(self,):
        self.novel_pose_dataset = MonoDataset_novel_pose(self.model_parms)
        return self.novel_pose_dataset

    def getNovelviewDataset(self,):
        self.novel_view_dataset = MonoDataset_novel_view(self.model_parms)
        return self.novel_view_dataset

    def zero_grad(self, epoch):
        self.optimizer.zero_grad()

        if self.model_parms.train_stage  ==1:
            if epoch > self.opt_parms.pose_op_start_iter:
                self.optimizer_pose.zero_grad()
    def step(self, epoch):

        self.optimizer.step()
        self.scheduler.step()
        if self.model_parms.train_stage  ==1:
            if epoch > self.opt_parms.pose_op_start_iter:
                self.optimizer_pose.step()
            
    def train_stage1(self, batch_data, iteration):
        
        rendered_images = []
        idx = batch_data['pose_idx']

        pose_batch = self.pose(idx)
        transl_batch = self.transl(idx)
        if self.model_parms.smpl_type == 'smplx':
            rest_pose = batch_data['rest_pose']
            live_smpl = self.smpl_model.forward(betas = self.betas,
                                                global_orient = pose_batch[:, :3],
                                                transl = transl_batch,
                                                body_pose = pose_batch[:, 3:66],
                                                jaw_pose = rest_pose[:, :3],
                                                leye_pose=rest_pose[:, 3:6],
                                                reye_pose=rest_pose[:, 6:9],
                                                left_hand_pose= rest_pose[:, 9:54],
                                                right_hand_pose= rest_pose[:, 54:])
        else:
            live_smpl = self.smpl_model.forward(betas=self.betas,
                                global_orient=pose_batch[:, :3],
                                transl = transl_batch,
                                body_pose=pose_batch[:, 3:])
        
        cano2live_jnt_mats = torch.matmul(live_smpl.A, self.inv_mats)

        geom_featmap = self.geo_feature.expand(self.batch_size, -1, -1, -1).contiguous()
        uv_coord_map = self.uv_coord_map.expand(self.batch_size, -1, -1).contiguous()


        pred_res,pred_scales, pred_shs, = self.net.forward(pose_featmap=None,
                                                    geom_featmap=geom_featmap,
                                                    uv_loc=uv_coord_map)

        
        
        pred_res = pred_res.permute([0,2,1]) * 0.02  #(B, H, W ,3)
        pred_point_res = pred_res[:, self.valid_idx, ...].contiguous()

        cano_deform_point = pred_point_res + self.query_points

        pt_mats = torch.einsum('bnj,bjxy->bnxy', self.query_lbs, cano2live_jnt_mats)
        full_pred = torch.einsum('bnxy,bny->bnx', pt_mats[..., :3, :3], cano_deform_point) + pt_mats[..., :3, 3]

        if iteration < 1000:
            pred_scales = pred_scales.permute([0,2,1]) * 1e-3 * iteration 
        else:
            pred_scales = pred_scales.permute([0,2,1])

        pred_shs = pred_shs.permute([0,2,1])

        pred_scales = pred_scales[:, self.valid_idx, ...].contiguous()
        pred_scales = pred_scales.repeat(1,1,3)

        pred_shs = pred_shs[:, self.valid_idx, ...].contiguous()

        offset_loss = torch.mean(pred_res ** 2)
        geo_loss = torch.mean(self.geo_feature**2)
        scale_loss = torch.mean(pred_scales)

        for batch_index in range(self.batch_size):
            FovX = batch_data['FovX'][batch_index]
            FovY = batch_data['FovY'][batch_index]
            height = batch_data['height'][batch_index]
            width = batch_data['width'][batch_index]
            world_view_transform = batch_data['world_view_transform'][batch_index]
            full_proj_transform = batch_data['full_proj_transform'][batch_index]
            camera_center = batch_data['camera_center'][batch_index]
        

            points = full_pred[batch_index]

            colors = pred_shs[batch_index]
            scales = pred_scales[batch_index]

            rendered_images.append(
                render_batch(
                    points=points,
                    shs=None,
                    colors_precomp=colors,
                    rotations=self.fix_rotation,
                    scales=scales,
                    opacity=self.fix_opacity,
                    FovX=FovX,
                    FovY=FovY,
                    height=height,
                    width=width,
                    bg_color=self.background,
                    world_view_transform=world_view_transform,
                    full_proj_transform=full_proj_transform,
                    active_sh_degree=0,
                    camera_center=camera_center
                )
            )

        return torch.stack(rendered_images, dim=0), full_pred, offset_loss, geo_loss, scale_loss

    def train_stage2(self, batch_data, iteration):
        
        rendered_images = []
        inp_posmap = batch_data['inp_pos_map']
        idx = batch_data['pose_idx']

        pose_batch = self.pose(idx)
        transl_batch = self.transl(idx)

        if self.model_parms.smpl_type == 'smplx':
            rest_pose = batch_data['rest_pose']
            live_smpl = self.smpl_model.forward(betas = self.betas,
                                                global_orient = pose_batch[:, :3],
                                                transl = transl_batch,
                                                body_pose = pose_batch[:, 3:66],
                                                jaw_pose = rest_pose[:, :3],
                                                leye_pose=rest_pose[:, 3:6],
                                                reye_pose=rest_pose[:, 6:9],
                                                left_hand_pose= rest_pose[:, 9:54],
                                                right_hand_pose= rest_pose[:, 54:])
        else:
            live_smpl = self.smpl_model.forward(betas=self.betas,
                                global_orient=pose_batch[:, :3],
                                transl = transl_batch,
                                body_pose=pose_batch[:, 3:])
        
        cano2live_jnt_mats = torch.matmul(live_smpl.A, self.inv_mats)

        uv_coord_map = self.uv_coord_map.expand(self.batch_size, -1, -1).contiguous()

        geom_featmap = self.geo_feature.expand(self.batch_size, -1, -1, -1).contiguous()

        pose_featmap = self.pose_encoder(inp_posmap)

        pred_res,pred_scales, pred_shs, = self.net.forward(pose_featmap=pose_featmap,
                                                    geom_featmap=geom_featmap,
                                                    uv_loc=uv_coord_map)

        
        
        pred_res = pred_res.permute([0,2,1]) * 0.02  #(B, H, W ,3)
        pred_point_res = pred_res[:, self.valid_idx, ...].contiguous()

        cano_deform_point = pred_point_res + self.query_points
        pt_mats = torch.einsum('bnj,bjxy->bnxy', self.query_lbs, cano2live_jnt_mats)
        full_pred = torch.einsum('bnxy,bny->bnx', pt_mats[..., :3, :3], cano_deform_point) + pt_mats[..., :3, 3]

        pred_scales = pred_scales.permute([0,2,1])

        pred_shs = pred_shs.permute([0,2,1])

        pred_scales = pred_scales[:, self.valid_idx, ...].contiguous()
        pred_scales = pred_scales.repeat(1,1,3)

        pred_shs = pred_shs[:, self.valid_idx, ...].contiguous()

        offset_loss = torch.mean(pred_res ** 2)
        pose_loss = torch.mean(pose_featmap ** 2)
        scale_loss = torch.mean(pred_scales)

        for batch_index in range(self.batch_size):
            FovX = batch_data['FovX'][batch_index]
            FovY = batch_data['FovY'][batch_index]
            height = batch_data['height'][batch_index]
            width = batch_data['width'][batch_index]
            world_view_transform = batch_data['world_view_transform'][batch_index]
            full_proj_transform = batch_data['full_proj_transform'][batch_index]
            camera_center = batch_data['camera_center'][batch_index]
        

            points = full_pred[batch_index]
            colors = pred_shs[batch_index]
            scales = pred_scales[batch_index]
        
            rendered_images.append(
                render_batch(
                    points=points,
                    shs=None,
                    colors_precomp=colors,
                    rotations=self.fix_rotation,
                    scales=scales,
                    opacity=self.fix_opacity,
                    FovX=FovX,
                    FovY=FovY,
                    height=height,
                    width=width,
                    bg_color=self.background,
                    world_view_transform=world_view_transform,
                    full_proj_transform=full_proj_transform,
                    active_sh_degree=0,
                    camera_center=camera_center
                )
            )

        return torch.stack(rendered_images, dim=0), full_pred, pose_loss, offset_loss,
        


    def render_free_stage1(self, batch_data, iteration):
        
        rendered_images = []
        pose_data = batch_data['pose_data']
        transl_data = batch_data['transl_data']

        if self.model_parms.smpl_type == 'smplx':
            rest_pose = batch_data['rest_pose']
            live_smpl = self.smpl_model.forward(betas = self.betas,
                                                global_orient = pose_data[:, :3],
                                                transl = transl_data,
                                                body_pose = pose_data[:, 3:66],
                                                jaw_pose = rest_pose[:, :3],
                                                leye_pose=rest_pose[:, 3:6],
                                                reye_pose=rest_pose[:, 6:9],
                                                left_hand_pose= rest_pose[:, 9:54],
                                                right_hand_pose= rest_pose[:, 54:])
        else:
            live_smpl = self.smpl_model.forward(betas=self.betas,
                                global_orient=pose_data[:, :3],
                                transl = transl_data,
                                body_pose=pose_data[:, 3:])
        
        cano2live_jnt_mats = torch.matmul(live_smpl.A, self.inv_mats)
        geom_featmap = self.geo_feature.expand(self.batch_size, -1, -1, -1).contiguous()
        uv_coord_map = self.uv_coord_map.expand(self.batch_size, -1, -1).contiguous()


        pred_res,pred_scales, pred_shs, = self.net.forward(pose_featmap=None,
                                                    geom_featmap=geom_featmap,
                                                    uv_loc=uv_coord_map)
        
        pred_res = pred_res.permute([0,2,1]) * 0.02  #(B, H, W ,3)
        pred_point_res = pred_res[:, self.valid_idx, ...].contiguous()

        cano_deform_point = pred_point_res + self.query_points 

        pt_mats = torch.einsum('bnj,bjxy->bnxy', self.query_lbs, cano2live_jnt_mats)
        full_pred = torch.einsum('bnxy,bny->bnx', pt_mats[..., :3, :3], cano_deform_point) + pt_mats[..., :3, 3]

        if iteration < 1000:
            pred_scales = pred_scales.permute([0,2,1]) * 1e-3 * iteration 
        else:
            pred_scales = pred_scales.permute([0,2,1])

        pred_shs = pred_shs.permute([0,2,1])

        pred_scales = pred_scales[:, self.valid_idx, ...].contiguous()
        pred_scales = pred_scales.repeat(1,1,3)

        pred_shs = pred_shs[:, self.valid_idx, ...].contiguous()

        for batch_index in range(self.batch_size):
            FovX = batch_data['FovX'][batch_index]
            FovY = batch_data['FovY'][batch_index]
            height = batch_data['height'][batch_index]
            width = batch_data['width'][batch_index]
            world_view_transform = batch_data['world_view_transform'][batch_index]
            full_proj_transform = batch_data['full_proj_transform'][batch_index]
            camera_center = batch_data['camera_center'][batch_index]
        

            points = full_pred[batch_index]

            colors = pred_shs[batch_index]
            scales = pred_scales[batch_index] 
        
            rendered_images.append(
                render_batch(
                    points=points,
                    shs=None,
                    colors_precomp=colors,
                    rotations=self.fix_rotation,
                    scales=scales,
                    opacity=self.fix_opacity,
                    FovX=FovX,
                    FovY=FovY,
                    height=height,
                    width=width,
                    bg_color=self.background,
                    world_view_transform=world_view_transform,
                    full_proj_transform=full_proj_transform,
                    active_sh_degree=0,
                    camera_center=camera_center
                )
            )

        return torch.stack(rendered_images, dim=0)


    def render_free_stage2(self, batch_data, iteration):
        
        rendered_images = []
        inp_posmap = batch_data['inp_pos_map']
        idx = batch_data['pose_idx']

        pose_batch = self.pose(idx)
        transl_batch = self.transl(idx)

        if self.model_parms.smpl_type == 'smplx':
            rest_pose = batch_data['rest_pose']
            live_smpl = self.smpl_model.forward(betas = self.betas,
                                                global_orient = pose_batch[:, :3],
                                                transl = transl_batch,
                                                body_pose = pose_batch[:, 3:66],
                                                jaw_pose = rest_pose[:, :3],
                                                leye_pose=rest_pose[:, 3:6],
                                                reye_pose=rest_pose[:, 6:9],
                                                left_hand_pose= rest_pose[:, 9:54],
                                                right_hand_pose= rest_pose[:, 54:])
        else:
            live_smpl = self.smpl_model.forward(betas=self.betas,
                                global_orient=pose_batch[:, :3],
                                transl = transl_batch,
                                body_pose=pose_batch[:, 3:])
        
        cano2live_jnt_mats = torch.matmul(live_smpl.A, self.inv_mats)

        uv_coord_map = self.uv_coord_map.expand(self.batch_size, -1, -1).contiguous()

        geom_featmap = self.geo_feature.expand(self.batch_size, -1, -1, -1).contiguous()

        pose_featmap = self.pose_encoder(inp_posmap)

        pred_res,pred_scales, pred_shs, = self.net.forward(pose_featmap=pose_featmap,
                                                    geom_featmap=geom_featmap,
                                                    uv_loc=uv_coord_map)

        
        
        pred_res = pred_res.permute([0,2,1]) * 0.02  #(B, H, W ,3)
        pred_point_res = pred_res[:, self.valid_idx, ...].contiguous()

        cano_deform_point = pred_point_res + self.query_points

        pt_mats = torch.einsum('bnj,bjxy->bnxy', self.query_lbs, cano2live_jnt_mats)
        full_pred = torch.einsum('bnxy,bny->bnx', pt_mats[..., :3, :3], cano_deform_point) + pt_mats[..., :3, 3]


        pred_scales = pred_scales.permute([0,2,1])

        pred_shs = pred_shs.permute([0,2,1])

        pred_scales = pred_scales[:, self.valid_idx, ...].contiguous()
        pred_scales = pred_scales.repeat(1,1,3)

        pred_shs = pred_shs[:, self.valid_idx, ...].contiguous()
        # aiap_all_loss = 0
        for batch_index in range(self.batch_size):
            FovX = batch_data['FovX'][batch_index]
            FovY = batch_data['FovY'][batch_index]
            height = batch_data['height'][batch_index]
            width = batch_data['width'][batch_index]
            world_view_transform = batch_data['world_view_transform'][batch_index]
            full_proj_transform = batch_data['full_proj_transform'][batch_index]
            camera_center = batch_data['camera_center'][batch_index]
        

            points = full_pred[batch_index]
            colors = pred_shs[batch_index]
            scales = pred_scales[batch_index]

            rendered_images.append(
                render_batch(
                    points=points,
                    shs=None,
                    colors_precomp=colors,
                    rotations=self.fix_rotation,
                    scales=scales,
                    opacity=self.fix_opacity,
                    FovX=FovX,
                    FovY=FovY,
                    height=height,
                    width=width,
                    bg_color=self.background,
                    world_view_transform=world_view_transform,
                    full_proj_transform=full_proj_transform,
                    active_sh_degree=0,
                    camera_center=camera_center
                )
            )

        return torch.stack(rendered_images, dim=0)
