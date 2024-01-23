import os
import torch
import numpy as np
from torch.utils.data import Dataset
from os.path import join
from PIL import Image
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov
import cv2

def _update_extrinsics(
        extrinsics, 
        angle, 
        trans=None, 
        rotate_axis='y'):
    r""" Uptate camera extrinsics when rotating it around a standard axis.

    Args:
        - extrinsics: Array (3, 3)
        - angle: Float
        - trans: Array (3, )
        - rotate_axis: String

    Returns:
        - Array (3, 3)
    """
    E = extrinsics
    inv_E = np.linalg.inv(E)

    camrot = inv_E[:3, :3]
    campos = inv_E[:3, 3]
    if trans is not None:
        campos -= trans

    rot_y_axis = camrot.T[1, 1]
    if rot_y_axis < 0.:
        angle = -angle
    
    rotate_coord = {
        'x': 0, 'y': 1, 'z':2
    }
    grot_vec = np.array([0., 0., 0.])
    grot_vec[rotate_coord[rotate_axis]] = angle
    grot_mtx = cv2.Rodrigues(grot_vec)[0].astype('float32')

    rot_campos = grot_mtx.dot(campos) 
    rot_camrot = grot_mtx.dot(camrot)
    if trans is not None:
        rot_campos += trans
    
    new_E = np.identity(4)
    new_E[:3, :3] = rot_camrot.T
    new_E[:3, 3] = -rot_camrot.T.dot(rot_campos)

    return new_E

def rotate_camera_by_frame_idx(
        extrinsics, 
        frame_idx, 
        trans=None,
        rotate_axis='y',
        period=196,
        inv_angle=False):
    r""" Get camera extrinsics based on frame index and rotation period.

    Args:
        - extrinsics: Array (3, 3)
        - frame_idx: Integer
        - trans: Array (3, )
        - rotate_axis: String
        - period: Integer
        - inv_angle: Boolean (clockwise/counterclockwise)

    Returns:
        - Array (3, 3)
    """

    angle = 2 * np.pi * (frame_idx / period)
    if inv_angle:
        angle = -angle
    return _update_extrinsics(
                extrinsics, angle, trans, rotate_axis)

# we reguire the data path as follows:
# data_path
#   train
#       -images 
#       -masks 
#       -cam_parms 
#       -smpl_parms.npy
#   test
#       -images
#       -masks 
#       -cam_parms 
#       -smpl_parms.npy
#each have the sanme name 
#and the smpl_parms like {beta:N 10;  trans,N 3; body_pose: N 165 or 72} 

class MonoDataset_train(Dataset):
    @torch.no_grad()
    def __init__(self, dataset_parms,
                 device = torch.device('cuda:0')):
        super(MonoDataset_train, self).__init__()

        self.dataset_parms = dataset_parms

        self.data_folder = join(dataset_parms.source_path, 'train')
        self.device = device
        self.gender = self.dataset_parms.smpl_gender

        self.zfar = 100.0
        self.znear = 0.01
        self.trans = np.array([0.0, 0.0, 0.0]),
        self.scale = 1.0

        self.no_mask = bool(self.dataset_parms.no_mask)

        # if dataset_parms.train_stage == 1:
        #     print('loading smpl data ', join(self.data_folder, 'smpl_parms.pth'))
        #     self.smpl_data = torch.load(join(self.data_folder, 'smpl_parms.pth'))
        # else:
        #     print('loading smpl data ', join(self.data_folder, 'smpl_parms_pred.pth'))
        #     self.smpl_data = torch.load(join(self.data_folder, 'smpl_parms_pred.pth'))
        print('loading smpl data ', join(self.data_folder, 'smpl_parms.pth'))
        self.smpl_data = torch.load(join(self.data_folder, 'smpl_parms.pth'))

        self.data_length = len(os.listdir(join(self.data_folder, 'images')))
        
        self.name_list = []
        for index, img in enumerate(sorted(os.listdir(join(self.data_folder, 'images')))):
            base_name = img.split('.')[0]
            self.name_list.append((index, base_name))
        
        self.image_fix = os.listdir(join(self.data_folder, 'images'))[0].split('.')[-1]
        
        if not dataset_parms.no_mask:
            self.mask_fix = os.listdir(join(self.data_folder, 'masks'))[0].split('.')[-1]

        print("total pose length", self.data_length )


        if dataset_parms.smpl_type == 'smplx':

            self.pose_data = self.smpl_data['body_pose'][:self.data_length, :66]

            self.transl_data = self.smpl_data['trans'][:self.data_length,:]
            self.rest_pose_data = self.smpl_data['body_pose'][:self.data_length, 66:]
            if not torch.is_tensor(self.smpl_data['body_pose']):
                self.pose_data = torch.from_numpy(self.pose_data)
            if not torch.is_tensor(self.smpl_data['trans']):
                self.transl_data = torch.from_numpy(self.transl_data)
        else:
            self.pose_data = self.smpl_data['body_pose'][:self.data_length]
            self.transl_data = self.smpl_data['trans'][:self.data_length,:]
            # self.rest_pose_data = self.smpl_data['body_pose'][:self.data_length, 66:]
            if not torch.is_tensor(self.smpl_data['body_pose']):
                self.pose_data = torch.from_numpy(self.pose_data)
            if not torch.is_tensor(self.smpl_data['trans']):
                self.transl_data = torch.from_numpy(self.transl_data)

        if dataset_parms.cam_static:
            cam_path = join(self.data_folder, 'cam_parms.npz')
            cam_npy = np.load(cam_path)
            extr_npy = cam_npy['extrinsic']
            intr_npy = cam_npy['intrinsic']
            self.R = np.array(extr_npy[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
            self.T = np.array([extr_npy[:3, 3]], np.float32)
            self.intrinsic = np.array(intr_npy, np.float32).reshape(3, 3)
        
    def __len__(self):
        return self.data_length

    def __getitem__(self, index, ignore_list=None):
        return self.getitem(index, ignore_list)

    @torch.no_grad()
    def getitem(self, index, ignore_list):
        pose_idx, name_idx = self.name_list[index]

        image_path = join(self.data_folder, 'images' ,name_idx + '.' + self.image_fix)
        
        cam_path = join(self.data_folder, 'cam_parms', name_idx + '.npz')

        if not self.dataset_parms.no_mask:
            mask_path = join(self.data_folder, 'masks', name_idx + '.' + self.mask_fix)

        if self.dataset_parms.train_stage == 2:

            inp_posmap_path = self.data_folder + '/inp_map/' +'inp_posemap_%s_%s.npz'% (str(self.dataset_parms.inp_posmap_size), str(pose_idx).zfill(8))
            inp_posmap = np.load(inp_posmap_path)['posmap' + str(self.dataset_parms.inp_posmap_size)]

        if not self.dataset_parms.cam_static:
            cam_npy = np.load(cam_path)
            extr_npy = cam_npy['extrinsic']
            intr_npy = cam_npy['intrinsic']
            R = np.array(extr_npy[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
            T = np.array(extr_npy[:3, 3], np.float32)
            intrinsic = np.array(intr_npy, np.float32).reshape(3, 3)
        
        else:
            R = self.R
            T = self.T
            intrinsic = self.intrinsic

        focal_length_x = intrinsic[0, 0]
        focal_length_y = intrinsic[1, 1]

        image = Image.open(image_path)
        width, height = image.size

        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)

        if not self.dataset_parms.no_mask:
            mask = np.array(Image.open(mask_path))

            if len(mask.shape) <3:
                mask = mask[...,None]

            mask[mask < 128] = 0
            mask[mask >= 128] = 1
            color_img = image * mask + (1 - mask) * 255
            image = Image.fromarray(np.array(color_img, dtype=np.byte), "RGB")

        
        data_item = dict()
        if self.dataset_parms.train_stage == 2:
            data_item['inp_pos_map'] = inp_posmap.transpose(2,0,1)


        resized_image = torch.from_numpy(np.array(image)) / 255.0
        if len(resized_image.shape) == 3:
            resized_image =  resized_image.permute(2, 0, 1)
        else:
            resized_image =  resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

        original_image = resized_image.clamp(0.0, 1.0)

        data_item['original_image'] = original_image
        data_item['FovX'] = FovX
        data_item['FovY'] = FovY
        data_item['width'] = width
        data_item['height'] = height
        data_item['pose_idx'] = pose_idx
        if self.dataset_parms.smpl_type == 'smplx':
            rest_pose = self.rest_pose_data[pose_idx]
            data_item['rest_pose'] = rest_pose
  
        world_view_transform = torch.tensor(getWorld2View2(R, T, self.trans, self.scale)).transpose(0, 1)
        projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=FovX, fovY=FovY, K=intrinsic, h=height, w=width).transpose(0,1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]
        data_item['world_view_transform'] = world_view_transform
        data_item['projection_matrix'] = projection_matrix
        data_item['full_proj_transform'] = full_proj_transform
        data_item['camera_center'] = camera_center
        
        return data_item

class MonoDataset_test(Dataset):
    @torch.no_grad()
    def __init__(self, dataset_parms,
                 device = torch.device('cuda:0')):
        super(MonoDataset_test, self).__init__()

        self.dataset_parms = dataset_parms

        self.data_folder = join(dataset_parms.source_path, 'test')
        self.device = device
        self.gender = self.dataset_parms.smpl_gender

        self.zfar = 100.0
        self.znear = 0.01
        self.trans = np.array([0.0, 0.0, 0.0]),
        self.scale = 1.0

        self.no_mask = bool(self.dataset_parms.no_mask)

        if dataset_parms.train_stage == 1:
            print('loading smpl data ', join(self.data_folder, 'smpl_parms.pth'))
            self.smpl_data = torch.load(join(self.data_folder, 'smpl_parms.pth'))
        else:
            print('loading smpl data ', join(self.data_folder, 'smpl_parms_pred.pth'))
            self.smpl_data = torch.load(join(self.data_folder, 'smpl_parms_pred.pth'))

        self.data_length = len(os.listdir(join(self.data_folder, 'images')))
        
        self.name_list = []
        for index, img in enumerate(sorted(os.listdir(join(self.data_folder, 'images')))):
            base_name = img.split('.')[0]
            self.name_list.append((index, base_name))
        
        self.image_fix = os.listdir(join(self.data_folder, 'images'))[0].split('.')[-1]
        
        if not dataset_parms.no_mask:
            self.mask_fix = os.listdir(join(self.data_folder, 'masks'))[0].split('.')[-1]

        print("total pose length", self.data_length )


        if dataset_parms.smpl_type == 'smplx':

            self.pose_data = self.smpl_data['body_pose'][:self.data_length, :66]
            self.transl_data = self.smpl_data['trans'][:self.data_length,:]
            self.rest_pose_data = self.smpl_data['body_pose'][:self.data_length, 66:]
        else:
            self.pose_data = self.smpl_data['body_pose'][:self.data_length]
            self.transl_data = self.smpl_data['trans'][:self.data_length,:]
            # self.rest_pose_data = self.smpl_data['body_pose'][:self.data_length, 66:]

        if dataset_parms.cam_static:
            cam_path = join(self.data_folder, 'cam_parms.npz')
            cam_npy = np.load(cam_path)
            extr_npy = cam_npy['extrinsic']
            intr_npy = cam_npy['intrinsic']
            self.R = np.array(extr_npy[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
            self.T = np.array([extr_npy[:3, 3]], np.float32)
            self.intrinsic = np.array(intr_npy, np.float32).reshape(3, 3)
        

        
    def __len__(self):
        return self.data_length

    def __getitem__(self, index, ignore_list=None):
        return self.getitem(index, ignore_list)

    @torch.no_grad()
    def getitem(self, index, ignore_list):
        pose_idx, name_idx = self.name_list[index]

        image_path = join(self.data_folder, 'images' ,name_idx + '.' + self.image_fix)
        
        cam_path = join(self.data_folder, 'cam_parms', name_idx + '.npz')

        if not self.dataset_parms.no_mask:
            mask_path = join(self.data_folder, 'masks', name_idx + '.' + self.mask_fix)
        if self.dataset_parms.train_stage == 2:

            inp_posmap_path = self.data_folder + '/inp_map/' +'inp_posemap_%s_%s.npz'% (str(self.dataset_parms.inp_posmap_size), str(pose_idx).zfill(8))
            inp_posmap = np.load(inp_posmap_path)['posmap' + str(self.dataset_parms.inp_posmap_size)]

        if not self.dataset_parms.cam_static:
            cam_npy = np.load(cam_path)
            extr_npy = cam_npy['extrinsic']
            intr_npy = cam_npy['intrinsic']
            R = np.array(extr_npy[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
            T = np.array(extr_npy[:3, 3], np.float32)
            intrinsic = np.array(intr_npy, np.float32).reshape(3, 3)
        
        else:
            R = self.R
            T = self.T
            intrinsic = self.intrinsic

        focal_length_x = intrinsic[0, 0]
        focal_length_y = intrinsic[1, 1]

        pose_data = self.pose_data[pose_idx]
        transl_data = self.transl_data[pose_idx]

        image = Image.open(image_path)
        width, height = image.size

        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)

        if not self.dataset_parms.no_mask:
            mask = np.array(Image.open(mask_path))

            if len(mask.shape) <3:
                mask = mask[...,None]

            mask[mask < 128] = 0
            mask[mask >= 128] = 1
            # color_img = image * mask 
            color_img = image * mask + (1 - mask) * 255
            image = Image.fromarray(np.array(color_img, dtype=np.byte), "RGB")

    
        data_item = dict()

        # data_item['vtransf'] = vtransf
        # data_item['query_pos_map'] = query_posmap.transpose(2,0,1)
        if self.dataset_parms.train_stage == 2:
            data_item['inp_pos_map'] = inp_posmap.transpose(2,0,1)


        resized_image = torch.from_numpy(np.array(image)) / 255.0
        if len(resized_image.shape) == 3:
            resized_image =  resized_image.permute(2, 0, 1)
        else:
            resized_image =  resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

        original_image = resized_image.clamp(0.0, 1.0)

        data_item['original_image'] = original_image
        data_item['FovX'] = FovX
        data_item['FovY'] = FovY
        data_item['width'] = width
        data_item['height'] = height
        data_item['pose_idx'] = pose_idx
        data_item['pose_data'] = pose_data
        data_item['transl_data'] = transl_data
        if self.dataset_parms.smpl_type == 'smplx':
            rest_pose = self.rest_pose_data[pose_idx]
            data_item['rest_pose'] = rest_pose
  
        world_view_transform = torch.tensor(getWorld2View2(R, T, self.trans, self.scale)).transpose(0, 1)
        projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=FovX, fovY=FovY, K=intrinsic, h=height, w=width).transpose(0,1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]
        data_item['world_view_transform'] = world_view_transform
        data_item['projection_matrix'] = projection_matrix
        data_item['full_proj_transform'] = full_proj_transform
        data_item['camera_center'] = camera_center
        
        return data_item

class MonoDataset_novel_pose(Dataset):
    @torch.no_grad()
    def __init__(self, dataset_parms,
                 device = torch.device('cuda:0')):
        super(MonoDataset_novel_pose, self).__init__()


        self.dataset_parms = dataset_parms
        self.data_folder = dataset_parms.test_folder
        self.device = device
        self.gender = self.dataset_parms.smpl_gender

        self.zfar = 100.0
        self.znear = 0.01
        self.trans = np.array([0.0, 0.0, 0.0]),
        self.scale = 1.0

        self.no_mask = bool(self.dataset_parms.no_mask)

        print('loading smpl data ', join(self.data_folder, 'smpl_parms.pth'))
        self.smpl_data = torch.load(join(self.data_folder, 'smpl_parms.pth'))

        self.data_length = self.smpl_data['body_pose'].shape[0]
        print("total pose length", self.data_length )

        if dataset_parms.smpl_type == 'smplx':

            self.pose_data = self.smpl_data['body_pose'][:self.data_length, :66]
            self.transl_data = self.smpl_data['trans'][:self.data_length,:]
            self.rest_pose_data = self.smpl_data['body_pose'][:self.data_length, 66:]
        else:
            self.pose_data = self.smpl_data['body_pose']
            self.transl_data = self.smpl_data['trans']
            # self.rest_pose_data = self.smpl_data['body_pose'][:self.data_length, 66:]

        print('novel pose shape', self.pose_data.shape)
        print('novel pose shape', self.transl_data.shape)
        if dataset_parms.cam_static:
            cam_path = join(self.data_folder, 'cam_parms.npz')
            cam_npy = np.load(cam_path)
            extr_npy = cam_npy['extrinsic']
            intr_npy = cam_npy['intrinsic']
            self.R = np.array(extr_npy[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
            self.T = np.array([extr_npy[:3, 3]], np.float32)
            self.intrinsic = np.array(intr_npy, np.float32).reshape(3, 3)
        

        
    def __len__(self):
        return self.data_length

    def __getitem__(self, index, ignore_list=None):
        return self.getitem(index, ignore_list)

    @torch.no_grad()
    def getitem(self, index, ignore_list):

        pose_idx  =  index
        if self.dataset_parms.train_stage == 2:
            inp_posmap_path = self.data_folder + '/inp_map/' +'inp_posemap_%s_%s.npz'% (str(self.dataset_parms.inp_posmap_size), str(pose_idx).zfill(8))
            inp_posmap = np.load(inp_posmap_path)['posmap' + str(self.dataset_parms.inp_posmap_size)]

        R = self.R
        T = self.T
        intrinsic = self.intrinsic

        focal_length_x = intrinsic[0, 0]
        focal_length_y = intrinsic[1, 1]

        pose_data = self.pose_data[pose_idx]
        transl_data = self.transl_data[pose_idx]


        width, height = 1024, 1024

        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)


        data_item = dict()
        if self.dataset_parms.train_stage == 2:
            data_item['inp_pos_map'] = inp_posmap.transpose(2,0,1)

        data_item['FovX'] = FovX
        data_item['FovY'] = FovY
        data_item['width'] = width
        data_item['height'] = height
        data_item['pose_idx'] = pose_idx
        data_item['pose_data'] = pose_data
        data_item['transl_data'] = transl_data
        if self.dataset_parms.smpl_type == 'smplx':
            rest_pose = self.rest_pose_data[pose_idx]
            data_item['rest_pose'] = rest_pose
  
        world_view_transform = torch.tensor(getWorld2View2(R, T, self.trans, self.scale)).transpose(0, 1)
        projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=FovX, fovY=FovY, K=intrinsic, h=height, w=width).transpose(0,1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]
        data_item['world_view_transform'] = world_view_transform
        data_item['projection_matrix'] = projection_matrix
        data_item['full_proj_transform'] = full_proj_transform
        data_item['camera_center'] = camera_center
        
        return data_item

class MonoDataset_novel_view(Dataset):
    # these code derive from humannerf(https://github.com/chungyiweng/humannerf), to keep the same view point
    ROT_CAM_PARAMS = {
        'zju_mocap': {'rotate_axis': 'z', 'inv_angle': True},
        'wild': {'rotate_axis': 'y', 'inv_angle': False}
    }
    @torch.no_grad()
    def __init__(self, dataset_parms, device = torch.device('cuda:0')):
        super(MonoDataset_novel_view, self).__init__()

        self.dataset_parms = dataset_parms

        self.data_folder = join(dataset_parms.source_path, 'test')
        self.device = device
        self.gender = self.dataset_parms.smpl_gender

        self.zfar = 100.0
        self.znear = 0.01
        self.trans = np.array([0.0, 0.0, 0.0]),
        self.scale = 1.0

        self.no_mask = bool(self.dataset_parms.no_mask)

        if dataset_parms.train_stage == 1:
            print('loading smpl data ', join(self.data_folder, 'smpl_parms.pth'))
            self.smpl_data = torch.load(join(self.data_folder, 'smpl_parms.pth'))
        else:
            print('loading smpl data ', join(self.data_folder, 'smpl_parms_pred.pth'))
            self.smpl_data = torch.load(join(self.data_folder, 'smpl_parms_pred.pth'))

        self.data_length = len(os.listdir(join(self.data_folder, 'images')))
        
        self.name_list = []
        for index, img in enumerate(sorted(os.listdir(join(self.data_folder, 'images')))):
            base_name = img.split('.')[0]
            self.name_list.append((index, base_name))
        
        self.image_fix = os.listdir(join(self.data_folder, 'images'))[0].split('.')[-1]
        
        if not dataset_parms.no_mask:
            self.mask_fix = os.listdir(join(self.data_folder, 'masks'))[0].split('.')[-1]

        print("total pose length", self.data_length )


        if dataset_parms.smpl_type == 'smplx':

            self.pose_data = self.smpl_data['body_pose'][:self.data_length, :66]
            self.transl_data = self.smpl_data['trans'][:self.data_length,:]
            self.rest_pose_data = self.smpl_data['body_pose'][:self.data_length, 66:]
        else:
            self.pose_data = self.smpl_data['body_pose'][:self.data_length]
            self.transl_data = self.smpl_data['trans'][:self.data_length,:]
            # self.rest_pose_data = self.smpl_data['body_pose'][:self.data_length, 66:]

        if dataset_parms.cam_static:
            cam_path = join(self.data_folder, 'cam_parms.npz')
            cam_npy = np.load(cam_path)
            extr_npy = cam_npy['extrinsic']
            intr_npy = cam_npy['intrinsic']
            self.extr_npy = extr_npy
            self.R = np.array(extr_npy[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
            self.T = np.array([extr_npy[:3, 3]], np.float32)
            self.intrinsic = np.array(intr_npy, np.float32).reshape(3, 3)
        
        self.src_type = 'wild'
        
    def __len__(self):
        return self.data_length

    def __getitem__(self, index,):
        return self.getitem(index)
    
    def update_smpl(self, pose_idx, frame_num):
        from third_parties.smpl.smpl_numpy import SMPL
        MODEL_DIR = self.dataset_parms.project_path + '/third_parties/smpl/models'
        smpl_model = SMPL(sex='neutral', model_dir=MODEL_DIR)
        _, tpose_joints = smpl_model(np.zeros((1, 72)), self.smpl_data['beta'].squeeze().numpy())

        pelvis_pos = tpose_joints[0].copy()

        Th = pelvis_pos +self.smpl_data['trans'][pose_idx].numpy()
        self.Th = Th

        self.data_length = frame_num
        self.fix_pose_idx = pose_idx

    def get_freeview_camera(self, frame_idx, total_frames, trans):
        E = rotate_camera_by_frame_idx(
                extrinsics= self.extr_npy, 
                frame_idx=frame_idx,
                period=total_frames,
                trans=trans,
                **self.ROT_CAM_PARAMS[self.src_type])
        return E

    @torch.no_grad()
    def getitem(self, index,):
        pose_idx = self.fix_pose_idx
        _, name_idx = self.name_list[0]

        image_path = join(self.data_folder, 'images' ,name_idx + '.' + self.image_fix)

        if self.dataset_parms.train_stage == 2:

            inp_posmap_path = self.data_folder + '/inp_map/' +'inp_posemap_%s_%s.npz'% (str(self.dataset_parms.inp_posmap_size), str(pose_idx).zfill(8))
            inp_posmap = np.load(inp_posmap_path)['posmap' + str(self.dataset_parms.inp_posmap_size)]

        extr_npy =  self.get_freeview_camera(index, self.data_length, self.Th)

        R = np.array(extr_npy[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
        T = np.array([extr_npy[:3, 3]], np.float32)

        intrinsic = self.intrinsic

        focal_length_x = intrinsic[0, 0]
        focal_length_y = intrinsic[1, 1]

        pose_data = self.pose_data[pose_idx]
        transl_data = self.transl_data[pose_idx]

        image = Image.open(image_path)
        width, height = image.size

        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)
        data_item = dict()
        if self.dataset_parms.train_stage == 2:
            data_item['inp_pos_map'] = inp_posmap.transpose(2,0,1)

        data_item['FovX'] = FovX
        data_item['FovY'] = FovY
        data_item['width'] = width
        data_item['height'] = height
        data_item['pose_idx'] = pose_idx
        data_item['pose_data'] = pose_data
        data_item['transl_data'] = transl_data
        if self.dataset_parms.smpl_type == 'smplx':
            rest_pose = self.rest_pose_data[pose_idx]
            data_item['rest_pose'] = rest_pose
  
        world_view_transform = torch.tensor(getWorld2View2(R, T, self.trans, self.scale)).transpose(0, 1)
        projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=FovX, fovY=FovY, K=intrinsic, h=height, w=width).transpose(0,1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]
        data_item['world_view_transform'] = world_view_transform
        data_item['projection_matrix'] = projection_matrix
        data_item['full_proj_transform'] = full_proj_transform
        data_item['camera_center'] = camera_center
        
        return data_item


