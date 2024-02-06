import sys
sys.path.append('../')
from submodules  import smplx
import pyrender
import torch
from os.path import join
from os import makedirs
import numpy as np
import trimesh
import cv2
import os
from pyrender.constants import RenderFlags

smpl_model = smplx.SMPL(model_path='./assets/smpl_files/smpl', gender='neutral', batch_size=1)
width = 1024
height = 1024
data_path = '' 
outpath = ''
beta_smpl_path = join(data_path, 'smpl_parms.pth')
beta_smpl_data = torch.load(beta_smpl_path)
smplx_parms_path = join(data_path, 'smpl_parms_pred.pth')
cam_parms_path  = data_path + '/cam_parms.npz'
image_path = join(data_path, 'images')

ori_render_path = join(outpath, 'pred_smplx_render')
makedirs(ori_render_path, exist_ok=True)

colors_dict = {
            'red': np.array([0.5, 0.2, 0.2]),
            'pink': np.array([0.7, 0.5, 0.5]),
            'neutral': np.array([0.7, 0.7, 0.6]),
            # 'purple': np.array([0.5, 0.5, 0.7]),
            'purple': np.array([0.55, 0.4, 0.9]),
            'green': np.array([0.5, 0.55, 0.3]),
            'sky': np.array([0.3, 0.5, 0.55]),
            'white': np.array([1.0, 0.98, 0.94]),
        }
color = colors_dict['white']
num_training_frames = len(os.listdir(image_path))
pose_embedding = torch.nn.Embedding(num_training_frames, 66, sparse=True)
transl_embedding = torch.nn.Embedding(num_training_frames, 3, sparse=True)
smpl_data = torch.load(smplx_parms_path)
print(smpl_data['body_pose'].shape)

renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
scene = pyrender.Scene()
cam_npy = np.load(cam_parms_path)
extr_npy = cam_npy['extrinsic']
intr_npy = np.array(cam_npy['intrinsic']).reshape(3, 3)

R = np.array(extr_npy[:3, :3], np.float32).reshape(3, 3)
T = np.array(extr_npy[:3, 3], np.float32)
K = [np.array(intr_npy[0,0] ), np.array(intr_npy[1,1]), np.array(intr_npy[0,2]), np.array(intr_npy[1,2])]

camera = pyrender.IntrinsicsCamera(fx = -K[0], fy = -K[1], cx = K[2], cy = K[3])
camera_pose = np.eye(4)
camera_pose[:3, :3] = R.transpose()
camera_pose[:3, 3] = -np.dot(R.transpose(), T)
camera_pose[:, 1:3] = -camera_pose[:, 1:3]

cam_node = scene.add(camera, pose=camera_pose)
scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0), pose=np.array([
                [1.0,    0.0,  0.0, 0.0],
                [0.0,    -1.0,  0.0, 0.0],
                [0.0,    0.0,  -1.0, 0.0],
                [0.0,    0.0,  0.0, 1.0], ]))
     
render_flags = RenderFlags.RGBA  | RenderFlags.SHADOWS_SPOT

for pose_idx, image_name in enumerate(sorted(os.listdir(image_path))):
    print('process: ', pose_idx)
    idx_name = image_name.split('.')[0]
    pose_idx_tensor = torch.tensor(pose_idx)
    idx_image_path  = join(image_path, image_name)
    idx_ori_rend_path = join(ori_render_path, image_name)

    ori_smpl = smpl_model.forward(betas=beta_smpl_data['beta'][0][None],
                                global_orient=smpl_data['body_pose'][pose_idx, :3][None],
                                transl =smpl_data['trans'][pose_idx][None],
                                # global_orient=cpose_param[:, :3],
                                body_pose=smpl_data['body_pose'][pose_idx, 3:][None])
    

    img = cv2.imread(idx_image_path)
    ori_smpl_mesh = trimesh.Trimesh(vertices=ori_smpl.vertices[0].detach().cpu().numpy().squeeze(), faces=smpl_model.faces)

    material = pyrender.MetallicRoughnessMaterial(
    metallicFactor=0.2,
    roughnessFactor=0.6,
    alphaMode='OPAQUE',
    baseColorFactor=(color[0], color[1], color[2], 1.0))

    ori_smpl_mesh = pyrender.Mesh.from_trimesh(ori_smpl_mesh,  material=material)

    ori_smpl_mesh_node = scene.add(ori_smpl_mesh) 
    render_img, depth_img = renderer.render(scene, flags=render_flags)
    render_img = cv2.flip(render_img, -1)
    depth_img = cv2.flip(depth_img, -1)
    valid_mask = (depth_img >  0)[..., np.newaxis]
    output_img = render_img[:, :, :-1] * valid_mask +  img *(1-valid_mask)
    cv2.imwrite(idx_ori_rend_path, output_img)
    scene.remove_node(ori_smpl_mesh_node)


