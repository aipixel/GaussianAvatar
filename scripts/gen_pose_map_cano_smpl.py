

import numpy  as np
import torch
from os.path import join
import os
import sys
sys.path.append('../')
from submodules  import smplx

from scipy.spatial.transform import Rotation as R
import trimesh
from utils.general_utils import load_masks, load_barycentric_coords, gen_lbs_weight_from_ori
from arguments import smplx_cpose_param, smpl_cpose_param


def render_posmap(v_minimal, faces, uvs, faces_uvs, img_size=32):
    '''
    v_minimal: vertices of the minimally-clothed SMPL body mesh
    faces: faces (triangles) of the minimally-clothed SMPL body mesh
    uvs: the uv coordinate of vertices of the SMPL body model
    faces_uvs: the faces (triangles) on the UV map of the SMPL body model
    '''
    from posmap_generator.lib.renderer.gl.pos_render import PosRender

    # instantiate renderer
    rndr = PosRender(width=img_size, height=img_size)

    # set mesh data on GPU
    rndr.set_mesh(v_minimal, faces, uvs, faces_uvs)

    # render
    rndr.display()

    # retrieve the rendered buffer
    uv_pos = rndr.get_color(0)
    uv_mask = uv_pos[:, :, 3]
    uv_pos = uv_pos[:, :, :3]

    uv_mask = uv_mask.reshape(-1)
    uv_pos = uv_pos.reshape(-1, 3)

    rendered_pos = uv_pos[uv_mask != 0.0]

    uv_pos = uv_pos.reshape(img_size, img_size, 3)

    # get face_id (triangle_id) per pixel
    face_id = uv_mask[uv_mask != 0].astype(np.int32) - 1

    assert len(face_id) == len(rendered_pos)

    return uv_pos, uv_mask, face_id

def save_obj(data_path, name):
    smpl_data = torch.load( data_path + '/smpl_parms.pth')
    smpl_model = smplx.SMPL(model_path ='/mnt/disk/avatar/gaussian_map/assets/smpl_files/smpl',batch_size = 1)
    cano_dir = os.path.join(data_path,)


    cano_smpl = smpl_model.forward(betas=smpl_data['beta'],
                            global_orient=smpl_cpose_param[:, :3],
                            transl = torch.tensor([[0, 0.30, 0]]),
                            # global_orient=cpose_param[:, :3],
                            body_pose=smpl_cpose_param[:, 3:],
                            )

    ori_vertices = cano_smpl.vertices.detach().cpu().numpy().squeeze()
    joint_mat = cano_smpl.A
    print(joint_mat.shape)
    torch.save(joint_mat ,join(cano_dir, 'smpl_cano_joint_mat.pth'))


    mesh = trimesh.Trimesh(ori_vertices, smpl_model.faces, process=False)
    mesh.export('%s/%s.obj' % (cano_dir, 'cano_smpl'))



def save_npz(data_path, res=128):
    from posmap_generator.lib.renderer.mesh import load_obj_mesh
    verts, faces, uvs, faces_uvs = load_obj_mesh(uv_template_fn, with_texture=True)
    start_obj_num = 0
    result = {}
    body_mesh = trimesh.load('%s/%s.obj'%(data_path, 'cano_smpl'), process=False)

    if res==128:
        posmap128, _, _ = render_posmap(body_mesh.vertices, body_mesh.faces, uvs, faces_uvs, img_size=128)
        result['posmap128'] = posmap128   
    elif res == 256:
    
        posmap256, _, _ = render_posmap(body_mesh.vertices, body_mesh.faces, uvs, faces_uvs, img_size=256)
        result['posmap256'] = posmap256

    else:
        posmap512, _, _ = render_posmap(body_mesh.vertices, body_mesh.faces, uvs, faces_uvs, img_size=512)
        result['posmap512'] = posmap512

    save_fn = join(data_path, 'query_posemap_%s_%s.npz'% (str(res), 'cano_smpl'))
    np.savez(save_fn, **result)


def save_joint(data_path, assets_path):

    resolution  = 256

    smpl_model = smplx.SMPLX(model_path='/mnt/disk/avatar/gaussian_map/assets/smpl_files/smplx', use_pca=False,
                             num_pca_comps=45, flat_hand_mean=True, batch_size=1)
    atten_map = torch.from_numpy(np.loadtxt('/mnt/disk/avatar/gaussian_avatar/asset/pose_map_scanimate.txt').transpose())
    
    ori_lbs_weight = smpl_model.lbs_weights
    
    flist_uv, valid_idx, uv_coord_map = load_masks(assets_path, resolution, body_model='smplx')
    bary_coords = load_barycentric_coords(assets_path, resolution, body_model='smplx')
    print(flist_uv.shape)
    print(ori_lbs_weight.shape)

    map_lbs = gen_lbs_weight_from_ori(ori_lbs_weight, bary_coords.cpu(), flist_uv.cpu()) #[, uvsize, uvsize, 24]
    np.save( join(data_path, "lbs_map_{}".format(str(resolution))), map_lbs.numpy())

    map_lbs = map_lbs.reshape(-1,55)
    lbs_ori = map_lbs[..., :24]
    lbs_ori[..., 22:] = 0.
    lbs = torch.einsum('ij,nj->ni', atten_map.float(), lbs_ori)
    # lbs = lbs[..., :-2].reshape(resolution, resolution, 21)
    lbs = lbs[..., :-2]
    save_lbs = torch.zeros_like(lbs)
    save_lbs[valid_idx.cpu(),...] = lbs[valid_idx.cpu(),...]
    save_lbs = save_lbs.reshape(resolution, resolution, 21)
    print(save_lbs.shape)
    from PIL import Image
    image = Image.fromarray(np.array(save_lbs.sum(dim=-1, keepdim=True).expand(resolution,resolution,3)*255, dtype=np.byte), "RGB")
    image.save('test.png')

    np.save( join(data_path, "attention_map_{}".format(str(resolution))), save_lbs.numpy())



if __name__ == '__main__':
    smplx_parm_path = '' # path to the folder that include smpl params
    parms_name = 'smpl_parms.pth'
    uv_template_fn = '../assets/template_mesh_smpl_uv.obj'
    assets_path = ''    # path to the folder that include 'assets'

    print('saving obj...')
    save_obj(smplx_parm_path, parms_name)

    print('saving pose_map 512 ...')
    save_npz(smplx_parm_path, 512)

    # save_joint(smplx_parm_path, assets_path)