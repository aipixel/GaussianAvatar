import os
import cv2
import numpy as np
import argparse

from os.path import realpath, join, dirname, basename

from lib.renderer.gl.pos_render import PosRender
from lib.renderer.mesh import load_obj_mesh, save_obj_mesh


def render_posmap(obj_file, H=32, W=32, output_dir='.'):
    # load obj file
    vertices, faces, uvs, faces_uvs = load_obj_mesh(obj_file, with_texture=True)

    # instantiate renderer
    rndr = PosRender(width=W, height=H)

    # set mesh data on GPU
    rndr.set_mesh(vertices, faces, uvs, faces_uvs)   

    # render
    rndr.display()

    # retrieve the rendered buffer
    uv_pos = rndr.get_color(0)
    uv_mask = uv_pos[:,:,3]
    uv_pos = uv_pos[:,:,:3]

    # save mask file    
    cv2.imwrite(join(output_dir, basename(obj_file).replace('.obj', '_posmap.png')), 255.0*uv_pos)

    # save mask file    
    cv2.imwrite(join(output_dir, basename(obj_file).replace('.obj', '_mask.png')), 255.0*uv_mask)

    # save the rendered pos map as point cloud
    uv_mask = uv_mask.reshape(-1)
    uv_pos = uv_pos.reshape(-1,3)
    rendered_pos = uv_pos[uv_mask != 0.0]
    save_obj_mesh(join(output_dir, basename(obj_file).replace('.obj', '_in3D.obj')), rendered_pos)

    # get face_id per pixel
    face_id = uv_mask.astype(np.int32) - 1

    p_list = []
    c_list = []
    
    # randomly assign color per face id
    for i in range(faces.shape[0]):
        pos = uv_pos[face_id == i]
        c = np.random.rand(1,3).repeat(pos.shape[0],axis=0)
        p_list.append(pos)
        c_list.append(c)

    p_list = np.concatenate(p_list, 0)
    c_list = np.concatenate(c_list, 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str)
    args = parser.parse_args()

    SCRIPT_DIR = dirname(realpath(__file__))
    smpl_template_pth = join(SCRIPT_DIR, '../../../assets/template_mesh_uv.obj')
    
    output_dir = join(SCRIPT_DIR, '../example_outputs')
    os.makedirs(output_dir, exist_ok=True)

    render_posmap(smpl_template_pth, H=128, W=128, output_dir=output_dir)
