
import torch
import sys
from datetime import datetime
import numpy as np
import random
from os.path import join, dirname, realpath

def worker_init_fn(worker_id):  # set numpy's random seed
    seed = torch.initial_seed()
    seed = seed % (2 ** 32)
    np.random.seed(seed + worker_id)

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))


def to_cuda(items: dict, device, add_batch = False, precision = torch.float32):
    items_cuda = dict()
    for key, data in items.items():
        if isinstance(data, torch.Tensor):
            items_cuda[key] = data.to(device)
        elif isinstance(data, np.ndarray):
            items_cuda[key] = torch.from_numpy(data).to(device)
        elif isinstance(data, dict):  # usually some float tensors
            for key2, data2 in data.items():
                if isinstance(data2, np.ndarray):
                    data[key2] = torch.from_numpy(data2).to(device)
                elif isinstance(data2, torch.Tensor):
                    data[key2] = data2.to(device)
                else:
                    raise TypeError('Do not support other data types.')
                if data[key2].dtype == torch.float32 or data[key2].dtype == torch.float64:
                    data[key2] = data[key2].to(precision)
            items_cuda[key] = data
        else:
            items_cuda[key] = data
        if isinstance(items_cuda[key], torch.Tensor) and\
                (items_cuda[key].dtype == torch.float32 or items_cuda[key].dtype == torch.float64):
            items_cuda[key] = items_cuda[key].to(precision)
        if add_batch:
            if isinstance(items_cuda[key], torch.Tensor):
                items_cuda[key] = items_cuda[key].unsqueeze(0)
            elif isinstance(items_cuda[key], dict):
                for k in items_cuda[key].keys():
                    items_cuda[key][k] = items_cuda[key][k].unsqueeze(0)
            else:
                items_cuda[key] = [items_cuda[key]]
    return items_cuda

def getIdxMap_torch(img, offset=False):
    # img has shape [channels, H, W]
    C, H, W = img.shape
    import torch
    idx = torch.stack(torch.where(~torch.isnan(img[0])))
    if offset:
        idx = idx.float() + 0.5
    idx = idx.view(2, H * W).float().contiguous()
    idx = idx.transpose(0, 1)

    idx = idx / (H-1) if not offset else idx / H
    return idx

def load_masks(PROJECT_DIR, posmap_size, body_model='smpl'):
    uv_mask_faceid = np.load(join(PROJECT_DIR, 'assets', 'uv_masks', 'uv_mask{}_with_faceid_{}.npy'.format(posmap_size, body_model))).reshape(posmap_size, posmap_size)
    uv_mask_faceid = torch.from_numpy(uv_mask_faceid).long().cuda()
    
    smpl_faces = np.load(join(PROJECT_DIR, 'assets', '{}_faces.npy'.format(body_model.lower()))) # faces = triangle list of the body mesh
    flist = torch.tensor(smpl_faces.astype(np.int32)).long().cuda()
    flist_uv = get_face_per_pixel(uv_mask_faceid, flist).cuda() # Each (valid) pixel on the uv map corresponds to a point on the SMPL body; flist_uv is a list of these triangles

    points_idx_from_posmap = (uv_mask_faceid!=-1).reshape(-1)

    uv_coord_map = getIdxMap_torch(torch.rand(3, posmap_size, posmap_size)).cuda()
    uv_coord_map.requires_grad = True

    return flist_uv, points_idx_from_posmap, uv_coord_map


def load_barycentric_coords(PROJECT_DIR, posmap_size, body_model='smpl'):
    '''
    load the barycentric coordinates (pre-computed and saved) of each pixel on the positional map.
    Each pixel on the positional map corresponds to a point on the SMPL / SMPL-X body (mesh)
    which falls into a triangle in the mesh. This function loads the barycentric coordinate of the point in that triangle.
    '''
    bary = np.load(join(PROJECT_DIR, 'assets', 'bary_coords_uv_map', 'bary_coords_{}_uv{}.npy'.format(body_model, posmap_size)))
    bary = bary.reshape(posmap_size, posmap_size, 3)
    return torch.from_numpy(bary).cuda()


def get_face_per_pixel(mask, flist):
    '''
    :param mask: the uv_mask returned from posmap renderer, where -1 stands for background
                 pixels in the uv map, where other value (int) is the face index that this
                 pixel point corresponds to.
    :param flist: the face list of the body model,
        - smpl, it should be an [13776, 3] array
        - smplx, it should be an [20908,3] array
    :return:
        flist_uv: an [img_size, img_size, 3] array, each pixel is the index of the 3 verts that belong to the triangle
    Note: we set all background (-1) pixels to be 0 to make it easy to parralelize, but later we
        will just mask out these pixels, so it's fine that they are wrong.
    '''
    mask2 = mask.clone()
    mask2[mask == -1] = 0 #remove the -1 in the mask, so that all mask elements can be seen as meaningful faceid
    flist_uv = flist[mask2]
    return flist_uv

def gen_transf_mtx_from_vtransf(vtransf, bary_coords, faces, scaling=1.0):
    '''
    interpolate the local -> global coord transormation given such transformations defined on 
    the body verts (pre-computed) and barycentric coordinates of the query points from the uv map.

    Note: The output of this function, i.e. the transformation matrix of each point, is not a pure rotation matrix (SO3).
    
    args:
        vtransf: [batch, #verts, 3, 3] # per-vertex rotation matrix
        bary_coords: [uv_size, uv_size, 3] # barycentric coordinates of each query point (pixel) on the query uv map 
        faces: [uv_size, uv_size, 3] # the vert id of the 3 vertices of the triangle where each uv pixel locates

    returns: 
        [batch, uv_size, uv_size, 3, 3], transformation matrix for points on the uv surface
    '''
    #  
    vtransf_by_tris = vtransf[:, faces] # shape will be [batch, uvsize, uvsize, 3, 3, 3], where the the last 2 dims being the transf (pure rotation) matrices, the other "3" are 3 points of each triangle
    transf_mtx_uv_pts = torch.einsum('bpqijk,pqi->bpqjk', vtransf_by_tris, bary_coords) # [batch, uvsize, uvsize, 3, 3], last 2 dims are the rotation matix
    transf_mtx_uv_pts *= scaling
    return transf_mtx_uv_pts


def gen_lbs_weight_from_ori(lbs_weight_smpl, bary_coords, faces):
    '''
    
    args:
        lbs_weight_smpl: [verts, 24] # per-vertex rotation matrix
        bary_coords: [uv_size, uv_size, 3] # barycentric coordinates of each query point (pixel) on the query uv map 
        faces: [uv_size, uv_size, 3] # the vert id of the 3 vertices of the triangle where each uv pixel locates

    returns: 
        [batch, uv_size, uv_size, 24], transformation matrix for points on the uv surface
    '''
    #  
    vtransf_by_tris = lbs_weight_smpl[faces] # shape will be [ uvsize, uvsize, 3, 24], where the the last 2 dims being the transf (pure rotation) matrices, the other "3" are 3 points of each triangle
    transf_mtx_uv_pts = torch.einsum('pqik,pqi->pqk', vtransf_by_tris, bary_coords) # [, uvsize, uvsize, 24], last 2 dims are the rotation matix
    return transf_mtx_uv_pts

def adjust_loss_weights(init_weight, current_epoch, mode='decay', start=400, every=20):
    # decay or rise the loss weights according to the given policy and current epoch
    # mode: decay, rise or binary

    if mode != 'binary':
        if current_epoch < start:
            if mode == 'rise':
                weight = init_weight * 1e-6 # use a very small weight for the normal loss in the beginning until the chamfer dist stabalizes
            else:
                weight = init_weight
        else:
            if every == 0:
                weight = init_weight # don't rise, keep const
            else:
                if mode == 'rise':
                    weight = init_weight * (1.05 ** ((current_epoch - start) // every))
                else:
                    weight = init_weight * (0.85 ** ((current_epoch - start) // every))

    return weight


def save_video(input_path, outname, photo_size=(1024, 1024), fps = 30):
    import os
    import cv2
    from utils.system_utils import mkdir_p
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    save_path = os.path.join('./test_video')
    save_name = os.path.join('./test_video', outname+'.mp4')
    mkdir_p(save_path)

    videoWriter = cv2.VideoWriter(save_name, fourcc, fps, photo_size)

    for index in range(len(os.listdir(input_path))):
        # print(1)
        pred_path = os.path.join(input_path, '%05d.png' % index)
        # print(pred_path)
        pred_frame = cv2.imread(pred_path)
        # print(pred_frame)
    # print(gt_path)
        videoWriter.write(pred_frame)
    videoWriter.release()