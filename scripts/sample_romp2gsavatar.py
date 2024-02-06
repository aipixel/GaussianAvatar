
import os
import torch
import shutil
import numpy as np
import numpy as np
from os.path import join


def load_smpl_param(path, data_list, return_thata=True):
    smpl_params = dict(np.load(str(path)))
    if "thetas" in smpl_params:
        smpl_params["body_pose"] = smpl_params["thetas"][..., 3:]
        smpl_params["global_orient"] = smpl_params["thetas"][..., :3]

    if not return_thata:
        return {
        "betas": smpl_params["betas"].astype(np.float32),
        "body_pose": smpl_params["body_pose"].astype(np.float32),
        "global_orient": smpl_params["global_orient"].astype(np.float32),
        "transl": smpl_params["transl"].astype(np.float32),
    }

    theta = np.zeros((len(data_list), 72)).astype(np.float32)
    trans  = np.zeros((len(data_list), 3)).astype(np.float32)
    iter = 0
    for idx in data_list:
        theta[iter, :3] = smpl_params["global_orient"][idx].astype(np.float32)
        theta[iter, 3:] = smpl_params["body_pose"][idx].astype(np.float32)
        trans[iter, :] = smpl_params["transl"][idx].astype(np.float32)

        iter +=1

    return {
        "beta": torch.from_numpy(smpl_params["betas"].reshape(1,10).astype(np.float32)),
        "body_pose": torch.from_numpy(theta),
        "trans": torch.from_numpy(trans),
    }


# snap male 3 casual
# train_list = [0, 455, 4]
# test_list = [456, 675, 4]

# snap female 3 casual
# train_list = [0:445:4]
# test_list = 446:647:4]
snap = False

data_folder = '/mnt/disk/data/Mono/data/'
subject = 'snap_male3casual'

all_image_path = join(data_folder, subject, 'images')
all_mask_apth = join(data_folder, subject, 'masks')

if snap:
    train_split_name = sorted(os.listdir(all_image_path))[0:455:4]
    test_split_name = sorted(os.listdir(all_image_path))[456:675:4]
    scene_length = len(os.listdir(all_image_path))
    train_list = list(range(scene_length))[0:455:4]
    test_list = list(range(scene_length))[456:675:4]

# the rule to split data is derived from InstantAvatar
else:
    scene_length = len(os.listdir(all_image_path))
    print('len:', scene_length)
    num_val = scene_length // 5
    length = int(1 / (num_val) * scene_length)
    offset = length // 2
    val_list = list(range(scene_length))[offset::length]
    train_list = list(set(range(scene_length)) - set(val_list))
    test_list = val_list[:len(val_list) // 2]
    val_list = val_list[len(val_list) // 2:]

    train_split_name = []
    test_split_name = []
    for idx,idx_name in enumerate(sorted(os.listdir(all_image_path))):
        if idx in train_list:
            train_split_name.append(idx_name)
        if idx in test_list:
            test_split_name.append(idx_name)


data_path = join(data_folder, subject)

out_path = join(data_path, 'train')
out_image_path =join(out_path, 'images')
out_mask_path =join(out_path, 'masks')

os.makedirs(out_image_path, exist_ok=True)
os.makedirs(out_mask_path, exist_ok=True)

test_path = join(data_path, 'test')
test_image_path =join(test_path, 'images')
test_mask_path =join(test_path, 'masks')

os.makedirs(test_image_path, exist_ok=True)
os.makedirs(test_mask_path, exist_ok=True)

# load camera
camera = np.load(join(data_path, "cameras.npz"))
intrinsic = np.array(camera["intrinsic"])
extrinsic = np.array(camera["extrinsic"])
cam_all = {} 

cam_all['intrinsic'] = intrinsic
cam_all['extrinsic'] = extrinsic
np.savez(join(out_path, 'cam_parms.npz'), **cam_all)
np.savez(join(test_path, 'cam_parms.npz'), **cam_all)

train_smpl_params = load_smpl_param(join(data_path, "poses_optimized.npz"), train_list)

torch.save(train_smpl_params ,join(out_path, 'smpl_parms.pth'))

test_smpl_params = load_smpl_param(join(data_path, "poses_optimized.npz"), test_list)

torch.save(test_smpl_params ,join(test_path, 'smpl_parms.pth'))

assert len(os.listdir(all_image_path)) == len(os.listdir(all_mask_apth))

train_sum_dict = {}
for image_name in train_split_name:
    shutil.copy(join(all_image_path, image_name), join(out_image_path, image_name))
    shutil.copy(join(all_mask_apth, image_name), join(out_mask_path, image_name))

test_sum_dict = {}
for image_name in test_split_name:
    shutil.copy(join(all_image_path, image_name), join(test_image_path, image_name))
    shutil.copy(join(all_mask_apth, image_name), join(test_mask_path, image_name))
