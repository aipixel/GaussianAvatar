import torch
import os
from tqdm import tqdm
from os import makedirs
import torchvision
from utils.general_utils import safe_state, to_cuda
from argparse import ArgumentParser
from arguments import ModelParams, get_combined_args, NetworkParams, OptimizationParams
from model.avatar_model import  AvatarModel


def render_sets(model, net, opt, epoch:int):
    with torch.no_grad():
        avatarmodel = AvatarModel(model, net, opt, train=False)
        avatarmodel.training_setup()
        avatarmodel.load(epoch)
        
        novel_pose_dataset = avatarmodel.getNovelposeDataset()
        novel_pose_loader = torch.utils.data.DataLoader(novel_pose_dataset,
                                            batch_size = 1,
                                            shuffle = False,
                                            num_workers = 4,)

        render_path = os.path.join(avatarmodel.model_path, 'novel_pose', "ours_{}".format(epoch))
        makedirs(render_path, exist_ok=True)
        for idx, batch_data in enumerate(tqdm(novel_pose_loader, desc="Rendering progress")):
            batch_data = to_cuda(batch_data, device=torch.device('cuda:0'))

            if model.train_stage ==1:
                image, = avatarmodel.render_free_stage1(batch_data, 59400)
            else:
                image, = avatarmodel.render_free_stage2(batch_data, 59400)

            torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    network = NetworkParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--epoch", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)

    render_sets(model.extract(args), network.extract(args), op.extract(args), args.epoch,)