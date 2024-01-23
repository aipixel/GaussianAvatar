import torch
import os
from tqdm import tqdm
from os import makedirs
import torch.nn as nn
import torchvision
import numpy as np
from utils.general_utils import safe_state, to_cuda
from argparse import ArgumentParser
from arguments import ModelParams, get_combined_args, NetworkParams, OptimizationParams
from model.avatar_model import  AvatarModel
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from torch.cuda.amp import custom_fwd
class Evaluator(nn.Module):
    """adapted from https://github.com/JanaldoChen/Anim-NeRF/blob/main/models/evaluator.py"""
    def __init__(self):
        super().__init__()
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex")
        self.psnr = PeakSignalNoiseRatio(data_range=1)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1)

    # custom_fwd: turn off mixed precision to avoid numerical instability during evaluation
    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, rgb, rgb_gt):
        # torchmetrics assumes NCHW format
        # rgb = rgb.permute(0, 3, 1, 2).clamp(max=1.0)
        # rgb_gt = rgb_gt.permute(0, 3, 1, 2)

        return {
            "psnr": self.psnr(rgb, rgb_gt),
            "ssim": self.ssim(rgb, rgb_gt),
            "lpips": self.lpips(rgb, rgb_gt),
        }

def render_sets(model, net, opt, epoch:int):

    evaluator = Evaluator()
    evaluator = evaluator.cuda()
    evaluator.eval()
    with torch.no_grad():
        avatarmodel = AvatarModel(model, net, opt, train=False)
        avatarmodel.training_setup()
        avatarmodel.load(epoch, test=False)
        
        test_dataset = avatarmodel.getTestDataset()
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size = 1,
                                            shuffle = False,
                                            num_workers = 4,)

        render_path = os.path.join(avatarmodel.model_path, 'test_free', "ours_{}".format(epoch))
        gt_path  = os.path.join(avatarmodel.model_path, 'test_free', 'gt_image')

        makedirs(render_path, exist_ok=True)
        makedirs(gt_path, exist_ok=True)
        results = []

        for idx, batch_data in enumerate(tqdm(test_loader, desc="Rendering progress")):
            batch_data = to_cuda(batch_data, device=torch.device('cuda:0'))
            gt_image = batch_data['original_image']

            image, = avatarmodel.render_free_stage1(batch_data, 59400)
            results.append(evaluator(image.unsqueeze(0), gt_image))
              
            torchvision.utils.save_image(gt_image, os.path.join(gt_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

        with open("results.txt", "w") as f:
            psnr = torch.stack([r['psnr'] for r in results]).mean().item()
            print(f"PSNR: {psnr:.2f}")
            f.write(f"PSNR: {psnr:.2f}\n")

            ssim = torch.stack([r['ssim'] for r in results]).mean().item()
            print(f"SSIM: {ssim:.4f}")
            f.write(f"SSIM: {ssim:.4f}\n")

            lpips = torch.stack([r['lpips'] for r in results]).mean().item()
            print(f"LPIPS: {lpips:.4f}")
            f.write(f"LPIPS: {lpips:.4f}\n")        

        print('save video...')

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    network = NetworkParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--epoch", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_epochs", nargs="+", type=int, default=[])
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)

    render_sets(model.extract(args), network.extract(args), op.extract(args), args.epoch,)