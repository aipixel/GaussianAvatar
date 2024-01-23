import os
import torch
import lpips
import torchvision
import open3d as o3d
import sys
import uuid
from tqdm import tqdm
from utils.loss_utils import l1_loss_w, ssim
from utils.general_utils import safe_state
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, OptimizationParams, NetworkParams
from model.avatar_model import AvatarModel
from utils.general_utils import to_cuda, adjust_loss_weights

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def train(model, net, opt, saving_epochs, checkpoint_epochs):
    tb_writer = prepare_output_and_logger(model)
    avatarmodel = AvatarModel(model, net, opt, train=True)
    
    loss_fn_vgg = lpips.LPIPS(net='alex').cuda()
    train_loader = avatarmodel.getTrainDataloader()
    
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    first_iter = 0
    epoch_start = 0
    data_length = len(train_loader)
    avatarmodel.training_setup()

    if checkpoint_epochs:
        avatarmodel.load(checkpoint_epochs[0])
        epoch_start += checkpoint_epochs[0]
        first_iter += epoch_start * data_length

    if model.train_stage == 2:
        avatarmodel.stage_load(model.stage1_out_path)
    
    progress_bar = tqdm(range(first_iter, data_length * opt.epochs), desc="Training progress")
    ema_loss_for_log = 0.0
    
    for epoch in range(epoch_start + 1, opt.epochs + 1):

        if model.train_stage ==1:
            avatarmodel.net.train()
            avatarmodel.pose.train()
            avatarmodel.transl.train()
        else:
            avatarmodel.net.train()
            avatarmodel.pose.eval()
            avatarmodel.transl.eval()
            avatarmodel.pose_encoder.train()
        
        iter_start.record()

        wdecay_rgl = adjust_loss_weights(opt.lambda_rgl, epoch, mode='decay', start=epoch_start, every=20)

        for _, batch_data in enumerate(train_loader):
            
            first_iter += 1
            batch_data = to_cuda(batch_data, device=torch.device('cuda:0'))
            gt_image = batch_data['original_image']

            if model.train_stage ==1:
                image, points, offset_loss, geo_loss, scale_loss = avatarmodel.train_stage1(batch_data, first_iter)
                scale_loss = opt.lambda_scale  * scale_loss
                offset_loss = wdecay_rgl * offset_loss
                
                Ll1 = (1.0 - opt.lambda_dssim) * l1_loss_w(image, gt_image)
                ssim_loss = opt.lambda_dssim * (1.0 - ssim(image, gt_image)) 

                loss = scale_loss + offset_loss + Ll1 + ssim_loss + geo_loss
            else:
                image, points, pose_loss, offset_loss, = avatarmodel.train_stage2(batch_data, first_iter)

                offset_loss = wdecay_rgl * offset_loss
                
                Ll1 = (1.0 - opt.lambda_dssim) * l1_loss_w(image, gt_image)
                ssim_loss = opt.lambda_dssim * (1.0 - ssim(image, gt_image)) 

                loss =  offset_loss + Ll1 + ssim_loss + pose_loss * 10


            if epoch > opt.lpips_start_iter:
                vgg_loss = opt.lambda_lpips * loss_fn_vgg((image-0.5)*2, (gt_image- 0.5)*2).mean()
                loss = loss + vgg_loss
            
            avatarmodel.zero_grad(epoch)

            loss.backward(retain_graph=True)
            iter_end.record()
            avatarmodel.step(epoch)

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if first_iter % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)

                if (first_iter-1) % opt.log_iter == 0:
                    save_poitns = points.clone().detach().cpu().numpy()
                    for i in range(save_poitns.shape[0]):
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(save_poitns[i])
                        o3d.io.write_point_cloud(os.path.join(model.model_path, 'log',"pred_%d.ply" % i) , pcd)

                    torchvision.utils.save_image(image, os.path.join(model.model_path, 'log', '{0:05d}_pred'.format(first_iter) + ".png"))
                    torchvision.utils.save_image(gt_image, os.path.join(model.model_path, 'log', '{0:05d}_gt'.format(first_iter) + ".png"))
                    
            if tb_writer:
                tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), first_iter)
                tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), first_iter)
                tb_writer.add_scalar('train_loss_patches/scale_loss', scale_loss.item(), first_iter)
                tb_writer.add_scalar('train_loss_patches/offset_loss', offset_loss.item(), first_iter)
                # tb_writer.add_scalar('train_loss_patches/aiap_loss', aiap_loss.item(), first_iter)
                tb_writer.add_scalar('iter_time', iter_start.elapsed_time(iter_end), first_iter)
                if model.train_stage ==1:
                    tb_writer.add_scalar('train_loss_patches/geo_loss', geo_loss.item(), first_iter)
                else:
                    tb_writer.add_scalar('train_loss_patches/pose_loss', pose_loss.item(), first_iter)
                if epoch > opt.lpips_start_iter:
                    tb_writer.add_scalar('train_loss_patches/vgg_loss', vgg_loss.item(), first_iter)

        if (epoch > saving_epochs[0]) and epoch % model.save_epoch == 0:
            print("\n[Epoch {}] Saving Model".format(epoch))
            avatarmodel.save(epoch)


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)

    os.makedirs(os.path.join(args.model_path, 'log'), exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    np = NetworkParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--save_epochs", nargs="+", type=int, default=[100])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_epochs", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_epochs.append(args.epochs)
    
    print("Optimizing " + args.model_path)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    train(lp.extract(args), np.extract(args), op.extract(args), args.save_epochs, args.checkpoint_epochs)

    print("\nTraining complete.")
