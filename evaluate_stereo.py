from __future__ import print_function, division
import sys

sys.path.append('core')
import os
import wandb
import argparse
import time
import skimage
import logging
import numpy as np
import torch
from tqdm import tqdm
from core.tc_stereo import TCStereo, autocast
from core.new_stereo import NewStereo
import core.stereo_datasets as datasets
from core.utils.utils import InputPadder
from core.utils.frame_utils import readDispTartanAir, read_gen
import cv2
import pykitti
from core.utils.visualization import pseudoColorMap


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def submit_kitti(args, model, iters=32, mixed_prec=False):
    """ Peform submission using the KITTI-2015 (seq test) split """
    model.eval()
    aug_params = {}
    submission = True
    imageset = 'kitti_seq/kitti2015_testings'
    P = 'P_rect_02'

    val_dataset = datasets.KITTI(aug_params,
                                 is_test=True,
                                 mode='temporal',
                                 image_set=imageset,
                                 index_by_scene=True,
                                 num_frames=11 if submission else 21)
    torch.backends.cudnn.benchmark = True
    params = dict()
    flow_q = None
    fmap1 = None
    previous_T = None
    net_list = None
    baseline = torch.tensor(0.54).float().cuda()[None]
    out_list, epe_list, elapsed_list = [], [], []

    def load(args, image1, image2, T):
        # load image & disparity
        image1 = read_gen(image1)
        image2 = read_gen(image2)
        image1 = np.array(image1)
        image2 = np.array(image2)
        image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
        image2 = torch.from_numpy(image2).permute(2, 0, 1).float()
        T = torch.from_numpy(T).float()
        T = T[None].cuda()
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        return image1, image2, T

    for val_id in tqdm(range(len(val_dataset))):
        image1_list, image2_list, scene_path, pose_list = val_dataset[val_id]
        scene_name   = os.path.basename(scene_path)
        Pr2 = pykitti.utils.read_calib_file(os.path.join(scene_path, scene_name + '.txt'))[P]
        K = np.array([[Pr2[0], 0, Pr2[2]],
                      [0, Pr2[5], Pr2[6]],
                      [0, 0, 1]])
        K_raw = torch.from_numpy(K).float().cuda()[None]
        for frame_ind, (image1, image2, T) in tqdm(enumerate(zip(image1_list, image2_list, pose_list))):
            image1, image2, T = load(args, image1, image2, T)
            padder = InputPadder(image1.shape, divis_by=32)
            imgs, K = padder.pad(image1, image2, K=K_raw)
            image1, image2 = imgs
            params.update({'K': K,
                           'T': T,
                           'previous_T': previous_T,
                           'last_disp': flow_q,
                           'last_net_list': net_list,
                           'fmap1': fmap1,
                           'baseline': baseline})
            with autocast(enabled=mixed_prec):
                start = time.time()
                testing_output = model(image1, image2, iters=iters, test_mode=True, params=params if (flow_q is not None) and args.temporal else None)
                end = time.time()
            if val_id > 50 and frame_ind > 6:
                elapsed_list.append(end - start)
            disp_pr = -testing_output['flow']
            flow_q = testing_output['flow_q']
            net_list = testing_output['net_list']
            fmap1 = testing_output['fmap1']
            previous_T = T
            disp_pr, K = padder.unpad(disp_pr, K)  # 1,1,h,w
            # save
            if submission:
                if frame_ind == 10:
                    disp_pr = disp_pr.squeeze(0).detach().cpu().numpy()  # 1,h,w
                    submit_dir = os.path.join('./kitti_15_seq_out', 'disp_0')
                    os.makedirs(submit_dir, exist_ok=True)
                    skimage.io.imsave(os.path.join(submit_dir, scene_name + '_10.png'), (disp_pr * 256).astype('uint16'))
            else:  # output as rgb video visualization
                disp_pr = disp_pr[0, 0].detach().cpu().numpy()  # 1,h,w
                disp_pr = pseudoColorMap(disp_pr, vmin=0, vmax=96, kitti_style=True)
                if frame_ind == 0:
                    video_dir = os.path.join('./kitti_15_seq_out', 'video')
                    os.makedirs(video_dir, exist_ok=True)
                    video_path = os.path.join(video_dir, scene_name + '.avi')
                    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'MJPG'), 2, (disp_pr.shape[1], disp_pr.shape[0]))  # 2fps
                video.write(disp_pr)
        if not submission:
            video.release()
    avg_runtime = np.mean(elapsed_list)
    print(f"Submission KITTI: {format(1 / (avg_runtime + 1e-5), '.2f')}-FPS ({format(avg_runtime, '.3f')}s)")
    return {'kitti-fps': 1 / (avg_runtime + 1e-5)}


@torch.no_grad()
def validate_tartanair(args, model, iters=32, mixed_prec=False):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    aug_params = {}
    # test set
    keyword_list = []
    scene_list = ['abandonedfactory', 'abandonedfactory_night']
    part_list = ['P008', 'P009', 'P010', 'P011', 'P012', 'P013', 'P014']

    for i, (s, p) in enumerate(zip(scene_list, part_list)):
        keyword_list.append(os.path.join(s, 'Easy', p))
        keyword_list.append(os.path.join(s, 'Hard', p))

    val_dataset = datasets.TartanAir(aug_params, root='datasets', scene_list=scene_list, test_keywords=keyword_list,
                                     is_test=True, mode='temporal', load_flow=False)

    # camera parameters
    K = np.array([[320.0, 0, 320.0],
                  [0, 320.0, 240.0],
                  [0, 0, 1]])
    K_raw = torch.from_numpy(K).float().cuda()[None]
    baseline = torch.tensor(0.25).float().cuda()[None]

    # Evaluate Metrics list
    out_list, out3_list, epe_list = [], [], []

    # load function
    def load(args, image1, image2, disp_gt, T):
        # load image & disparity
        image1 = read_gen(image1)
        image2 = read_gen(image2)
        image1 = np.array(image1)
        image2 = np.array(image2)
        disp_gt = readDispTartanAir(disp_gt)
        disp_gt = torch.from_numpy(np.array(disp_gt).astype(np.float32))[:1]
        image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
        image2 = torch.from_numpy(image2).permute(2, 0, 1).float()
        T = torch.from_numpy(T).float()

        T = T[None].cuda()
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        disp_gt = disp_gt[None].cuda()  # 1,1,h,w
        return image1, image2, disp_gt, T

    # Testing
    for val_id in range(len(val_dataset)):
        image1_list, image2_list, flow_gt_list, pose_list = val_dataset[val_id]
        # temporal parameters
        params = dict()
        flow_q = None
        fmap1 = None
        previous_T = None
        net_list = None

        length = len(image1_list)
        for j, (image1, image2, disp_gt, T) in tqdm(enumerate(zip(image1_list, image2_list, flow_gt_list, pose_list)), total=length):
            # load
            image1, image2, disp_gt, T = load(args, image1, image2, disp_gt, T)
            padder = InputPadder(image1.shape, divis_by=32)
            imgs, K = padder.pad(image1, image2, K=K_raw)
            image1, image2 = imgs
            params.update({'K': K,
                           'T': T,
                           'previous_T': previous_T,
                           'last_disp': flow_q,
                           'last_net_list': net_list,
                           'fmap1': fmap1,
                           'baseline': baseline})

            with autocast(enabled=mixed_prec):
                testing_output = model(image1, image2, iters=iters, test_mode=True, params=params if (flow_q is not None) and args.temporal else None)

            disp_pr = -testing_output['flow']
            flow_q = testing_output['flow_q']
            net_list = testing_output['net_list']
            fmap1 = testing_output['fmap1']
            previous_T = T
            disp_pr, K = padder.unpad(disp_pr, K=K)

            # epe evaluation
            assert disp_pr.shape == disp_gt.shape, (disp_pr.shape, disp_gt.shape)
            epe = torch.sum((disp_pr.squeeze(0) - disp_gt.squeeze(0)) ** 2, dim=0).sqrt()

            epe = epe.flatten()
            val = (disp_gt.squeeze(0).abs().flatten() < 192)
            if (val == False).all():
                continue
            out = (epe > 1.0).float()[val].mean().cpu().item()
            out3 = (epe > 3.0).float()[val].mean().cpu().item()
            mask_rate = val.float().mean().cpu().item()
            epe_list.append(epe[val].mean().cpu().item())
            out_list.append(np.array([out * mask_rate, mask_rate]))
            out3_list.append(np.array([out3 * mask_rate, mask_rate]))
    epe_list = np.array(epe_list)
    out_list = np.stack(out_list, axis=0)
    out3_list = np.stack(out3_list, axis=0)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list[:, 0]) / np.mean(out_list[:, 1])
    d3 = 100 * np.mean(out3_list[:, 0]) / np.mean(out3_list[:, 1])

    print("Validation TartanAir: EPE %f, D1 %f, D3 %f" % (epe, d1, d3))
    return {'TartanAir-epe': epe, 'TartanAir-d1': d1, 'TartanAir-d3': d3}



if __name__ == '__main__':
    import os
    import psutil
    pid = os.getpid()
    # process = psutil.Process(pid)
    # process.nice(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default=None)
    parser.add_argument('--dataset', help="dataset for evaluation", required=True,
                        choices=["kitti", "things", "TartanAir"])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
    parser.add_argument('--init_thres', type=float, default=0.5, help="the threshold gap of contrastive loss for cost volume.")
    parser.add_argument('--visualize', action='store_true', help='visualize the results')
    parser.add_argument('--device', default=0, type=int, help='the device id')

    # Architecure choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128] * 3, help="hidden state and context dimensions")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--temporal', action='store_true', help="temporal mode")  # TODO: MODEL temporal mode

    # Use TC+DEFOM model
    parser.add_argument('--use_defom', action='store_true', help='build NewStereo (TC-Stereo integrated with DEFOM) instead of the original TC-Stereo')
    parser.add_argument('--scale_iters', type=int, default=8)
    parser.add_argument('--scale_list', type=float, nargs='+', default=[0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    parser.add_argument('--scale_corr_radius', type=int, default=2)
    parser.add_argument('--defom_variant', type=str, default='vits', choices=['vits', 'vitl'], help='DINOv2 variant')
    
    args = parser.parse_args()

    # if args.visualize:
    wandb.init(
        job_type="test",
        project="vis",
        entity="oscar17chen-university-of-toronto"
    )
    # add the args to wandb
    wandb.config.update(args)

    ###  Choose model
    if args.use_defom:
        model = NewStereo(args)
    else:
        model = TCStereo(args)

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        model.load_state_dict(checkpoint['model'], strict=True)
        logging.info(f"Done loading checkpoint")

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    model.cuda()
    model.eval()

    print(f"The model has {format(count_parameters(model) / 1e6, '.2f')}M learnable parameters.")

    use_mixed_precision = False

    if args.dataset == 'kitti':
        submit_kitti(args, model, iters=args.valid_iters, mixed_prec=use_mixed_precision)

    elif args.dataset == 'TartanAir':
        validate_tartanair(args, model, iters=args.valid_iters, mixed_prec=use_mixed_precision)
