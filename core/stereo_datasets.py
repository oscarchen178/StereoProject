# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import logging
import os
import copy
import random
from pathlib import Path
from glob import glob
import os.path as osp
import pykitti
from core.utils import frame_utils
from core.utils.augmentor import FlowAugmentor, SparseFlowAugmentor, TemporalFlowAugmentor, TemporalSparseFlowAugmentor


class StereoDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, reader=None, temporal=False, frame_sample_length=4, is_test=False, ddp=False, load_flow=False,index_by_scene=False):
        self.augmentor = None
        self.index_by_scene = index_by_scene
        self.sparse = sparse
        self.temporal = temporal
        self.img_pad = aug_params.pop("img_pad", None) if aug_params is not None else None
        self.device = aug_params.pop("device", None) if aug_params is not None else None
        self.ddp = ddp
        self.load_flow = load_flow
        if aug_params is not None and "crop_size" in aug_params:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params) if not temporal else TemporalSparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params) if not temporal else TemporalFlowAugmentor(**aug_params)

        if reader is None:
            self.disparity_reader = frame_utils.read_gen
        else:
            self.disparity_reader = reader

        self.is_test = is_test
        self.init_seed = False
        self.flow_list = []
        self.pose_list = []
        self.frame_sample_length = frame_sample_length
        self.intrinsic_K = None
        self.baseline = None
        self.disparity_list = []
        self.image_list = []
        self.extra_info = []

    def __getitem__(self, index):
        # set seed
        if not self.init_seed and not self.is_test:
            # worker_num = int(os.environ.get('SLURM_CPUS_PER_TASK', 6)) - 2
            worker_num = 4
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                id = worker_info.id + worker_num * self.device if self.ddp else worker_info.id
                print(f"worker_info.id:{worker_info.id},worker_num:{worker_num},self.device:{self.device},id:{id}")
                torch.manual_seed(id)
                np.random.seed(id)
                random.seed(id)
                self.init_seed = True

        if self.temporal:  # temporal loader: load data by image pair sequences

            # sample a sequence
            if self.index_by_scene:  # first, index by scene (or sequences), then index by frame slice
                # sequence path
                index = index % len(self.image_list)
                image1_list = self.image_list[index][0]
                image2_list = self.image_list[index][1]
                pose_list = self.pose_list[index]
                disp_list = self.disparity_list[index]
                # assert len(image1_list) == len(image2_list) == len(disp_list), [len(image1_list), len(image2_list), len(disp_list)]
                if self.is_test or self.augmentor is None:
                    if self.load_flow:
                        flow_list = self.flow_list[index]
                        assert len(image1_list) == len(flow_list), [len(image1_list), len(flow_list)]
                        return image1_list, image2_list, disp_list, pose_list, flow_list
                    else:
                        return image1_list, image2_list, disp_list, pose_list  # read image pairs online

                # sequences slicing
                frame_length = len(image1_list)
                low = np.random.randint(0, frame_length - self.frame_sample_length)
                high = low + self.frame_sample_length
                image1_list = image1_list[low:high]
                image2_list = image2_list[low:high]
                T_seq = np.stack(pose_list[low:high], axis=0).astype(np.float32)  # n,4,4
                disp_list = disp_list[low:high]
            else:  # index by frame slice; the image_list is already sliced, the sliced sequences from different videos are concatenated
                index = index % len(self.image_list)
                image1_list = self.image_list[index][0]
                image2_list = self.image_list[index][1]
                pose_list = self.pose_list[index]  # n,4,4
                T_seq = np.stack(pose_list, axis=0).astype(np.float32)  # n,4,4
                disp_list = self.disparity_list[index]
                assert len(image1_list) == len(image2_list) == len(disp_list), [len(image1_list), len(image2_list), len(disp_list)]
                assert not self.is_test and self.augmentor is not None, "test mode should set index_by_scene=True"

            # read, process and convert to tensor
            left_seq = []
            right_seq = []
            flow_seq = []
            valid_seq = []
            for (img1_path, img2_path, disp_path) in zip(image1_list, image2_list, disp_list):
                disp = self.disparity_reader(disp_path)
                if isinstance(disp, tuple):
                    disp, valid = disp  # h,w
                else:
                    valid = disp < 512
                img1 = frame_utils.read_gen(img1_path)
                img2 = frame_utils.read_gen(img2_path)

                img1 = torch.from_numpy(np.array(img1).astype(np.uint8)).permute(2, 0, 1).cuda(self.device)
                img2 = torch.from_numpy(np.array(img2).astype(np.uint8)).permute(2, 0, 1).cuda(self.device)

                disp = np.array(disp).astype(np.float32)
                flow = np.stack([-disp, np.zeros_like(disp)], axis=-1)  # h,w,2
                flow = torch.from_numpy(flow).permute(2, 0, 1).float().cuda(self.device)  # 2,h,w
                valid = torch.from_numpy(valid)[None].float().cuda(self.device)

                # for grayscale images
                if len(img1.shape) == 2:
                    img1 = np.tile(img1[..., None], (1, 1, 3))
                    img2 = np.tile(img2[..., None], (1, 1, 3))
                else:
                    img1 = img1[:3]
                    img2 = img2[:3]

                left_seq.append(img1)
                right_seq.append(img2)
                flow_seq.append(flow)
                valid_seq.append(valid)

            # do augmentations
            if self.sparse:
                # if intrinsic_K is a list
                if self.intrinsic_K is not None and isinstance(self.intrinsic_K, list):
                    K = self.intrinsic_K[index]
                else:
                    K = self.intrinsic_K
                left_seq, right_seq, flow_seq, valid_seq, K = self.augmentor(left_seq, right_seq, flow_seq, valid_seq, torch.from_numpy(K.copy()).cuda(self.device))
            else:
                if self.intrinsic_K is not None and isinstance(self.intrinsic_K, list):
                    K = self.intrinsic_K[index]
                else:
                    K = self.intrinsic_K
                left_seq, right_seq, flow_seq, K = self.augmentor(left_seq, right_seq, flow_seq, torch.from_numpy(K.copy()).cuda(self.device))

            flow_seq = flow_seq[:, :1].float()
            K = K.float()
            T_seq = torch.from_numpy(T_seq).float().cuda(self.device)
            baseline = torch.tensor(self.baseline).float().cuda(self.device)  # 1

            if self.sparse:
                valid_seq = valid_seq.squeeze(1)  # n,h,w
            else:
                valid_seq = (flow_seq.abs() < 512).float().squeeze(1)

            return [image1_list[0], image1_list[-1]], left_seq, right_seq, flow_seq, valid_seq, T_seq, K, baseline

        else:  # single pair loader: load data by image pair
            if self.is_test:
                img1 = frame_utils.read_gen(self.image_list[index][0])
                img2 = frame_utils.read_gen(self.image_list[index][1])
                img1 = np.array(img1).astype(np.uint8)[..., :3]
                img2 = np.array(img2).astype(np.uint8)[..., :3]
                img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
                img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
                return img1, img2, self.extra_info[index]

            index = index % len(self.image_list)
            disp = self.disparity_reader(self.disparity_list[index])
            if isinstance(disp, tuple):
                disp, valid = disp
            else:
                valid = disp < 512

            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])

            img1 = np.array(img1).astype(np.uint8)
            img2 = np.array(img2).astype(np.uint8)

            disp = np.array(disp).astype(np.float32)
            flow = np.stack([-disp, np.zeros_like(disp)], axis=-1)

            # grayscale images
            if len(img1.shape) == 2:
                img1 = np.tile(img1[..., None], (1, 1, 3))
                img2 = np.tile(img2[..., None], (1, 1, 3))
            else:
                img1 = img1[..., :3]
                img2 = img2[..., :3]

            if self.augmentor is not None:
                if self.sparse:
                    img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
                else:
                    img1, img2, flow = self.augmentor(img1, img2, flow)

            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            flow = torch.from_numpy(flow).permute(2, 0, 1).float()

            if self.sparse:
                valid = torch.from_numpy(valid)
            else:
                valid = (flow[0].abs() < 512) & (flow[1].abs() < 512)  # h , w

            flow = flow[:1]
            return self.image_list[index] + [self.disparity_list[index]], img1, img2, flow, valid.float()

    def __mul__(self, v):
        copy_of_self = copy.deepcopy(self)
        copy_of_self.flow_list = v * copy_of_self.flow_list
        copy_of_self.image_list = v * copy_of_self.image_list
        copy_of_self.pose_list = v * copy_of_self.pose_list
        copy_of_self.disparity_list = v * copy_of_self.disparity_list
        copy_of_self.extra_info = v * copy_of_self.extra_info
        if self.intrinsic_K is not None and isinstance(self.intrinsic_K, list):
            copy_of_self.intrinsic_K = v * copy_of_self.intrinsic_K
        return copy_of_self

    def __len__(self):
        return len(self.image_list)



class TartanAir(StereoDataset):
    def __init__(self, aug_params=None, root='datasets', scene_list=[], test_keywords=[], is_test=False, mode='single_frame', frame_sample_length=4,ddp=False, load_flow=False):
        super().__init__(aug_params, reader=frame_utils.readDispTartanAir, temporal=(mode == 'temporal'), frame_sample_length=frame_sample_length,
                         is_test=is_test, ddp=ddp, load_flow=load_flow,index_by_scene=True)
        assert os.path.exists(root)
        assert mode in ['single_frame', 'temporal']
        if mode == 'single_frame':
            image1_list = sorted(glob(os.path.join(root, 'TartanAir/**/**/**/**/image_left/*_left.png')))
            image2_list = sorted(glob(os.path.join(root, 'TartanAir/**/**/**/**/image_right/*_right.png')))
            disp_list = sorted(glob(os.path.join(root, 'TartanAir/**/**/**/**/depth_left/*_left_depth.npy')))
            if is_test:
                _, image1_list = self.split_train_valid(image1_list, test_keywords)
                _, image2_list = self.split_train_valid(image2_list, test_keywords)
                _, disp_list = self.split_train_valid(disp_list, test_keywords)
            else:
                image1_list, _ = self.split_train_valid(image1_list, test_keywords)
                image2_list, _ = self.split_train_valid(image2_list, test_keywords)
                disp_list, _ = self.split_train_valid(disp_list, test_keywords)

            for img1, img2, disp in zip(image1_list, image2_list, disp_list):
                self.image_list += [[img1, img2]]
                self.disparity_list += [disp]
        elif mode == 'temporal':
            if load_flow:
                assert is_test, 'flow is only available in test mode'

            # ablation study, use all scenes and Hard and Easy
            frames_list = sorted(glob(os.path.join(root, 'TartanAir/**/**/**/P*')))

            image1_list = []
            image2_list = []
            disp_list = []
            pose_list = []
            flow_list = []

            if is_test:
                _, frames_list = self.split_train_valid(frames_list, test_keywords)
            else:
                frames_list, _ = self.split_train_valid(frames_list, test_keywords)

            for x in frames_list:
                disp_frames = sorted(glob(os.path.join(x, 'depth_left/*_left_depth.npy')))
                left_frames = sorted(glob(os.path.join(x, 'image_left/*_left.png')))
                right_frames = sorted(glob(os.path.join(x, 'image_right/*_right.png')))
                pose_frames = frame_utils.read_tartanair_extrinsic(os.path.join(x, 'pose_left.txt'), 'left')
                if load_flow:
                    flow_frames = sorted(glob(os.path.join(x.replace('TartanAir', 'TartanAir_flow'), 'flow/*_*_flow.npy')))
                    # add a fake flow for the last frame
                    flow_frames.append(flow_frames[-1])

                if is_test:  # no augmentation for test
                    argument_rate = 1
                else:  # augment the data to keep the same sampling probability for each scene (or video).
                    frame_len = len(disp_frames)
                    argument_rate = frame_len//300
                assert argument_rate>=1,['please check 300']
                for _ in range(argument_rate):
                    disp_list.append(disp_frames)
                    image1_list.append(left_frames)
                    image2_list.append(right_frames)
                    pose_list.append(pose_frames)
                    if load_flow:
                        flow_list.append(flow_frames)

            for img1, img2, disp, pose in zip(image1_list, image2_list, disp_list, pose_list):
                self.image_list += [[img1, img2]]  # (frames,2,images)
                self.disparity_list += [disp]
                self.pose_list += [pose]  # (frames,images) T
            if load_flow:
                self.flow_list = flow_list
            self.intrinsic_K = np.array([[320.0, 0, 320.0],
                                         [0, 320.0, 240.0],
                                         [0, 0, 1]])
            self.baseline = 0.25

    def split_train_valid(self, path_list, valid_keywords):
        path_list_init = path_list
        for kw in valid_keywords:
            path_list = list(filter(lambda s: kw not in s, path_list))
        train_path_list = sorted(path_list)
        valid_path_list = sorted(list(set(path_list_init) - set(train_path_list)))
        return train_path_list, valid_path_list


class KITTI(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/KITTI', is_test=False, mode='single_frame', frame_sample_length=4, image_set='training',ddp=False, index_by_scene=False, num_frames=11):
        super(KITTI, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispKITTI, temporal=(mode == 'temporal'), frame_sample_length=frame_sample_length,
                                    is_test=is_test, ddp=ddp, load_flow=False, index_by_scene=index_by_scene)
        assert os.path.exists(root)
        if is_test:
            if mode == 'single_frame':
                raise NotImplementedError
            else:  # temporal
                num_frames = num_frames
                scene_list = sorted(glob(os.path.join(root, image_set, 'sequences', '**')))  # imageset in 'kitti_seq/kitti2012_testings', 'kitti_seq/kitti2015_testings'
                image1_list = []
                image2_list = []
                pose_list = []
                disp_list = []
                for scene in scene_list:
                    image1_list.append(sorted(glob(os.path.join(scene, 'image_2', '*.png')))[:num_frames])
                    image2_list.append(sorted(glob(os.path.join(scene, 'image_3', '*.png')))[:num_frames])
                    pose_path = os.path.join(scene, 'orbslam3_pose.txt')
                    pose_list.append(frame_utils.read_kitti_extrinsic(pose_path)[:num_frames])
                    disp_list.append(scene)  # a fake disp_list, pass the scene path
                for idx, (img1, img2, disp, pose) in enumerate(zip(image1_list, image2_list, disp_list, pose_list)):
                    self.image_list += [[img1, img2]]  # scenes , 2, frames
                    self.disparity_list += [disp]  # scene path
                    self.pose_list += [pose]   # scenes, frames, numpy(4*4)

        else:
            if mode=='single_frame':
                image1_list = sorted(glob(os.path.join(root, 'Kitti15', image_set, 'image_2/*_10.png')))
                image2_list = sorted(glob(os.path.join(root, 'Kitti15', image_set, 'image_3/*_10.png')))
                disp_list = sorted(glob(
                    os.path.join(root, 'Kitti15', 'training', 'disp_occ_0/*_10.png'))) if image_set == 'training' else \
                    [osp.join(root, 'training/disp_occ_0/000085_10.png')] * len(image1_list)
                image1_list += sorted(glob(os.path.join(root, 'Kitti12', image_set, 'image_0/*_10.png')))
                image2_list += sorted(glob(os.path.join(root, 'Kitti12', image_set, 'image_1/*_10.png')))
                disp_list += sorted(
                    glob(os.path.join(root, 'Kitti12', 'training', 'disp_occ/*_10.png'))) if image_set == 'training' else \
                    [osp.join(root, 'training/disp_occ_0/000085_10.png')] * len(image1_list)

                for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
                    self.image_list += [[img1, img2]]
                    self.disparity_list += [disp]
            else:
                raise NotImplementedError


class KITTIraw(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/kitti_raw', is_test=False, mode='single_frame', frame_sample_length=4, ddp=False):
        super(KITTIraw, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispKITTI, temporal=(mode == 'temporal'), frame_sample_length=frame_sample_length,
                         is_test=is_test, ddp=ddp, load_flow=False)
        assert os.path.exists(root)
        scenes_list = sorted(glob(os.path.join(root, '**')))
        image1_list = []
        image2_list = []
        disp_list = []
        intrinsic_list = []
        pose_list = []
        if is_test:  # index by scene
            for scene in scenes_list:
                intrinsic_path = os.path.join(scene, scene.split('/')[-1], '.txt')
                image1_list.append(sorted(glob(os.path.join(scene, 'image_2/*.png'))))
                image2_list.append(sorted(glob(os.path.join(scene, 'image_3/*.png'))))
                intrinsic_list.append(intrinsic_path)
                pose_list.append(frame_utils.read_kitti_extrinsic(os.path.join(scene, 'orbslam3_pose.txt')))
                disp_list.append([x.replace('image_2/data/', 'leastereo/data/') for x in image1_list[-1]])  # fake path
        else:  # index by image slice
            for scene in scenes_list:  # date
                intrinsic_path = os.path.join(scene, 'calib_cam_to_cam.txt')
                seqs_list = sorted(glob(os.path.join(scene, '*_sync')))
                for seq in seqs_list:  # sync
                    img1_seq = sorted(glob(os.path.join(seq, 'image_02/data/*.png')))
                    img2_seq = sorted(glob(os.path.join(seq, 'image_03/data/*.png')))
                    disp_seq = sorted(glob(os.path.join(seq, 'leastereo/data/*.png')))
                    pose_seq = frame_utils.read_kitti_extrinsic(os.path.join(seq, 'pose.txt'))  # n 4,4
                    if len(img1_seq) != len(disp_seq) or len(img1_seq) != len(img2_seq) or len(img1_seq) != len(pose_seq):
                        print(f"Warning: {seq} has different length of images, disparity or pose")
                        continue
                    intrinsic_list += [intrinsic_path]*len(img1_seq)
                    # assert len(pose_seq) == len(img1_seq) == len(disp_seq), [len(pose_seq), len(img1_seq),len(disp_seq),seq]
                    img1_seq_slices = [img1_seq[i:i + frame_sample_length] for i in range(len(img1_seq) - frame_sample_length + 1)]
                    img2_seq_slices = [img2_seq[i:i + frame_sample_length] for i in range(len(img2_seq) - frame_sample_length + 1)]
                    disp_seq_slices = [disp_seq[i:i + frame_sample_length] for i in range(len(disp_seq) - frame_sample_length + 1)]
                    pose_seq_slices = [pose_seq[i:i + frame_sample_length] for i in range(len(pose_seq) - frame_sample_length + 1)]
                    image1_list += img1_seq_slices  # (slices, frame length)
                    image2_list += img2_seq_slices  # (slices, frame length)
                    disp_list += disp_seq_slices  # (slices, frame length)
                    pose_list += pose_seq_slices  # (slices, frame length)

        self.intrinsic_K = []
        for idx, (img1, img2, disp, pose) in enumerate(zip(image1_list, image2_list, disp_list, pose_list)):
            self.image_list += [[img1, img2]]  # (slices, 2, frame length)
            self.disparity_list += [disp]  # (slices, frame length)
            self.pose_list += [pose]  # (slices, frame length)
            Pr2 = pykitti.utils.read_calib_file(intrinsic_list[idx])['P_rect_02']
            self.intrinsic_K += [np.array([[Pr2[0], 0,  Pr2[2]],
                                         [0, Pr2[5], Pr2[6]],
                                         [0,      0,     1]])]
        self.baseline = 0.54



def fetch_dataloader(args):
    """ Create the data loader for the corresponding trainign set """

    aug_params = {'crop_size': args.image_size, 'min_scale': args.spatial_scale[0], 'max_scale': args.spatial_scale[1], 'do_flip': False,
                  'yjitter': not args.noyjitter}
    if hasattr(args, "saturation_range") and args.saturation_range is not None:
        aug_params["saturation_range"] = args.saturation_range
    if hasattr(args, "img_gamma") and args.img_gamma is not None:
        aug_params["gamma"] = args.img_gamma
    if hasattr(args, "do_flip") and args.do_flip is not None:
        aug_params["do_flip"] = args.do_flip
    if hasattr(args, "device") and args.device is not None:
        aug_params["device"] = args.device

    train_dataset = None
    dataset_name = args.train_dataset
    if dataset_name == 'kitti_raw':
        new_dataset = KITTIraw(aug_params,
                               mode='temporal' if args.temporal else 'single_frame',
                               frame_sample_length=args.frame_length,
                               ddp=args.ddp)
        logging.info(f"Adding {len(new_dataset)} samples from KITTI raw")
    elif 'kitti' in dataset_name:
        new_dataset = KITTI(aug_params,
                            mode='temporal' if args.temporal else 'single_frame',
                            frame_sample_length=args.frame_length,
                            ddp=args.ddp)
        logging.info(f"Adding {len(new_dataset)} samples from KITTI")
    elif dataset_name == 'TartanAir':
        keyword_list = []
        scene_list = ['abandonedfactory', 'amusement', 'carwelding', 'endofworld', 'gascola', 'hospital', 'office', 'office2',
                      'oldtown', 'soulcity']
        part_list = ['P002', 'P007', 'P003', 'P006', 'P001', 'P042', 'P006', 'P004', 'P006', 'P008']

        for i, (s, p) in enumerate(zip(scene_list, part_list)):
            keyword_list.append(os.path.join(s, 'Easy', p))  # temporal stereo off
            keyword_list.append(os.path.join(s, 'Hard', p))
        if args.temporal:
            scale_factor = 100
        else:
            scale_factor = 1
        root = 'datasets'
        new_dataset = TartanAir(aug_params, root=root, scene_list=scene_list, test_keywords=keyword_list, mode='temporal' if args.temporal else 'single_frame',
                                frame_sample_length=args.frame_length, ddp=args.ddp) * scale_factor
        logging.info(f"Adding {len(new_dataset)} samples from TartanAir")
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")
    train_dataset = new_dataset

    if args.ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            shuffle=True,
        )
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                       pin_memory=False, num_workers=4, prefetch_factor=4, drop_last=True,
                                       sampler=train_sampler, persistent_workers=True)
    else:
        # train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
        #                                pin_memory=False, num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', 6)) - 2, prefetch_factor=4, drop_last=True,
        #                                shuffle=False)
        train_loader = data.DataLoader(train_dataset, batch_size=4,
                                       pin_memory=False, num_workers=0, prefetch_factor=None, drop_last=True,
                                       shuffle=True)

    logging.info('Training with %d image pairs' % len(train_dataset))
    return train_loader
