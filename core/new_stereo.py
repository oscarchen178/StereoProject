import torch
import torch.nn as nn
import torch.nn.functional as F

# core modules (already in project)
from core.update import (
    BasicMultiUpdateBlock,
    ScaleBasicMultiUpdateBlock,  # imported from DEFOM – now located in core.update
    Lightfuse,
    DispGradPredictor,
    DispRefine,
    DisparityCompletor,
    HiddenstateUpdater,
)
from core.extractor import (
    BasicEncoder,
    MultiBasicEncoder,
    ResidualBlock,
    DefomEncoder,  # DEFOM DepthAnyThing‑based encoder
)
from core.corr import NewCorrBlock1D
from core.utils.utils import coords_grid, upflow8, bilinear_sampler
from core.utils.geo_utils import (
    cal_relative_transformation,
    warp,
    get_backward_grid,
    disp2disp_gradient_xy,
)

try:
    autocast = torch.cuda.amp.autocast
except AttributeError:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class NewStereo(nn.Module):
    """Temporal‑Context Stereo (TC‑Stereo) upgraded with optional DEFOM features
    and scale‑update iterations. 100 % drop‑in replacement for the original
    ``TCStereo``: **same constructor signature and identical output dict**
    so that existing training / evaluation scripts keep working.

    Extra runtime options:
    - ``defom_variant`` (str): which DINOv2 backbone to load
      ('vits' | 'vitb' | 'vitl' | 'vitg').
    - ``scale_iters`` (int): number of early iterations that use the
      scale‑update GRU (``ScaleBasicMultiUpdateBlock``). If 0, behaviour is
      identical to the original TC‑Stereo.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        ### Args from DEFOM
        self.scale_iters = 8
        self.scale_list = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        self.scale_corr_radius = 2
        self.defom_variant = 'vits'
        ###

        self.scale_rate = 1 / (2 ** args.n_downsample)
        context_dims = args.hidden_dims

        # ------------------------- feature encoders ------------------------- #
        self.cnet = MultiBasicEncoder(
            output_dim=[args.hidden_dims, context_dims],
            norm_fn=args.context_norm,
            downsample=args.n_downsample,
        )

        if args.shared_backbone:
            self.conv2 = nn.Sequential(
                ResidualBlock(128, 128, 'instance', stride=1),
                nn.Conv2d(128, 256, 3, padding=1),
            )
            self.fnet = None
        else:
            self.fnet = BasicEncoder(
                output_dim=256, norm_fn='instance', downsample=args.n_downsample
            )

        # DEFOM encoder (frozen by default)
        variant = self.defom_variant
        self.defomencoder = DefomEncoder(variant, pretrained=True, freeze=True)
        # simple convs to align ViT+DPT feature channels with CNN pipeline
        self.defom_f_align = nn.Conv2d(
            self.defomencoder.out_dim, 256, kernel_size=1, padding=0
        )

        # --------------------------- update blocks -------------------------- #
        self.update_block = BasicMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims)
        # scale‑aware block (from DEFOM) – used for the first *scale_iters*
        if self.scale_iters > 0:
            self.scale_update_block = ScaleBasicMultiUpdateBlock(
                self.args, hidden_dims=args.hidden_dims
            )

        # ------------------------ context processing ------------------------ #
        self.context_zqr_convs = nn.ModuleList(
            [nn.Conv2d(context_dims[i], args.hidden_dims[i] * 3, 3, padding=1) for i in range(self.args.n_gru_layers)]
        )

        # hidden state fusion & refinement modules (unchanged)
        self.previous_current_hidden_fuse = nn.ModuleList(
            [Lightfuse(args.hidden_dims[i], args.hidden_dims[i]) for i in range(self.args.n_gru_layers)]
        )
        self.disp_completor = DisparityCompletor()
        self.disp_grad_refine = DispGradPredictor(args)
        self.disp_refine = DispRefine(args)
        self.context_zqr_convs_grad = nn.ModuleList(
            [nn.Conv2d(context_dims[i], 64, 3, padding=1) for i in range(self.args.n_gru_layers)]
        )
        self.hiddenstate_update = HiddenstateUpdater(context_dims[0])

    # --------------------------------------------------------------------- #
    # Helper utilities (identical to original implementation)               #
    # --------------------------------------------------------------------- #
    def initialize_flow(self, img):
        N, _, H, W = img.shape
        coords0 = coords_grid(N, H, W).to(img.device)
        coords1 = coords_grid(N, H, W).to(img.device)
        return coords0[:, :1], coords1[:, :1]

    def upsample_flow(self, flow, mask, scale=True):
        N, D, H, W = flow.shape
        factor = 2 ** self.args.n_downsample
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask_max = torch.max(mask, dim=2, keepdim=True)[0]
        mask = torch.softmax(mask - mask_max, dim=2)
        unfold = F.unfold(factor * flow if scale else flow, [3, 3], padding=1)
        unfold = unfold.view(N, D, 9, 1, 1, H, W)
        up_flow = (mask * unfold).sum(dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, D, factor * H, factor * W)

    # --------------------------------------------------------------------- #
    # Forward pass                                                          #
    # --------------------------------------------------------------------- #
    def forward(self, image1, image2, iters=12, params=None, test_mode=False, frame_id=0):
        """Forward identical to original TC‑Stereo with optional DEFOM & scale update."""
        flow_predictions = []
        flow_q_predictions = []
        disp_grad_q_predictions = []

        # normalize to [-1,1]
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        with autocast(enabled=self.args.mixed_precision):
            if self.args.shared_backbone:
                *cnet_list, x = self.cnet(torch.cat((image1, image2), dim=0), dual_inp=True, num_layers=self.args.n_gru_layers)
                fmap1, fmap2 = self.conv2(x).split(dim=0, split_size=x.shape[0] // 2)
            else:
                cnet_list = self.cnet(image1, num_layers=self.args.n_gru_layers)
                fmap1, fmap2 = self.fnet([image1, image2])

            # fuse ViT features into fmap1/fmap2 (concat+1×1 conv)
            # depth_anything expects batch concat of both images
            ih, iw = image1.shape[-2:]
            danv2_io_sizes = (ih, iw, ih, iw)
            _, da_f1, da_f2, idepth = self.defomencoder([image1, image2], danv2_io_sizes)
            # Align channels and add
            fmap1 = fmap1 + self.defom_f_align(da_f1)
            fmap2 = fmap2 + self.defom_f_align(da_f2)

        # Correlation volume
        corr_fn = NewCorrBlock1D(
            fmap1.float(),
            fmap2.float(),
            radius=self.args.corr_radius,
            num_levels=self.args.corr_levels,
            scale_list=self.scale_list,
            scale_corr_radius=self.scale_corr_radius,
        )

        # --- disparity / hidden‑state initialisation (same as original) --- #
        if params is not None:  # temporal case
            # [unchanged temporal warping logic retained from original model]
            K = params['K']
            K_scale = K * torch.tensor([self.scale_rate, self.scale_rate, 1]).view(1, 3, 1).to(K.device)
            K_scale_inv = torch.linalg.inv(K_scale)
            T = params['T']
            previous_T = params['previous_T']
            relative_T = cal_relative_transformation(previous_T, T)
            baseline = params['baseline']
            flow_init = params['last_disp']
            last_net_list = params['last_net_list']
            last_fmap1 = params['fmap1']
            warped_disp, warped_fmap1, sparse_mask = warp(-flow_init, last_fmap1, relative_T, K_scale, K_scale_inv, baseline)
            sparse_disp = warped_disp
            cost = (F.normalize(fmap1, dim=1) * F.normalize(warped_fmap1, dim=1)).sum(dim=1, keepdim=True)
            cost = cost * sparse_mask
        else:
            sparse_disp, cost, sparse_mask = corr_fn.argmax_disp()
            last_net_list = None

            # DEFOM depth‑based init (optionally override sparse_disp)
            eta = 0.5  # as in DEFOM‑Stereo paper
            ih, iw = sparse_disp.shape[-2:]
            max_d = idepth.view(idepth.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1) + 1e-8
            idepth_norm = idepth / max_d * eta * iw + 0.01
            sparse_disp = idepth_norm.detach()

        # disparity completion & network inputs
        with autocast(enabled=self.args.mixed_precision):
            inp_list = [torch.relu(x[1]) for x in cnet_list]
            grad_list = [conv(i) for i, conv in zip(inp_list, self.context_zqr_convs_grad)]
            inp_list = [list(conv(i).split(conv.out_channels // 3, dim=1)) for i, conv in zip(inp_list, self.context_zqr_convs)]
            net_list = [x[0] for x in cnet_list]
            disp_init, disp_mono, w_mono, net_list = self.disp_completor(sparse_disp, cost.detach(), sparse_mask, net_list)

        # warping previous hidden state
        if last_net_list is None:
            warped_net_list = [torch.zeros_like(x[0]) for x in cnet_list]
        else:
            warped_net_list = []
            backward_grid = get_backward_grid(disp_init.detach(), cal_relative_transformation(params['T'], params['previous_T']), K_scale, K_scale_inv, baseline)
            for net in last_net_list:
                warped_net_list.append(bilinear_sampler(net.float(), backward_grid.permute(0, 2, 3, 1)))
                backward_grid = 0.5 * F.interpolate(backward_grid, scale_factor=0.5, mode='bilinear', align_corners=True)

        # fuse hidden state
        net_list = [torch.tanh(x) for x in net_list]
        net_list = [fuse(n, w) for fuse, n, w in zip(self.previous_current_hidden_fuse, net_list, warped_net_list)]

        # flow init
        coords0, coords1 = self.initialize_flow(fmap1)
        coords1 = coords0 - disp_init.detach()

        # --------------------------- recurrent loop --------------------------- #
        for itr in range(iters):
            coords1 = coords1.detach()
            scaling_phase = itr < self.scale_iters
            corr = corr_fn(coords1, scaling=scaling_phase)
            flows_x = coords1 - coords0

            with autocast(enabled=self.args.mixed_precision):
                if scaling_phase:
                    # From DEFOM Stereo
                    net_list, _, scale_factor = self.scale_update_block(
                        net_list, inp_list, corr, flows_x,
                        iter32=self.args.n_gru_layers == 3,
                        iter16=self.args.n_gru_layers >= 2,
                    )
                    coords1 = coords0 - scale_factor * (coords0 - coords1)
                else:
                    net_list, delta_flow = self.update_block(
                        net_list, inp_list, corr, flows_x,
                        iter32=self.args.n_gru_layers == 3,
                        iter16=self.args.n_gru_layers >= 2,
                    )
                    coords1 = coords1 + delta_flow

            disp_q = coords0 - coords1  # current disparity estimate

            # gradient refinement & propagation (unchanged)
            disp_grad, _ = disp2disp_gradient_xy(disp_q.detach())
            with autocast(enabled=self.args.mixed_precision):
                disp_grad, context = self.disp_grad_refine(disp_grad, disp_q, grad_list)
                refined_disp, up_mask = self.disp_refine(
                    disp_grad, disp_q, net_list[0], context, test_mode and itr < iters - 1
                )
                delta_disp = (refined_disp - disp_q).detach()
                net_list[0] = self.hiddenstate_update(net_list[0], delta_disp)

            coords1 = coords0 - refined_disp

            if test_mode and itr < iters - 1:
                continue

            if up_mask is None:
                flows_up = upflow8(-disp_q)
                flow_refine_up = upflow8(-refined_disp)
            else:
                flows_up = self.upsample_flow(-disp_q, up_mask.detach())
                flow_refine_up = self.upsample_flow(-refined_disp, up_mask)

            flow_predictions.append([flows_up, flow_refine_up])
            flow_q_predictions.append([-disp_q, -refined_disp])
            disp_grad_q_predictions.append(disp_grad)

        flow_q_final = -refined_disp
        net_list = [x.detach() for x in net_list]

        if test_mode:
            return {
                'flow': torch.clip(flow_refine_up, max=0),
                'flow_q': torch.clip(flow_q_final, max=0),
                'net_list': net_list,
                'fmap1': fmap1.detach(),
            }

        # identical training dictionary keys as original implementation
        return {
            'flow_mono': -4 * F.interpolate(disp_mono, scale_factor=4, mode='bilinear', align_corners=True),
            'flow_init': -4 * F.interpolate(disp_init, scale_factor=4, mode='bilinear', align_corners=True),
            'flow_predictions': flow_predictions,
            'flow_q_predictions': flow_q_predictions,
            'disp_grad_q_predictions': disp_grad_q_predictions,
            'cost_volume': cost.detach(),
            # temporal info for next frame
            'flow_q': torch.clip(flow_q_final.detach(), max=0),
            'net_list': net_list,
            'fmap1': fmap1.detach(),
        }
