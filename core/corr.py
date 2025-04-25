import torch
import torch.nn.functional as F
from core.utils.utils import bilinear_sampler

### From DEFOM Stereo

class NewCorrBlock1D:
    def __init__(self, fmap1, fmap2, coords, num_levels=4, radius=4,
                 scale_list=[0.25, 0.5, 2.0, 4.0], scale_corr_radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.scale_list = scale_list
        self.scale_corr_radius = scale_corr_radius
        self.corr_pyramid = []
        self.coords_pyramid = []
        dx = torch.linspace(-radius, radius, 2*radius+1)
        self.dx = dx[:, None].to(coords.device)

        sdx = torch.linspace(-scale_corr_radius, scale_corr_radius, 2*scale_corr_radius+1)
        self.sdx = sdx[:, None].to(coords.device)

        # all pairs correlation
        corr = CorrBlock1D.corr(fmap1, fmap2)

        batch, h1, w1, _, w2 = corr.shape
        self.batch = batch
        self.h1 = h1
        self.w1 = w1
        self.w2 = w2
        corr = corr.reshape(batch*h1*w1, 1, 1, w2)
        self.coords = coords.reshape(batch*h1*w1, 1, 1, 1)

        self.corr_pyramid.append(corr)
        for i in range(1, self.num_levels):
            corr = F.avg_pool2d(corr, [1, 2], stride=[1, 2])
            self.corr_pyramid.append(corr)

    def __call__(self, disp, scaling=False):
        batch, _, h1, w1 = disp.shape

        disp = disp.reshape(self.batch*self.h1*self.w1, 1, 1, 1)
        out_pyramid = []

        if scaling:
            corr = self.corr_pyramid[0]
            for scale in self.scale_list:
                x0 = self.sdx + self.coords - scale * disp
                y0 = torch.zeros_like(x0)
                coords_lvl = torch.cat([x0, y0], dim=-1)
                corr_s = bilinear_sampler(corr, coords_lvl)
                corr_s = corr_s.view(self.batch, self.h1, self.w1, -1)
                out_pyramid.append(corr_s)
        else:
            coords = self.coords - disp
            for i in range(self.num_levels):
                corr = self.corr_pyramid[i]
                x0 = self.dx + coords / 2**i
                y0 = torch.zeros_like(x0)
                coords_lvl = torch.cat([x0, y0], dim=-1)
                corr_s = bilinear_sampler(corr, coords_lvl)
                corr_s = corr_s.view(self.batch, self.h1, self.w1, -1)
                out_pyramid.append(corr_s)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        B, D, H, W1 = fmap1.shape
        _, _, _, W2 = fmap2.shape
        fmap1 = fmap1.view(B, D, H, W1)
        fmap2 = fmap2.view(B, D, H, W2)
        corr = torch.einsum('aijk,aijh->ajkh', fmap1, fmap2)
        corr = corr.reshape(B, H, W1, 1, W2).contiguous()
        return corr / torch.sqrt(torch.tensor(D).float())

    def get_cost_volume(self):
        return self.corr_pyramid[-1]

    def argmax_disp(self):
        B, W2, H, W1 = self.corr_pyramid[-1].shape
        main_cost, index = self.corr_pyramid[-1].max(dim=1, keepdim=True)
        cost_index_volume = torch.arange(W2).to(main_cost.device).reshape(1, W2, 1, 1).expand(B, W2, H, W1)
        masked_cost_volume = torch.where((cost_index_volume >= index-1.5) & (cost_index_volume < index+1.5), torch.zeros_like(self.corr_pyramid[-1]), self.corr_pyramid[-1])
        sub_cost, _ = masked_cost_volume.max(dim=1, keepdim=True)
        mask = (main_cost - sub_cost > 0.3).float()  # 0.3 is the threshold when inference, 0.5 is the threshold when training
        disp = torch.arange(W1).reshape(1, 1, 1, W1).to(index.device).expand(B, 1, H, W1) - index
        sparse_disp = disp * mask
        main_cost = main_cost * mask
        assert not torch.isnan(sparse_disp).any() and not torch.isinf(sparse_disp).any(), [sparse_disp,torch.max(sparse_disp)]
        assert not torch.isnan(mask).any() and not torch.isinf(mask).any(), [mask,torch.max(mask)]
        return sparse_disp, main_cost, mask

### TC Stereo

class CorrBlock1D:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4, thres=0.2):
        self.num_levels = num_levels
        self.thres = thres
        self.radius = radius
        self.corr_pyramid = []
        self.cost_volume = CorrBlock1D.corr(fmap1, fmap2)
        # all pairs correlation
        corr = self.cost_volume.clone()

        batch, h1, w1, _, w2 = corr.shape
        corr = corr.reshape(batch*h1*w1, 1, 1, w2)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels):
            corr = F.avg_pool2d(corr, [1,2], stride=[1,2])
            self.corr_pyramid.append(corr)
        # for cost volume reshape to [b,w2,h1,w1]
        self.cost_volume = self.cost_volume.squeeze(3).permute(0, 3, 1, 2).contiguous()
        # mask corr to zero when w2 > w1 to avoid negative disparity
        mask = torch.zeros_like(self.cost_volume)
        w1_index = torch.arange(w1).to(corr.device).view(1, 1, 1, w1).expand(batch, w2, h1, w1)
        w2_index = torch.arange(w2).to(corr.device).view(1, w2, 1, 1).expand(batch, w2, h1, w1)
        mask = torch.where((w1_index < w2_index), mask, torch.ones_like(mask))
        self.cost_volume = self.cost_volume * mask

    def __call__(self, coords):
        r = self.radius
        coords = coords[:, :1].permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1)
            dx = dx.view(2*r+1, 1).to(coords.device)
            x0 = dx + coords.reshape(batch*h1*w1, 1, 1, 1) / 2**i
            y0 = torch.zeros_like(x0)

            coords_lvl = torch.cat([x0,y0], dim=-1)
            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        B, D, H, W1 = fmap1.shape
        _, _, _, W2 = fmap2.shape
        fmap1 = F.normalize(fmap1.view(B, D, H, W1),dim=1)
        fmap2 = F.normalize(fmap2.view(B, D, H, W2),dim=1)
        corr = torch.einsum('aijk,aijh->ajkh', fmap1, fmap2)
        corr = corr.reshape(B, H, W1, 1, W2).contiguous()
        return corr

    def get_cost_volume(self):
        return self.cost_volume

    def argmax_disp(self):
        B, W2, H, W1 = self.cost_volume.shape
        main_cost, index = self.cost_volume.max(dim=1, keepdim=True)
        cost_index_volume = torch.arange(W2).to(main_cost.device).reshape(1, W2, 1, 1).expand(B, W2, H, W1)
        masked_cost_volume = torch.where((cost_index_volume >= index-1.5) & (cost_index_volume < index+1.5), torch.zeros_like(self.cost_volume), self.cost_volume)
        sub_cost, _ = masked_cost_volume.max(dim=1, keepdim=True)
        mask = (main_cost - sub_cost > 0.3).float()  # 0.3 is the threshold when inference, 0.5 is the threshold when training
        disp = torch.arange(W1).reshape(1, 1, 1, W1).to(index.device).expand(B, 1, H, W1) - index
        sparse_disp = disp * mask
        main_cost = main_cost * mask
        assert not torch.isnan(sparse_disp).any() and not torch.isinf(sparse_disp).any(), [sparse_disp,torch.max(sparse_disp)]
        assert not torch.isnan(mask).any() and not torch.isinf(mask).any(), [mask,torch.max(mask)]
        return sparse_disp, main_cost, mask