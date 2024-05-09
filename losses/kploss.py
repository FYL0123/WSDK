import torch
import torch.nn as nn
import torch.nn.functional as F 
from .preprocess_utils import *
from torch.distributions import Categorical, Bernoulli
from .soft_detect import SoftDetect
class DiskLoss(nn.Module):
    def __init__(self, configs, device=None):
        super(DiskLoss, self).__init__()
        self.__lossname__ = 'DiskLoss'
        self.config = configs
        self.unfold_size = self.config['grid_size']
        self.t_base = self.config['temperature_base']
        self.t_max = self.config['temperature_max']
        self.reward = getattr(self, self.config['epipolar_reward'])
        self.good_reward = self.config['good_reward']
        self.bad_reward = self.config['bad_reward']
        self.kp_penalty = self.config['kp_penalty']
        self.radius = 2
        self.softdetect = SoftDetect()
        self.PeakyLoss = PeakyLoss()

    def compute_keypoints_distance(self, kpts0, kpts1, p=2):
        """
        Args:
            kpts0: torch.tensor [M,2]
            kpts1: torch.tensor [N,2]
            p: (int, float, inf, -inf, 'fro', 'nuc', optional): the order of norm

        Returns:
            dist, torch.tensor [N,M]
        """
        dist = kpts0[:, None, :] - kpts1[None, :, :]  # [M,N,2]
        dist = torch.norm(dist, p=p, dim=2)  # [M,N]
        return dist

    def compute_correspondence(self, pred0, pred1, inputs, rand=False):
        b, c, h, w = pred0['local_point'].shape
        wh = pred0['local_point'][0].new_tensor([[w - 1, h - 1]])

        pred0_with_rand = pred0
        pred1_with_rand = pred1
        pred0_with_rand['scores'] = []
        pred1_with_rand['scores'] = []
        pred0_with_rand['descriptors'] = []
        pred1_with_rand['descriptors'] = []
        pred0_with_rand['num_det'] = []
        pred1_with_rand['num_det'] = []
        #pred0_with_rand['kpt_prob'] = []
        #pred1_with_rand['kpt_prob'] = []

        kps, score_dispersity, scores, kpt_prob = self.softdetect.detect_keypoints(pred0['local_point'])
        pred0_with_rand['keypoints'] = kps
        pred0_with_rand['score_dispersity'] = score_dispersity
        pred0_with_rand['kpt_prob'] = kpt_prob

        kps, score_dispersity, scores, kpt_prob = self.softdetect.detect_keypoints(pred1['local_point'])
        pred1_with_rand['keypoints'] = kps
        pred1_with_rand['score_dispersity'] = score_dispersity
        pred1_with_rand['kpt_prob'] = kpt_prob

        for idx in range(b):
            # =========================== prepare keypoints
            kpts0, kpts1 = pred0['keypoints'][idx], pred1['keypoints'][idx]  # (x,y), shape: Nx2

            # additional random keypoints
            if rand:
                rand0 = torch.rand(len(kpts0), 2, device=kpts0.device) * 2 - 1  # -1~1
                rand1 = torch.rand(len(kpts1), 2, device=kpts1.device) * 2 - 1  # -1~1
                kpts0 = torch.cat([kpts0, rand0])
                kpts1 = torch.cat([kpts1, rand1])

                pred0_with_rand['keypoints'][idx] = kpts0
                pred1_with_rand['keypoints'][idx] = kpts1

            scores_map0 = pred0['local_point'][idx]
            scores_map1 = pred1['local_point'][idx]
            scores_kpts0 = torch.nn.functional.grid_sample(scores_map0.unsqueeze(0), kpts0.view(1, 1, -1, 2),
                                                           mode='bilinear', align_corners=True).squeeze()
            scores_kpts1 = torch.nn.functional.grid_sample(scores_map1.unsqueeze(0), kpts1.view(1, 1, -1, 2),
                                                           mode='bilinear', align_corners=True).squeeze()

            kpts0_wh_ = (kpts0 / 2 + 0.5) * wh  # N0x2, (w,h)
            kpts1_wh_ = (kpts1 / 2 + 0.5) * wh  # N1x2, (w,h)

            # ========================= nms
            dist = self.compute_keypoints_distance(kpts0_wh_.detach(), kpts0_wh_.detach())
            local_mask = dist < self.radius
            valid_cnt = torch.sum(local_mask, dim=1)
            indices_need_nms = torch.where(valid_cnt > 1)[0]
            for i in indices_need_nms:
                if valid_cnt[i] > 0:
                    kpt_indices = torch.where(local_mask[i])[0]
                    scs_max_idx = scores_kpts0[kpt_indices].argmax()

                    tmp_mask = kpt_indices.new_ones(len(kpt_indices)).bool()
                    tmp_mask[scs_max_idx] = False
                    suppressed_indices = kpt_indices[tmp_mask]

                    valid_cnt[suppressed_indices] = 0
            valid_mask = valid_cnt > 0
            kpts0_wh = kpts0_wh_[valid_mask]
            kpts0 = kpts0[valid_mask]
            scores_kpts0 = scores_kpts0[valid_mask].unsqueeze(0)
            pred0_with_rand['keypoints'][idx] = kpts0.unsqueeze(0)

            valid_mask = valid_mask[:len(pred0_with_rand['score_dispersity'][idx])]
            pred0_with_rand['score_dispersity'][idx] = pred0_with_rand['score_dispersity'][idx][valid_mask].unsqueeze(0)
            pred0_with_rand['kpt_prob'][idx] = pred0_with_rand['kpt_prob'][idx][valid_mask].unsqueeze(0)
            pred0_with_rand['num_det'].append(valid_mask.sum())

            dist = self.compute_keypoints_distance(kpts1_wh_.detach(), kpts1_wh_.detach())
            local_mask = dist < self.radius
            valid_cnt = torch.sum(local_mask, dim=1)
            indices_need_nms = torch.where(valid_cnt > 1)[0]
            for i in indices_need_nms:
                if valid_cnt[i] > 0:
                    kpt_indices = torch.where(local_mask[i])[0]
                    scs_max_idx = scores_kpts1[kpt_indices].argmax()

                    tmp_mask = kpt_indices.new_ones(len(kpt_indices)).bool()
                    tmp_mask[scs_max_idx] = False
                    suppressed_indices = kpt_indices[tmp_mask]

                    valid_cnt[suppressed_indices] = 0
            valid_mask = valid_cnt > 0
            kpts1_wh = kpts1_wh_[valid_mask]
            kpts1 = kpts1[valid_mask]
            scores_kpts1 = scores_kpts1[valid_mask].unsqueeze(0)
            pred1_with_rand['keypoints'][idx] = kpts1.unsqueeze(0)

            valid_mask = valid_mask[:len(pred1_with_rand['score_dispersity'][idx])]
            pred1_with_rand['score_dispersity'][idx] = pred1_with_rand['score_dispersity'][idx][valid_mask].unsqueeze(0)
            pred1_with_rand['kpt_prob'][idx] = pred1_with_rand['kpt_prob'][idx][valid_mask].unsqueeze(0)
            pred1_with_rand['num_det'].append(valid_mask.sum())

            # del dist, local_mask, valid_cnt, indices_need_nms, scs_max_idx, tmp_mask, suppressed_indices, valid_mask
            # torch.cuda.empty_cache()
            # ========================= nms

            pred0_with_rand['scores'].append(scores_kpts0)
            pred1_with_rand['scores'].append(scores_kpts1)
            descriptor_map0, descriptor_map1 = pred0['local_map'][idx], pred1['local_map'][idx]
            #desc0 = torch.nn.functional.grid_sample(descriptor_map0.unsqueeze(0), kpts0.view(1, 1, -1, 2),
            #                                        mode='bilinear', align_corners=True)[0, :, 0, :].t().unsqueeze(0)
            #desc1 = torch.nn.functional.grid_sample(descriptor_map1.unsqueeze(0), kpts1.view(1, 1, -1, 2),
            #                                        mode='bilinear', align_corners=True)[0, :, 0, :].t().unsqueeze(0)
            #desc0 = torch.nn.functional.normalize(desc0, p=2, dim=1)
            #desc1 = torch.nn.functional.normalize(desc1, p=2, dim=1)

            #pred0_with_rand['descriptors'].append(desc0)
            #pred1_with_rand['descriptors'].append(desc1)

        return pred0_with_rand, pred1_with_rand
    def mutual_argmax(self, value, mask=None, as_tuple=True):
        """
        Args:
            value: MxN
            mask:  MxN

        Returns:

        """
        value = value - value.min()  # convert to non-negative tensor
        if mask is not None:
            value = value * mask

        max0 = value.max(dim=1, keepdim=True)  # the col index the max value in each row
        max1 = value.max(dim=0, keepdim=True)

        valid_max0 = value == max0[0]
        valid_max1 = value == max1[0]

        mutual = valid_max0 * valid_max1
        if mask is not None:
            mutual = mutual * mask

        return mutual.nonzero(as_tuple=as_tuple)

    def mutual_argmin(self, value, mask=None):
        return self.mutual_argmax(-value, mask)

    def point_distribution(self, logits):
        proposal_dist = Categorical(logits=logits) # bx1x(h//g)x(w//g)x(g*g)
        proposals     = proposal_dist.sample() # bx1x(h//g)x(w//g)
        proposal_logp = proposal_dist.log_prob(proposals) # bx1x(h//g)x(w//g)

        # accept_logits = select_on_last(logits, proposals).squeeze(-1)
        accept_logits = torch.gather(logits, dim=-1, index=proposals[..., None]).squeeze(-1) # bx1x(h//g)x(w//g)

        accept_dist    = Bernoulli(logits=accept_logits)
        accept_samples = accept_dist.sample() # bx1x(h//g)x(w//g)
        accept_logp    = accept_dist.log_prob(accept_samples) # for accepted points, equals to sigmoid() then log(); for denied, (1-sigmoid).log
        accept_mask    = accept_samples == 1.

        logp = proposal_logp + accept_logp

        return proposals, accept_mask, logp

    def point_sample(self, kp_map):
        kpmap_unfold = unfold(kp_map, self.unfold_size)
        proposals, accept_mask, logp = self.point_distribution(kpmap_unfold)

        b, _, h, w = kp_map.shape
        grids_org = gen_grid(h_min=0, h_max=h-1, w_min=0, w_max=w-1, len_h=h, len_w=w)
        grids_org = grids_org.reshape(h, w, 2)[None, :, :, :].repeat(b, 1, 1, 1).to(kp_map)
        grids_org = grids_org.permute(0,3,1,2) # bx2xhxw
        grids_unfold = unfold(grids_org, self.unfold_size) # bx2x(h//g)x(w//g)x(g*g)

        kps = grids_unfold.gather(dim=4, index=proposals.unsqueeze(-1).repeat(1,2,1,1,1))
        return kps.squeeze(4).permute(0,2,3,1), logp, accept_mask

    @ torch.no_grad()
    def constant_reward(self, inputs, outputs, coord1, coord2, idx, reward_thr, rescale_thr):
        coord1_h = homogenize(coord1).transpose(1, 2)  # bx3xm
        coord2_h = homogenize(coord2).transpose(1, 2)  # bx3xn
        fmatrix = inputs['F1'][idx].unsqueeze(0)
        fmatrix2 = inputs['F2'][idx].unsqueeze(0)

        # compute the distance of the points in the second image
        epipolar_line = fmatrix.bmm(coord1_h)
        epipolar_line_ = epipolar_line / torch.clamp(
            torch.norm(epipolar_line[:, :2, :], p=2, dim=1, keepdim=True), min=1e-8)
        epipolar_dist = torch.abs(epipolar_line_.transpose(1, 2) @ coord2_h)  # bxmxn

        # compute the distance of the points in the first image
        epipolar_line2 = fmatrix2.bmm(coord2_h)
        epipolar_line2_ = epipolar_line2 / torch.clamp(
            torch.norm(epipolar_line2[:, :2, :], p=2, dim=1, keepdim=True), min=1e-8)
        epipolar_dist2 = torch.abs(epipolar_line2_.transpose(1, 2) @ coord1_h)  # bxnxm
        epipolar_dist2 = epipolar_dist2.transpose(1, 2)  # bxmxn

        if rescale_thr:
            b, _, _ = epipolar_dist.shape
            dist1 = epipolar_dist.detach().reshape(b, -1).mean(1, True)
            dist2 = epipolar_dist2.detach().reshape(b, -1).mean(1, True)
            dist_ = torch.cat([dist1, dist2], dim=1)
            scale1 = dist1 / dist_.min(1, True)[0].clamp(1e-6)
            scale2 = dist2 / dist_.min(1, True)[0].clamp(1e-6)
            thr1 = reward_thr * scale1
            thr2 = reward_thr * scale2
            thr1 = thr1.reshape(b, 1, 1)
            thr2 = thr2.reshape(b, 1, 1)
        else:
            thr1 = reward_thr
            thr2 = reward_thr
            scale1 = epipolar_dist2.new_tensor(1.)
            scale2 = epipolar_dist2.new_tensor(1.)

        #good = (epipolar_dist < thr1) & (epipolar_dist2 < thr2)
        good = torch.zeros_like(epipolar_dist).to('cuda')
        dist_l2 = ((epipolar_dist + epipolar_dist2) / 2.).squeeze(0)
        # find mutual correspondences by calculating the distance
        # between keypoints (I1) and warpped keypoints (I2->I1)
        mutual_min_indices = self.mutual_argmin(dist_l2)

        dist_mutual_min = dist_l2[mutual_min_indices]
        valid_dist_mutual_min = dist_mutual_min.detach() < thr1 + thr2

        ids0_d = mutual_min_indices[0][valid_dist_mutual_min]
        ids1_d = mutual_min_indices[1][valid_dist_mutual_min]

        good[:, ids0_d, ids1_d] = 1
        good = good.bool()
        reward = self.good_reward * good + self.bad_reward * (~good)
        return reward, scale1, scale2

    @ torch.no_grad()
    def dynamic_reward(self, inputs, outputs, coord1, coord2, reward_thr, rescale_thr):
        coord1_h = homogenize(coord1).transpose(1, 2) #bx3xm
        coord2_h = homogenize(coord2).transpose(1, 2) #bx3xn
        fmatrix = inputs['F1']
        fmatrix2 = inputs['F2']

        # compute the distance of the points in the second image
        epipolar_line = fmatrix.bmm(coord1_h)
        epipolar_line_ = epipolar_line / torch.clamp(
            torch.norm(epipolar_line[:, :2, :], p=2, dim=1, keepdim=True), min=1e-8)
        epipolar_dist = torch.abs(epipolar_line_.transpose(1, 2)@coord2_h) #bxmxn

        # compute the distance of the points in the first image
        epipolar_line2 = fmatrix2.bmm(coord2_h)
        epipolar_line2_ = epipolar_line2 / torch.clamp(
            torch.norm(epipolar_line2[:, :2, :], p=2, dim=1, keepdim=True), min=1e-8)
        epipolar_dist2 = torch.abs(epipolar_line2_.transpose(1, 2)@coord1_h) #bxnxm
        epipolar_dist2 = epipolar_dist2.transpose(1,2) #bxmxn

        if rescale_thr:
            b, _, _ = epipolar_dist.shape
            dist1 = epipolar_dist.detach().reshape(b, -1).mean(1,True)
            dist2 = epipolar_dist2.detach().reshape(b,-1).mean(1,True)
            dist_ = torch.cat([dist1, dist2], dim=1)
            scale1 = dist1/dist_.min(1,True)[0].clamp(1e-6)
            scale2 = dist2/dist_.min(1,True)[0].clamp(1e-6)
            thr1 = reward_thr*scale1
            thr2 = reward_thr*scale2
            thr1 = thr1.reshape(b,1,1)
            thr2 = thr2.reshape(b,1,1)
        else:
            thr1 = reward_thr
            thr2 = reward_thr
            scale1 = epipolar_dist2.new_tensor(1.) 
            scale2 = epipolar_dist2.new_tensor(1.) 

        reward = torch.exp(-epipolar_dist/thr1) + torch.exp(-epipolar_dist2/thr2) - 2/torch.exp(torch.ones_like(epipolar_dist)).to(epipolar_dist)
        reward = reward.clamp(min=self.bad_reward)
        return reward, scale1, scale2

    def forward(self, inputs, outputs, processed):
        preds1 = outputs['preds1']
        preds2 = outputs['preds2']
        kp_map1, kp_map2 = preds1['local_point'], preds2['local_point']
        xf1, xf2 = preds1['local_map'], preds2['local_map']
        b,c,h4,w4 = xf1.shape
        _, _, h, w = kp_map1.shape
        temperature = min(self.t_base + outputs['epoch'], self.t_max)
        pred0_with_rand, pred1_with_rand = self.compute_correspondence(preds1, preds2, inputs)
        loss = 0.
        reinforce1 = 0.
        kp_penalty1 = 0.
        loss_peaky1 = 0.
        for i in range(b):
            coord1_n = pred0_with_rand['keypoints'][i]
            coord2_n = pred1_with_rand['keypoints'][i]
            coord1 = denormalize_coords(coord1_n, h, w)
            coord2 = denormalize_coords(coord2_n, h, w)
            # feat1 = pred0_with_rand['descriptors'][i]
            # feat2 = pred1_with_rand['descriptors'][i]
            logp1 = pred0_with_rand['kpt_prob'][i].log()
            logp2 = pred1_with_rand['kpt_prob'][i].log()
            # coord1 = coord1.reshape(b,-1,2)
            # coord2 = coord2.reshape(b,-1,2)

            # coord1_n = normalize_coords(coord1, h, w) # bx((h//g)*(w//g))x2
            # coord2_n = normalize_coords(coord2, h, w)

            # feat1 = F.grid_sample(xf1, coord1_n, align_corners=False).reshape(b,c,-1) # bxcx((h//g)*(w//g))
            # feat2 = F.grid_sample(xf2, coord2_n, align_corners=False).reshape(b,c,-1)
            feat1 = sample_feat_by_coord(xf1[i].unsqueeze(0), coord1_n, self.config['loss_distance'] == 'cos')  # bxmxc
            feat2 = sample_feat_by_coord(xf2[i].unsqueeze(0), coord2_n, self.config['loss_distance'] == 'cos')  # bxnxc

            # matching
            if self.config['match_grad']:
                costs = 1 - feat1 @ feat2.transpose(1, 2)  # bxmxn 0-2
            else:
                with torch.no_grad():
                    costs = 1 - feat1 @ feat2.transpose(1, 2)  # bxmxn 0-2
            affinity = -temperature * costs

            cat_I = Categorical(logits=affinity)
            cat_T = Categorical(logits=affinity.transpose(1, 2))

            dense_p = cat_I.probs * cat_T.probs.transpose(1, 2)
            dense_logp = cat_I.logits + cat_T.logits.transpose(1, 2)

            if self.config['cor_detach']:
                sample_p = dense_p.detach()
            else:
                sample_p = dense_p

            reward, scale1, scale2 = self.reward(inputs, outputs, coord1, coord2, i, **self.config['reward_config'])

            kps_logp = logp1.reshape(1, 1, -1).transpose(1,2) + logp2.reshape(1,1,-1) # bxmxn
            sample_plogp = sample_p * (dense_logp + kps_logp)
            #accept_mask = accept_mask1.reshape(1,1,-1).transpose(1,2) * accept_mask2.reshape(1,1,-1) # bxmxn

            reinforce = (reward * sample_plogp).sum()
            kp_penalty = self.kp_penalty * (logp1.sum() + logp2.sum())

            loss_peaky0 = self.PeakyLoss(pred0_with_rand)
            loss_peaky1 = self.PeakyLoss(pred1_with_rand)
            loss_peaky = (loss_peaky0 + loss_peaky1) / 2.

            loss += (-reinforce - kp_penalty) + loss_peaky
            #loss += loss_peaky
            reinforce1 += reinforce.detach()
            kp_penalty1 += kp_penalty.detach()
            loss_peaky1 += loss_peaky.detach()
            #sample_p_detach = sample_p.detach()
        components = {'reinforce':reinforce1, 'kp_penalty': kp_penalty1, 'loss_peaky':loss_peaky1
            }
        return loss, components
class PeakyLoss(object):
    """ PeakyLoss to avoid an uniform score map """

    def __init__(self, scores_th: float = 0.1):
        super().__init__()
        self.scores_th = scores_th

    def __call__(self, pred):
        b, c, h, w = pred['local_point'].shape
        loss_mean = 0
        CNT = 0

        for idx in range(b):
            n_original = len(pred['score_dispersity'][idx])
            scores_kpts = pred['scores'][idx][:n_original]
            valid = scores_kpts > self.scores_th
            loss_peaky = pred['score_dispersity'][idx][valid]

            loss_mean = loss_mean + loss_peaky.sum()
            CNT = CNT + len(loss_peaky)

        loss_mean = loss_mean / CNT if CNT != 0 else pred['local_point'].new_tensor(0)
        assert not torch.isnan(loss_mean)
        return loss_mean