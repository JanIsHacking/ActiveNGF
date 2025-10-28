# This file is a part of ESLAM.
#
# ESLAM is a NeRF-based SLAM system. It utilizes Neural Radiance Fields (NeRF)
# to perform Simultaneous Localization and Mapping (SLAM) in real-time.
# This software is the implementation of the paper "ESLAM: Efficient Dense SLAM
# System Based on Hybrid Representation of Signed Distance Fields" by
# Mohammad Mahdi Johari, Camilla Carta, and Francois Fleuret.
#
# Copyright 2023 ams-OSRAM AG
#
# Author: Mohammad Mahdi Johari <mohammad.johari@idiap.ch>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file is a modified version of https://github.com/cvg/nice-slam/blob/master/src/utils/Renderer.py
# which is covered by the following copyright and permission notice:
    #
    # Copyright 2022 Zihan Zhu, Songyou Peng, Viktor Larsson, Weiwei Xu, Hujun Bao, Zhaopeng Cui, Martin R. Oswald, Marc Pollefeys
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.

import torch
import torch.nn.functional as F
from src.common import get_rays, sample_pdf, normalize_3d_coordinate, get_rays_downsample

class Renderer(object):
    """
    Renderer class for rendering depth and color.
    Args:
        cfg (dict): configuration.
        eslam (ESLAM): ESLAM object.
        ray_batch_size (int): batch size for sampling rays.
    """
    def __init__(self, cfg, eslam, ray_batch_size=10000):
        self.ray_batch_size = ray_batch_size

        self.perturb = cfg['rendering']['perturb']
        self.n_stratified = cfg['rendering']['n_stratified']
        self.n_importance = cfg['rendering']['n_importance']

        self.scale = cfg['scale']
        self.bound = eslam.bound.to(eslam.device, non_blocking=True)

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = eslam.H, eslam.W, eslam.fx, eslam.fy, eslam.cx, eslam.cy

    def perturbation(self, z_vals):
        """
        Add perturbation to sampled depth values on the rays.
        Args:
            z_vals (tensor): sampled depth values on the rays.
        Returns:
            z_vals (tensor): perturbed depth values on the rays.
        """
        # get intervals between samples
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape, device=z_vals.device)

        return lower + (upper - lower) * t_rand

    def render_batch_ray(self, all_planes, decoders, rays_d, rays_o, device, truncation, gt_depth=None):
        """
        Render depth and color for a batch of rays.
        Args:
            all_planes (Tuple): all feature planes.
            decoders (torch.nn.Module): decoders for TSDF and color.
            rays_d (tensor): ray directions.
            rays_o (tensor): ray origins.
            device (torch.device): device to run on.
            truncation (float): truncation threshold.
            gt_depth (tensor): ground truth depth.
        Returns:
            depth_map (tensor): depth map.
            color_map (tensor): color map.
            volume_densities (tensor): volume densities for sampled points.
            z_vals (tensor): sampled depth values on the rays.

        """
        n_stratified = self.n_stratified
        n_importance = self.n_importance
        n_rays = rays_o.shape[0]

        z_vals = torch.empty([n_rays, n_stratified + n_importance], device=device)
        near = 0.0
        t_vals_uni = torch.linspace(0., 1., steps=n_stratified, device=device)
        t_vals_surface = torch.linspace(0., 1., steps=n_importance, device=device)

        ### pixels with gt depth:
        gt_depth = gt_depth.reshape(-1, 1)
        gt_mask = (gt_depth > 0).squeeze()
        gt_nonezero = gt_depth[gt_mask]

        ## Sampling points around the gt depth (surface)
        gt_depth_surface = gt_nonezero.expand(-1, n_importance)
        z_vals_surface = gt_depth_surface - (1.5 * truncation)  + (3 * truncation * t_vals_surface)

        gt_depth_free = gt_nonezero.expand(-1, n_stratified)
        z_vals_free = near + 1.2 * gt_depth_free * t_vals_uni

        z_vals_nonzero, _ = torch.sort(torch.cat([z_vals_free, z_vals_surface], dim=-1), dim=-1)
        if self.perturb:
            z_vals_nonzero = self.perturbation(z_vals_nonzero)
        z_vals[gt_mask] = z_vals_nonzero

        ### pixels without gt depth (importance sampling):
        if not gt_mask.all():
            with torch.no_grad():
                rays_o_uni = rays_o[~gt_mask].detach()
                rays_d_uni = rays_d[~gt_mask].detach()
                det_rays_o = rays_o_uni.unsqueeze(-1)  # (N, 3, 1)
                det_rays_d = rays_d_uni.unsqueeze(-1)  # (N, 3, 1)
                t = (self.bound.unsqueeze(0) - det_rays_o)/det_rays_d  # (N, 3, 2)
                far_bb, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
                far_bb = far_bb.unsqueeze(-1)
                far_bb += 0.01

                z_vals_uni = near * (1. - t_vals_uni) + far_bb * t_vals_uni
                if self.perturb:
                    z_vals_uni = self.perturbation(z_vals_uni)
                pts_uni = rays_o_uni.unsqueeze(1) + rays_d_uni.unsqueeze(1) * z_vals_uni.unsqueeze(-1)  # [n_rays, n_stratified, 3]

                pts_uni_nor = normalize_3d_coordinate(pts_uni.clone(), self.bound)
                sdf_uni = decoders.get_raw_sdf(pts_uni_nor, all_planes)
                sdf_uni = sdf_uni.reshape(*pts_uni.shape[0:2])
                alpha_uni = self.sdf2alpha(sdf_uni, decoders.beta)
                weights_uni = alpha_uni * torch.cumprod(torch.cat([torch.ones((alpha_uni.shape[0], 1), device=device)
                                                        , (1. - alpha_uni + 1e-10)], -1), -1)[:, :-1]

                z_vals_uni_mid = .5 * (z_vals_uni[..., 1:] + z_vals_uni[..., :-1])
                z_samples_uni = sample_pdf(z_vals_uni_mid, weights_uni[..., 1:-1], n_importance, det=False, device=device)
                z_vals_uni, ind = torch.sort(torch.cat([z_vals_uni, z_samples_uni], -1), -1)
                z_vals[~gt_mask] = z_vals_uni

        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
              z_vals[..., :, None]  # [n_rays, n_stratified+n_importance, 3]

        raw = decoders(pts, all_planes)
        alpha = self.sdf2alpha(raw[..., -1], decoders.beta)
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=device)
                                                , (1. - alpha + 1e-10)], -1), -1)[:, :-1]

        rendered_rgb = torch.sum(weights[..., None] * raw[..., :3], -2)
        rendered_depth = torch.sum(weights * z_vals, -1)

        return rendered_depth, rendered_rgb, raw[..., -1], z_vals

    def sdf2alpha(self, sdf, beta=10):
        """

        """
        return 1. - torch.exp(-beta * torch.sigmoid(-sdf * beta))

    def render_img(self, all_planes, decoders, c2w, truncation, device, gt_depth=None):
        """
        Renders out depth and color images.
        Args:
            all_planes (Tuple): feature planes
            decoders (torch.nn.Module): decoders for TSDF and color.
            c2w (tensor, 4*4): camera pose.
            truncation (float): truncation distance.
            device (torch.device): device to run on.
            gt_depth (tensor, H*W): ground truth depth image.
        Returns:
            rendered_depth (tensor, H*W): rendered depth image.
            rendered_rgb (tensor, H*W*3): rendered color image.

        """
        with torch.no_grad():
            H = self.H
            W = self.W
            rays_o, rays_d = get_rays(H, W, self.fx, self.fy, self.cx, self.cy,  c2w, device)
            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)

            depth_list = []
            color_list = []

            ray_batch_size = self.ray_batch_size
            gt_depth = gt_depth.reshape(-1)

            for i in range(0, rays_d.shape[0], ray_batch_size):
                rays_d_batch = rays_d[i:i+ray_batch_size]
                rays_o_batch = rays_o[i:i+ray_batch_size]
                if gt_depth is None:
                    ret = self.render_batch_ray(all_planes, decoders, rays_d_batch, rays_o_batch,
                                                device, truncation, gt_depth=None)
                else:
                    gt_depth_batch = gt_depth[i:i+ray_batch_size]
                    ret = self.render_batch_ray(all_planes, decoders, rays_d_batch, rays_o_batch,
                                                device, truncation, gt_depth=gt_depth_batch)

                depth, color, _, _ = ret
                depth_list.append(depth.double())
                color_list.append(color)

            depth = torch.cat(depth_list, dim=0)
            color = torch.cat(color_list, dim=0)

            depth = depth.reshape(H, W)
            color = color.reshape(H, W, 3)

            return depth, color

class GraspnessRender(Renderer):
    def render_batch_ray(self, all_planes, decoders, rays_d, rays_o, device, truncation, gt_depth=None):
        """
        Render depth and color for a batch of rays.
        Args:
            all_planes (Tuple): all feature planes.
            decoders (torch.nn.Module): decoders for TSDF and color.
            rays_d (tensor): ray directions.
            rays_o (tensor): ray origins.
            device (torch.device): device to run on.
            truncation (float): truncation threshold.
            gt_depth (tensor): ground truth depth.
        Returns:
            depth_map (tensor): depth map.
            color_map (tensor): color map.
            volume_densities (tensor): volume densities for sampled points.
            z_vals (tensor): sampled depth values on the rays.

        """
        n_stratified = self.n_stratified
        n_importance = self.n_importance
        n_rays = rays_o.shape[0]

        z_vals = torch.empty([n_rays, n_stratified + n_importance], device=device)
        near = 0.0
        t_vals_uni = torch.linspace(0., 1., steps=n_stratified, device=device)
        t_vals_surface = torch.linspace(0., 1., steps=n_importance, device=device)

        ### pixels with gt depth:
        gt_depth = gt_depth.reshape(-1, 1)
        gt_mask = (gt_depth > 0).squeeze()
        gt_nonezero = gt_depth[gt_mask]

        ## Sampling points around the gt depth (surface)
        gt_depth_surface = gt_nonezero.expand(-1, n_importance)
        z_vals_surface = gt_depth_surface - (1.5 * truncation)  + (3 * truncation * t_vals_surface)

        gt_depth_free = gt_nonezero.expand(-1, n_stratified)
        z_vals_free = near + 1.2 * gt_depth_free * t_vals_uni

        z_vals_nonzero, _ = torch.sort(torch.cat([z_vals_free, z_vals_surface], dim=-1), dim=-1)
        if self.perturb:
            z_vals_nonzero = self.perturbation(z_vals_nonzero)
        z_vals[gt_mask] = z_vals_nonzero

        ### pixels without gt depth (importance sampling):
        if not gt_mask.all():
            with torch.no_grad():
                rays_o_uni = rays_o[~gt_mask].detach()
                rays_d_uni = rays_d[~gt_mask].detach()
                det_rays_o = rays_o_uni.unsqueeze(-1)  # (N, 3, 1)
                det_rays_d = rays_d_uni.unsqueeze(-1)  # (N, 3, 1)
                t = (self.bound.unsqueeze(0) - det_rays_o)/det_rays_d  # (N, 3, 2)
                far_bb, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
                far_bb = far_bb.unsqueeze(-1)
                far_bb += 0.01

                z_vals_uni = near * (1. - t_vals_uni) + far_bb * t_vals_uni
                if self.perturb:
                    z_vals_uni = self.perturbation(z_vals_uni)
                pts_uni = rays_o_uni.unsqueeze(1) + rays_d_uni.unsqueeze(1) * z_vals_uni.unsqueeze(-1)  # [n_rays, n_stratified, 3]

                pts_uni_nor = normalize_3d_coordinate(pts_uni.clone(), self.bound)
                sdf_uni = decoders.get_raw_sdf(pts_uni_nor, all_planes[:6])
                sdf_uni = sdf_uni.reshape(*pts_uni.shape[0:2])
                alpha_uni = self.sdf2alpha(sdf_uni, decoders.beta)
                weights_uni = alpha_uni * torch.cumprod(torch.cat([torch.ones((alpha_uni.shape[0], 1), device=device)
                                                        , (1. - alpha_uni + 1e-10)], -1), -1)[:, :-1]

                z_vals_uni_mid = .5 * (z_vals_uni[..., 1:] + z_vals_uni[..., :-1])
                z_samples_uni = sample_pdf(z_vals_uni_mid, weights_uni[..., 1:-1], n_importance, det=False, device=device)
                z_vals_uni, ind = torch.sort(torch.cat([z_vals_uni, z_samples_uni], -1), -1)
                z_vals[~gt_mask] = z_vals_uni

        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
              z_vals[..., :, None]  # [n_rays, n_stratified+n_importance, 3]

        raw = decoders(pts, all_planes)
        alpha = self.sdf2alpha(raw[..., 3], decoders.beta)
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=device)
                                                , (1. - alpha + 1e-10)], -1), -1)[:, :-1]

        rendered_rgb = torch.sum(weights[..., None] * raw[..., :3], -2)
        rendered_depth = torch.sum(weights * z_vals, -1)
        rendered_graspness = torch.sum(weights * raw[..., 4], -1)
        # uncertainty = torch.sum(weights * weights*raw[..., 5],dim=-1)
        uncertainty = torch.mean(raw[..., 5], dim=-1)
        return rendered_depth, rendered_rgb, rendered_graspness, raw[..., 3], z_vals, uncertainty
    
    def render_pointcloud(
        self,
        all_planes,
        decoders,
        points: torch.Tensor,
        device: torch.device,
        truncation: float
    ):
        """
        Render depth, color, and other properties for a point cloud.
        Args:
            all_planes (Tuple): all feature planes.
            decoders (torch.nn.Module): decoders for TSDF and color.
            points (tensor): [N, 3] input point cloud in world coordinates.
            device (torch.device): device to run on.
            truncation (float): truncation threshold (kept for consistency).
        Returns:
            depth_map (tensor): per-point depth relative to origin (L2 norm).
            color_map (tensor): per-point RGB color.
            graspness (tensor): per-point graspness score (from raw[...,4]).
            sdf (tensor): signed distance field values for each point.
            uncertainty (tensor): per-point uncertainty estimate.
        """

        # normalize to [-1, 1] grid coords for decoder
        pts_nor = normalize_3d_coordinate(points.clone(), self.bound)

        # query decoder
        raw = decoders(points.unsqueeze(1), all_planes)  # [N, 1, C]
        raw = raw.squeeze(1)  # [N, C]

        # extract fields
        sdf = raw[..., 3]
        alpha = self.sdf2alpha(sdf, decoders.beta)

        # outputs
        color_map = raw[..., :3]
        depth_map = torch.norm(points, dim=-1)  # distance from origin
        graspness = raw[..., 4]
        uncertainty = raw[..., 5]

        return depth_map, color_map, graspness, sdf, uncertainty
    
    def render_batch_ray_wo_gtdepth(self, all_planes, decoders, rays_d, rays_o, device, truncation):
        """
        Render depth and color for a batch of rays, with detailed debug prints.
        """
        n_stratified = self.n_stratified * 2
        n_importance = self.n_importance
        near = 0.0
        t_vals_uni = torch.linspace(0., 1., steps=n_stratified, device=device)
        t_vals_surface = torch.linspace(0., 1., steps=n_importance, device=device)

        with torch.no_grad():
            rays_o_uni = rays_o.detach()
            rays_d_uni = rays_d.detach()
            det_rays_o = rays_o_uni.unsqueeze(-1)
            det_rays_d = rays_d_uni.unsqueeze(-1)

            # --- Bounding box intersection ---
            t = (self.bound.unsqueeze(0) - det_rays_o) / det_rays_d
            far_bb, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
            far_bb = far_bb.unsqueeze(-1)
            far_bb += 0.01

            print(f"[DEBUG] bound: {self.bound}")
            print(f"[DEBUG] far_bb stats -> min: {far_bb.min().item():.4f}, max: {far_bb.max().item():.4f}, mean: {far_bb.mean().item():.4f}")

            # --- Uniform sampling ---
            z_vals_uni = near * (1. - t_vals_uni) + far_bb * t_vals_uni
            pts_uni = rays_o_uni.unsqueeze(1) + rays_d_uni.unsqueeze(1) * z_vals_uni.unsqueeze(-1)
            print(f"[DEBUG] z_vals_uni range: {z_vals_uni.min().item():.4f} to {z_vals_uni.max().item():.4f}")

            # --- Evaluate SDF field ---
            pts_uni_nor = normalize_3d_coordinate(pts_uni.clone(), self.bound)
            sdf_uni = decoders.get_raw_sdf(pts_uni_nor, all_planes[:6])
            print(f"[DEBUG] sdf_uni stats -> min: {sdf_uni.min().item():.4f}, max: {sdf_uni.max().item():.4f}, mean: {sdf_uni.mean().item():.4f}")

            sdf_uni = sdf_uni.reshape(*pts_uni.shape[0:2])
            min_index = torch.argmin(torch.abs(sdf_uni), dim=-1, keepdim=True)
            pesudo_surface = torch.gather(z_vals_uni, 1, min_index)
            print(f"[DEBUG] pesudo_surface mean: {pesudo_surface.mean().item():.4f}")

            # --- Importance samples around surface ---
            z_vals_surface = pesudo_surface - (1.5 * truncation) + (3 * truncation * t_vals_surface)
            pesudo_free = pesudo_surface.expand(-1, n_stratified)
            z_vals_free = near + 1.2 * pesudo_free * t_vals_uni
            z_vals, _ = torch.sort(torch.cat([z_vals_free, z_vals_surface], dim=-1), dim=-1)
            print(f"[DEBUG] z_vals final range: {z_vals.min().item():.4f} to {z_vals.max().item():.4f}")

        # --- Evaluate field again for rendering ---
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        raw = decoders(pts, all_planes)

        print(f"[DEBUG] raw[...,3] (SDF) range: {raw[...,3].min().item():.4f} to {raw[...,3].max().item():.4f}")

        alpha = self.sdf2alpha(raw[..., 3], decoders.beta)
        print(f"[DEBUG] alpha stats -> min: {alpha.min().item():.4f}, max: {alpha.max().item():.4f}, mean: {alpha.mean().item():.4f}")

        weights = alpha * torch.cumprod(
            torch.cat([torch.ones((alpha.shape[0], 1), device=device),
                    (1. - alpha + 1e-10)], -1), -1)[:, :-1]

        rendered_rgb = torch.sum(weights[..., None] * raw[..., :3], -2)
        rendered_depth = torch.sum(weights * z_vals, -1)
        rendered_graspness = torch.sum(weights * raw[..., 4], -1)
        uncertainty = torch.mean(raw[..., 5], dim=-1)
        depth_uncertainty = torch.sum(-weights * torch.log(weights + 1e-8), dim=-1)

        print(f"[DEBUG] rendered_depth stats -> min: {rendered_depth.min().item():.4f}, max: {rendered_depth.max().item():.4f}, mean: {rendered_depth.mean().item():.4f}")
        print("----------------------------------------------------")

        return rendered_depth, rendered_rgb, rendered_graspness, raw[..., 5], z_vals, uncertainty, depth_uncertainty

    def render_batch_ray_wo_gtdepth2(self, all_planes, decoders, rays_d, rays_o, device, truncation):
        """
        Render depth and color for a batch of rays.
        Args:
            all_planes (Tuple): all feature planes.
            decoders (torch.nn.Module): decoders for TSDF and color.
            rays_d (tensor): ray directions.
            rays_o (tensor): ray origins.
            device (torch.device): device to run on.
            truncation (float): truncation threshold.
            gt_depth (tensor): ground truth depth.
        Returns:
            depth_map (tensor): depth map.
            color_map (tensor): color map.
            volume_densities (tensor): volume densities for sampled points.
            z_vals (tensor): sampled depth values on the rays.

        """
        n_stratified = self.n_stratified*2
        n_importance = self.n_importance
        n_rays = rays_o.shape[0]
        near = 0.0
        t_vals_uni = torch.linspace(0., 1., steps=n_stratified, device=device)

        t_vals_surface = torch.linspace(0., 1., steps=n_importance, device=device)

        ### pixels without gt depth (importance sampling):
        with torch.no_grad():
            rays_o_uni = rays_o.detach()
            rays_d_uni = rays_d.detach()
            det_rays_o = rays_o_uni.unsqueeze(-1)  # (N, 3, 1)
            det_rays_d = rays_d_uni.unsqueeze(-1)  # (N, 3, 1)
            t = (self.bound.unsqueeze(0) - det_rays_o)/det_rays_d  # (N, 3, 2)
            far_bb, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
            far_bb = far_bb.unsqueeze(-1)
            far_bb += 0.01
            z_vals_uni = near * (1. - t_vals_uni) + far_bb * t_vals_uni
            pts_uni = rays_o_uni.unsqueeze(1) + rays_d_uni.unsqueeze(1) * z_vals_uni.unsqueeze(-1)  # [n_rays, n_stratified, 3]

            pts_uni_nor = normalize_3d_coordinate(pts_uni.clone(), self.bound)
            sdf_uni = decoders.get_raw_sdf(pts_uni_nor, all_planes[:6])
            sdf_uni = sdf_uni.reshape(*pts_uni.shape[0:2])
            min_index = torch.argmin(torch.abs(sdf_uni), dim=-1, keepdim=True)
            pesudo_surface = torch.gather(z_vals_uni,1, min_index)
            z_vals_surface = pesudo_surface - (1.5 * truncation) + (3 * truncation * t_vals_surface)
            pesudo_free = pesudo_surface.expand(-1, n_stratified)
            z_vals_free = near + 1.2 * pesudo_free * t_vals_uni
            z_vals, _ = torch.sort(torch.cat([z_vals_free, z_vals_surface], dim=-1), dim=-1)


        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
              z_vals[..., :, None]  # [n_rays, n_stratified+n_importance, 3]

        raw = decoders(pts, all_planes)
        alpha = self.sdf2alpha(raw[..., 3], decoders.beta)
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=device)
                                                , (1. - alpha + 1e-10)], -1), -1)[:, :-1]

        rendered_rgb = torch.sum(weights[..., None] * raw[..., :3], -2)
        rendered_depth = torch.sum(weights * z_vals, -1)
        rendered_graspness = torch.sum(weights * raw[..., 4], -1)
        uncertainty = torch.mean(raw[..., 5], dim=-1)
        depth_uncertainty = torch.sum(-weights*torch.log(weights+1e-8),dim=-1)
        return rendered_depth, rendered_rgb, rendered_graspness, raw[..., 5], z_vals, uncertainty, depth_uncertainty

    def ray_cast(self, all_planes, decoders, c2w, device, foreground = False):
        with torch.no_grad():
            H = self.H
            W = self.W
            rays_o, rays_d = get_rays(H, W, self.fx, self.fy, self.cx, self.cy,  c2w, device)
            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)

            ray_batch_size = self.ray_batch_size
            ig = 0
            for i in range(0, rays_d.shape[0], ray_batch_size):
                rays_d_batch = rays_d[i:i + ray_batch_size]
                rays_o_batch = rays_o[i:i + ray_batch_size]

                n_stratified = self.n_stratified
                n_rays = rays_d_batch.shape[0]
                near = 0.0
                t_vals_uni = torch.linspace(0., 1., steps=n_stratified, device=device)
                rays_o_uni = rays_o_batch.detach()
                rays_d_uni = rays_d_batch.detach()
                det_rays_o = rays_o_uni.unsqueeze(-1)  # (N, 3, 1)
                det_rays_d = rays_d_uni.unsqueeze(-1)  # (N, 3, 1)
                t = (self.bound.unsqueeze(0) - det_rays_o)/det_rays_d  # (N, 3, 2)
                far_bb, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
                far_bb = far_bb.unsqueeze(-1)
                far_bb += 0.01
                z_vals_uni = near * (1. - t_vals_uni) + far_bb * t_vals_uni
                pts_uni = rays_o_uni.unsqueeze(1) + rays_d_uni.unsqueeze(1) * z_vals_uni.unsqueeze(-1)  # [n_rays, n_stratified, 3]

                pts_uni_nor = normalize_3d_coordinate(pts_uni.clone(), self.bound)
                sdf_uni = decoders.get_raw_sdf(pts_uni_nor, all_planes[:6])
                # sdf_uni = sdf_uni.reshape(*pts_uni.shape[0:2])

                mask = (pts_uni_nor>-1) & (pts_uni_nor<1)
                mask = torch.all(mask,dim=-1)

                inbox_sdf = sdf_uni[mask]

                b_ig = ((inbox_sdf>-1) & (inbox_sdf<0)).sum()
                ig += b_ig
        return ig

    def render_img(self, all_planes, decoders, c2w, truncation, device, gt_depth=None, downsample_rate = 1):
        """
        Renders out depth and color images.
        Args:
            all_planes (Tuple): feature planes
            decoders (torch.nn.Module): decoders for TSDF and color.
            c2w (tensor, 4*4): camera pose.
            truncation (float): truncation distance.
            device (torch.device): device to run on.
            gt_depth (tensor, H*W): ground truth depth image.
        Returns:
            rendered_depth (tensor, H*W): rendered depth image.
            rendered_rgb (tensor, H*W*3): rendered color image.

        """
        with torch.no_grad():
            H = self.H
            W = self.W
            rays_o, rays_d = get_rays(H, W, self.fx, self.fy, self.cx, self.cy,  c2w, device)
            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)

            depth_list = []
            color_list = []
            graspness_list = []

            ray_batch_size = self.ray_batch_size
            if not (gt_depth is None):
                gt_depth = gt_depth.reshape(-1)

            for i in range(0, rays_d.shape[0], ray_batch_size):
                rays_d_batch = rays_d[i:i+ray_batch_size]
                rays_o_batch = rays_o[i:i+ray_batch_size]
                if gt_depth is None:
                    depth, color, graspness, _, _,uncertainty,_ = self.render_batch_ray_wo_gtdepth(all_planes, decoders, rays_d_batch, rays_o_batch,
                                                device, truncation)

                else:
                    gt_depth_batch = gt_depth[i:i+ray_batch_size]
                    depth, color, graspness, _, _,uncertainty = self.render_batch_ray(all_planes, decoders, rays_d_batch, rays_o_batch,
                                                device, truncation, gt_depth=gt_depth_batch)

                # depth, color, graspness, _, _,uncertainty,_ = ret
                depth_list.append(depth.double())
                color_list.append(color)
                graspness_list.append(graspness)

            depth = torch.cat(depth_list, dim=0)
            color = torch.cat(color_list, dim=0)
            graspness = torch.cat(graspness_list, dim=0)

            depth = depth.reshape(H, W)
            color = color.reshape(H, W, 3)
            graspness = graspness.reshape(H, W)

            return depth, color, graspness

    def render_img_downsample(self, all_planes, decoders, c2w, truncation, device, gt_depth=None, downsample_rate=1):
        """
        Renders out depth and color images.
        Args:
            all_planes (Tuple): feature planes
            decoders (torch.nn.Module): decoders for TSDF and color.
            c2w (tensor, 4*4): camera pose.
            truncation (float): truncation distance.
            device (torch.device): device to run on.
            gt_depth (tensor, H*W): ground truth depth image.
        Returns:
            rendered_depth (tensor, H*W): rendered depth image.
            rendered_rgb (tensor, H*W*3): rendered color image.

        """
        with torch.no_grad():
            H = self.H
            W = self.W
            rays_o, rays_d = get_rays_downsample(H, W, self.fx, self.fy, self.cx, self.cy, c2w, device,
                                                 sample_rate=downsample_rate)
            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)

            depth_list = []
            color_list = []
            graspness_list = []
            uncertainty_list = []
            depth_uncertainty_list = []

            ray_batch_size = self.ray_batch_size
            if not (gt_depth is None):
                gt_depth = gt_depth.reshape(-1)

            for i in range(0, rays_d.shape[0], ray_batch_size):
                rays_d_batch = rays_d[i:i + ray_batch_size]
                rays_o_batch = rays_o[i:i + ray_batch_size]
                if gt_depth is None:
                    depth, color, graspness, _, _, uncertainty, depth_uncertainty = self.render_batch_ray_wo_gtdepth2(all_planes, decoders, rays_d_batch, rays_o_batch,
                                                           device, truncation)
                    # print(f"[DEBUG] depth shape: {depth.shape}")
                    # print(f"[DEBUG] depth mean: {depth.mean().item():.4f}")
                    # print(f"[DEBUG] depth min: {depth.min().item():.4f}")
                    # print(f"[DEBUG] depth max: {depth.max().item():.4f}")
                else:
                    gt_depth_batch = gt_depth[i:i + ray_batch_size]
                    depth, color, graspness, _, _, uncertainty = self.render_batch_ray(all_planes, decoders, rays_d_batch, rays_o_batch,
                                                device, truncation, gt_depth=gt_depth_batch)

                # depth, color, graspness, _, _, uncertainty, depth_uncertainty = ret
                depth_list.append(depth.double())
                color_list.append(color)
                graspness_list.append(graspness)
                uncertainty_list.append(uncertainty)
                depth_uncertainty_list.append(depth_uncertainty)

            depth_sample = torch.cat(depth_list, dim=0)

            # print(f"[DEBUG] depth_sample shape: {depth_sample.shape}")
            # print(f"[DEBUG] depth_sample mean: {depth_sample.mean().item():.4f}")
            # print(f"[DEBUG] depth_sample min: {depth_sample.min().item():.4f}")
            # print(f"[DEBUG] depth_sample max: {depth_sample.max().item():.4f}")

            color_sample = torch.cat(color_list, dim=0)
            graspness_sample = torch.cat(graspness_list, dim=0)
            uncertainty_sample = torch.cat(uncertainty_list, dim=0)
            depth_uncertainty_sample = torch.cat(depth_uncertainty_list, dim=0)

            Hd = int(H // downsample_rate)
            Wd = int(W // downsample_rate)
            assert Hd * Wd == depth_sample.shape[0], f"Sample count mismatch: {Hd * Wd} != {depth_sample.shape[0]}"

            # i, j = torch.meshgrid(torch.linspace(0, W - 1, int(W / downsample_rate)), torch.linspace(0, H - 1, int(H / downsample_rate)))
            # print(i.shape, j.shape)
            # index = torch.stack([i,j],dim=-1).reshape(-1,2).long()
            # print(index.shape)
            depth = torch.zeros(H, W).cuda()
            graspness = torch.zeros(H, W).cuda()
            uncertainty = torch.zeros(H, W).cuda()
            depth_uncertainty = torch.zeros(H, W).cuda()
            # depth[index] = depth_sample
            # graspness[index] = graspness_sample

            depth_downsampled = depth_sample.reshape(1, 1, Hd, Wd).float().to(device)
            grasp_down = graspness_sample.reshape(1, 1, Hd, Wd).float().to(device)
            uncertainty_down = uncertainty_sample.reshape(1, 1, Hd, Wd).float().to(device)
            depth_uncertainty_down = depth_uncertainty_sample.reshape(1, 1, Hd, Wd).float().to(device)

            valid_mask_down = torch.ones(1, 1, Hd, Wd, dtype=torch.bool, device=device)

            # Upsample depth, graspness, uncertainty and mask
            depth_full = F.interpolate(depth_downsampled, size=(H, W), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
            grasp_full = F.interpolate(grasp_down, size=(H, W), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
            uncertainty_full = F.interpolate(uncertainty_down, size=(H, W), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
            depth_uncertainty_full = F.interpolate(depth_uncertainty_down, size=(H, W), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
            valid_mask_full = F.interpolate(valid_mask_down.float(), size=(H, W), mode='nearest').squeeze(0).squeeze(0).bool()

            # Convert to numpy arrays for visualization
            # depth_np = depth_full.cpu().numpy().squeeze()
            
            # depth[0:H:downsample_rate, 0:W:downsample_rate] = depth_sample.reshape(int(H / downsample_rate), int(W / downsample_rate))
            # graspness[0:H:downsample_rate, 0:W:downsample_rate] = graspness_sample.reshape(int(H / downsample_rate),
            #                                                                        int(W / downsample_rate))
            # uncertainty[0:H:downsample_rate, 0:W:downsample_rate] = uncertainty_sample.reshape(int(H / downsample_rate),
            #                                                                               int(W / downsample_rate))
            # depth_uncertainty[0:H:downsample_rate, 0:W:downsample_rate] = depth_uncertainty_sample.reshape(int(H / downsample_rate),
            #                                                                                    int(W / downsample_rate))
                                                                                               
            # print(f"[DEBUG] depth shape: {depth_full.shape}")
            # print(f"[DEBUG] depth mean: {depth_full.mean().item():.4f}")
            # print(f"[DEBUG] depth min: {depth_full.min().item():.4f}")
            # print(f"[DEBUG] depth max: {depth_full.max().item():.4f}")

            return depth_full, grasp_full, depth_uncertainty_full, valid_mask_full