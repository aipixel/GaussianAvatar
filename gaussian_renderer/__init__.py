## partial code from origin 3D GS source
## https://github.com/graphdeco-inria/gaussian-splatting

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

def render_batch(points, shs, colors_precomp, rotations, scales, opacity, FovX, FovY, height, width, bg_color,
                 world_view_transform, full_proj_transform, active_sh_degree, camera_center):
        
    screenspace_points = torch.zeros_like(points, dtype=points.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(FovX * 0.5)
    tanfovy = math.tan(FovY * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(height),
        image_width=int(width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier= 1.0,
        viewmatrix=world_view_transform,
        projmatrix=full_proj_transform,
        sh_degree=active_sh_degree,
        campos=camera_center,
        prefiltered=False,
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    cov3D_precomp = None

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, _ = rasterizer(
        means3D = points,
        means2D = screenspace_points,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    return rendered_image
