"""Debug script for loading the renderer without stepping MJX.

Usage:
    python debug.py --mjcf <path_to_mjcf>
"""

import argparse
import os

import jax
import jax.numpy as jp
from mujoco import mjx
import numpy as np
import trimesh

from madrona_mjx.renderer import BatchRenderer
from madrona_mjx.viz import Visualizer
from madrona_mjx.viz import VisualizerGPUState
from madrona_mjx.wrapper import _identity_randomization_fn
from madrona_mjx.wrapper import load_model

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--mjcf', type=str, required=True)
arg_parser.add_argument('--gpu-id', type=int, default=0)
arg_parser.add_argument('--num-worlds', type=int, default=16)
arg_parser.add_argument('--window-width', type=int, default=1500)
arg_parser.add_argument('--window-height', type=int, default=1000)
arg_parser.add_argument('--batch-render-view-width', type=int, default=640)
arg_parser.add_argument('--batch-render-view-height', type=int, default=640)
arg_parser.add_argument('--add-cam-debug-geo', action='store_true')
arg_parser.add_argument('--use-rasterizer', action='store_true')
args = arg_parser.parse_args()

# CUDA kernels cache paths
os.environ["MADRONA_MWGPU_KERNEL_CACHE"] = "madrona_mjx/build/kernel_cache"
os.environ["MADRONA_BVH_KERNEL_CACHE"] = "madrona_mjx/build/bvh_cache"

def limit_jax_mem(limit):
  os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = f'{limit:.2f}'

limit_jax_mem(0.1)

# Tell XLA to use Triton GEMM
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

if __name__ == '__main__':  
  model = load_model(args.mjcf)
  mjx_model = mjx.put_model(model)
  mjx_data = mjx.make_data(mjx_model)

  # Get FOV from the mujoco model (first camera)
  FOV_RAD = np.radians(model.cam_fovy[0])

  renderer = BatchRenderer(
      mjx_model,
      args.gpu_id,
      args.num_worlds,
      args.batch_render_view_width,
      args.batch_render_view_width,
      np.array([0, 1, 2]),
      None,
      args.add_cam_debug_geo,
      args.use_rasterizer,
      None 
  )
  print("Renderer initialized")

  v_mjx_model, v_in_axes = _identity_randomization_fn(
      mjx_model, args.num_worlds
  )

  @jax.jit
  def init(rng, model):
    def init_(rng, model):
      data = mjx.make_data(model)
      data = mjx.forward(model, data)
      render_token, rgb, depth = renderer.init(data, model)
      return data, render_token, rgb, depth

    return jax.vmap(init_, in_axes=[0, v_in_axes])(rng, model)

  rng = jax.random.PRNGKey(seed=2)
  rng, *key = jax.random.split(rng, args.num_worlds + 1)
  v_mjx_data, render_token, rgb, depth = init(jp.asarray(key), v_mjx_model)
  print("Renderer init called")
  print(f"RGB shape: {rgb.shape}")

  def step(data):
    def step_(data):
      _, rgb, depth = renderer.render(render_token, data)
      return data, rgb, depth

    return jax.vmap(step_)(data)

  step = jax.jit(step).lower(v_mjx_data).compile()
  state = v_mjx_data
  for i in range(5):
    state, rgb, depth = step(state)
    print(f"Step {i}/5 successful")

  del renderer
  print("Finished first test")

  os.environ['MADRONA_DISABLE_CUDA_HEAP_SIZE'] = '1'

  renderer = BatchRenderer(
      mjx_model,
      args.gpu_id,
      args.num_worlds,
      args.batch_render_view_width,
      args.batch_render_view_width,
      np.array([0, 1, 2]),
      None,
      args.add_cam_debug_geo,
      args.use_rasterizer,
      None 
  )
  print("Renderer initialized")

  v_mjx_model, v_in_axes = _identity_randomization_fn(
      mjx_model, args.num_worlds
  )

  @jax.jit
  def init(rng, model):
    def init_(rng, model):
      data = mjx.make_data(model)
      data = mjx.forward(model, data)
      render_token, rgb, depth = renderer.init(data, model)
      return data, render_token, rgb, depth

    return jax.vmap(init_, in_axes=[0, v_in_axes])(rng, model)

  rng = jax.random.PRNGKey(seed=2)
  rng, *key = jax.random.split(rng, args.num_worlds + 1)
  v_mjx_data, render_token, rgb, depth = init(jp.asarray(key), v_mjx_model)
  print("Renderer init called")

  def step(data):
    def step_(data):
      _, rgb, depth = renderer.render(render_token, data)
      return data, rgb, depth

    return jax.vmap(step_)(data)

  step = jax.jit(step).lower(v_mjx_data).compile()
  state = v_mjx_data
  for i in range(5):
    state, rgb, depth = step(state)
    
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Extract the first image from the batch
    color_img = jax.device_get(rgb)[0][0]
    depth_img = jax.device_get(depth)[0][0]
    # Handle RGBA to RGB conversion if necessary
    if color_img.shape[-1] == 4:
        color_img = color_img[..., :3]
    axes[0].imshow(color_img)
    axes[0].set_title('Rendered Color Image')
    axes[0].axis('off')
    
    # filter out invalid depth values and more than 10 meter away
    depth_img = np.where((depth_img > 0) & (depth_img <= 10.0), depth_img, 0)
    # filter out points to be within a percentile range to remove outliers
    # z_min, z_max = np.percentile(depth_img[depth_img > 0], [3, 97])
    # depth_img = np.where((depth_img >= z_min) & (depth_img <= z_max), depth_img, 0)
    
    
    im = axes[1].imshow(depth_img, cmap='viridis')
    axes[1].set_title('Depth Map')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    
    # --- Point Cloud Visualization ---

    # Get FOV from the mujoco model (first camera)
    FOV_RAD = np.radians(model.cam_fovy[0])

    # Camera intrinsics from FOV
    if depth_img.ndim == 3:
        depth_img = depth_img[..., 0]
    height, width = depth_img.shape[-2:]
    focal_length = height / (2 * np.tan(FOV_RAD / 2))
    cx, cy = width / 2, height / 2

    # Create pixel coordinates
    u = np.arange(width)
    v = np.arange(height)
    uu, vv = np.meshgrid(u, v)

    # Unproject depth to 3D points
    z = depth_img    
    x = (uu - cx) * z / focal_length
    y = (vv - cy) * z / focal_length
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    # Filter out invalid/zero depth points
    valid_mask = (z > 0).reshape(-1)
    points = points[valid_mask]

    # Create and visualize point cloud with trimesh
    if points.shape[0] > 0:
        point_cloud = trimesh.PointCloud(vertices=points)
        point_cloud.show()
    else:
        print("No valid points in depth image")
    
    
    
    plt.show()
    
    print(f"Step {i}/5 successful")

  del renderer
  print("Finished second test")
