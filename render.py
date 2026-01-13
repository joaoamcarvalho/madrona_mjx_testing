import argparse
import functools
import os

# CUDA kernels cache paths
os.environ["MADRONA_MWGPU_KERNEL_CACHE"] = "madrona_mjx/build/kernel_cache"
os.environ["MADRONA_BVH_KERNEL_CACHE"] = "madrona_mjx/build/bvh_cache"

# Set environment variables for memory management
_GIB = 1 << 30  # 1 GiB
os.environ["MADRONA_MWGPU_DEVICE_HEAP_SIZE"] = str(4 * _GIB)  # 4 GiB for raytracer
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.1'

import time
from typing import List, Optional, Sequence, Tuple, Any, Dict, Callable, Union
import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

from madrona_mjx.renderer import BatchRenderer
from madrona_mjx.wrapper import load_model

# fix seed for reproducibility
seed = 42
np.random.seed(seed)



# --- Configuration and Constants ---
MESH_PATH = "./b88bcf33f25c6cb15b4f129f868dedb.obj"
SCALE = 0.0251334948498337
# MESH_PATH = "./diamond.obj"
# SCALE = 1.0
FOV_RAD = np.pi / 6
FOVY_DEG = float(FOV_RAD * 180 / np.pi)

# --- Domain Randomization Logic ---

def domain_randomize(sys, rng):
    """
    Randomizes the MJX Model along specified axes for Madrona batching.
    
    Madrona requires geom_rgba, geom_matid, and geom_size to be batched.
    - matid -1: Default material.
    - matid -2: Use color override from geom_rgba.
    - matid > 0: Specific pre-generated material.
    """
    @jax.vmap
    def rand(rng):
        rng, color_rng = jax.random.split(rng, 2)

        # Set first geom to use color override, others to default
        geom_matid = sys.geom_matid.at[:].set(-1)
        geom_matid = geom_matid.at[0].set(-2)

        # Randomize color (RGBA) and size
        new_color = jax.random.uniform(color_rng, (1,), minval=0.0, maxval=0.4)
        geom_rgba = sys.geom_rgba.at[0, 0:1].set(new_color)
        new_size = jax.random.uniform(color_rng, (3,), minval=0.8, maxval=1.1)
        geom_size = sys.geom_size.at[0, 0:3].set(new_size)

        # Lighting randomization (Position and Cutoff)
        new_light_pos = jax.random.uniform(
            rng, (3,),
            minval=jp.asarray([-0.5, -0.5, 2]),
            maxval=jp.asarray([0.5, 0.5, 2]),
        )
        light_pos = sys.light_pos.at[:].set(new_light_pos)
        light_dir = sys.light_dir.at[:].set(jp.asarray([0, 0, -1]))
        light_type = sys.light_type.at[:].set(False)
        light_castshadow = sys.light_castshadow.at[:].set(True)
        light_cutoff = sys.light_cutoff.at[:].set(
            jax.random.uniform(rng, (1,), minval=1, maxval=1.5)
        )
        
        return (geom_rgba, geom_matid, geom_size, light_pos, 
                light_dir, light_type, light_castshadow, light_cutoff)

    random_params = rand(rng)
    
    # Define which axes in the system tree are batched (0) vs shared (None)
    in_axes = jax.tree_util.tree_map(lambda x: None, sys)
    in_axes = in_axes.tree_replace({
        'geom_rgba': 0, 'geom_matid': 0, 'geom_size': 0,
        'light_pos': 0, 'light_dir': 0, 'light_type': 0,
        'light_castshadow': 0, 'light_cutoff': 0,
    })

    # Update system with randomized parameters
    sys = sys.tree_replace({
        'geom_rgba': random_params[0],
        'geom_matid': random_params[1],
        'geom_size': random_params[2],
        'light_pos': random_params[3],
        'light_dir': random_params[4],
        'light_type': random_params[5],
        'light_castshadow': random_params[6],
        'light_cutoff': random_params[7],
    })

    return sys, in_axes

# --- Scene Setup ---

# Load mesh with trimesh to calculate centering offset and bounding box
mesh = trimesh.load(MESH_PATH, force="mesh")
mesh.apply_scale(SCALE)

verts = np.asarray(mesh.vertices)
lbs, ubs = verts.min(0), verts.max(0)
object_distance = np.max(ubs - lbs) * 2.0

# Define relative camera-object transform
camera_H_object = np.eye(4)
camera_H_object[:3, :3] = Rotation.random().as_matrix().astype(np.float32)
camera_H_object[2, 3] -= 2.*object_distance

# Center geometry based on its bounding box
bbox_center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
geom_offset_vec = -bbox_center
geom_offset_str = f"{geom_offset_vec[0]} {geom_offset_vec[1]} {geom_offset_vec[2]}"

# MJCF Template
xml_template = """
<mujoco model="cart-pole-minimal">
    <option gravity="0 0 0"/>
    <visual>
        <headlight ambient=".4 .4 .4" diffuse=".8 .8 .8" specular="0.1 0.1 0.1"/>
        <map znear=".01" zfar="100"/>
    </visual>
    <asset>        
        <mesh name="obj" file="{mesh_path}" scale="{scale} {scale} {scale}"/>
        <material name="obj_material" rgba="0.7 0.5 0.3 1" specular="0.5" shininess="0.5" reflectance=".2"/>
    </asset>
    <worldbody>
        <light name="light" directional="true" pos="0 0 10" dir="0 0 -1" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2" castshadow="true"/>
        <camera name="cam" fovy="{fovy_deg}"/>
        <body name="object">
            <freejoint/>
            <geom type="mesh" mesh="obj" pos="{geom_offset}" material="obj_material"/>
        </body>
    </worldbody>
</mujoco>
"""

xml_content = xml_template.format(
    mesh_path=MESH_PATH,
    scale=SCALE,
    fovy_deg=FOVY_DEG,
    geom_offset=geom_offset_str
)

# Convert MuJoCo model to MJX
model = mujoco.MjModel.from_xml_string(xml_content)
mjx_model = mjx.put_model(model)

# Initialize Batch Renderer
num_worlds = 1  # Reduced from 256 to work with raytracer
renderer = BatchRenderer(
    mjx_model,
    gpu_id=0,
    num_worlds=num_worlds,
    batch_render_view_width=400,
    batch_render_view_height=400,
    enabled_geom_groups=np.array([0, 1, 2]),
    enabled_cameras=None,
    add_cam_debug_geo=False,
    use_rasterizer=False,  # Here to switch render mode
)

# --- JAX Initialization and Execution ---

rng = jax.random.PRNGKey(seed=seed)
rng, key = jax.random.split(rng)
randomization_rng = jax.random.split(rng, num_worlds)

# Apply domain randomization
v_mjx_model, v_in_axes = domain_randomize(mjx_model, randomization_rng)

# @jax.jit
def render_envs(rng_keys, sys):
    # @jax.jit
    def init_single_env(rng, s):
        data = mjx.make_data(s)
        
        # Safe assignment based on degrees of freedom (nq)
        if s.nq >= 7:  # Likely a freejoint object (3 pos + 4 quat)
            pos = camera_H_object[:3, 3]
            quat = Rotation.from_matrix(camera_H_object[:3, :3]).as_quat()
            # MuJoCo uses (w, x, y, z) format for quaternions
            quat_wxyz = jp.array([quat[3], quat[0], quat[1], quat[2]])
            data = data.replace(qpos=data.qpos.at[:3].set(pos).at[3:7].set(quat_wxyz))
        else:
            # Fallback for simple joints (e.g., cart-pole)
            data = data.replace(qpos=0.01 * jax.random.uniform(rng, shape=(s.nq,)))

        data = mjx.forward(s, data)
        # Initialize renderer for this instance
        render_token, rgb, depth = renderer.init(data, s)
        return data, render_token, rgb, depth

    # Map the initialization over the batched system
    return jax.vmap(init_single_env, in_axes=[0, v_in_axes])(rng_keys, sys)

# Render
for _ in range(3):
    s = time.time()
    init_keys = jax.random.split(key, num_worlds)
    v_mjx_data, render_token, rgb_batch, depth_batch = render_envs(init_keys, v_mjx_model)
    print("Render time:", time.time() - s)


# --- Visualization ---

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Extract the first image from the batch
color_img = jax.device_get(rgb_batch)[0][0]
depth_img = jax.device_get(depth_batch)[0][0]

# Handle RGBA to RGB conversion if necessary
if color_img.shape[-1] == 4:
    color_img = color_img[..., :3]

axes[0].imshow(color_img)
axes[0].set_title('Rendered Color Image')
axes[0].axis('off')

if depth_img.size > 0 and depth_img.max() > 0:
    
    # filter out invalid depth values and more than 1 meter away
    depth_img = np.where((depth_img > 0) & (depth_img <= 1.0), depth_img, 0)
    # filter out points to be within a percentile range to remove outliers
    z_min, z_max = np.percentile(depth_img[depth_img > 0], [3, 97])
    depth_img = np.where((depth_img >= z_min) & (depth_img <= z_max), depth_img, 0)
    
    
    im = axes[1].imshow(depth_img, cmap='viridis')
    axes[1].set_title('Depth Map')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    
    # --- Point Cloud Visualization ---

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
    
else:
    axes[1].text(0.5, 0.5, 'No valid depth data', 
                ha='center', va='center', transform=axes[1].transAxes)
    axes[1].set_title('Depth Map')
    axes[1].axis('off')

plt.tight_layout()
plt.show()

