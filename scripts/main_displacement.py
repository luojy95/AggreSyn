import sys
sys.path.append(".")

import datetime
import argparse

import os
import glob
import math
import json
import tqdm
import numpy as np
import bpy

from scripts import common as C
from scripts import math_utils as MU
from scripts.scene import Scene
from scripts.gradation_helper import MIN_DIAMETER, MAX_FINE_DIAMETER, example_gradations
from scripts.addon_helper import AddonHelper as A
from scripts.gradation_helper import GradationHelper as G
from scripts.material_helper import MaterialHelper as M
from scripts.render_helper import RenderHelper as R
from scripts.blender_utils import BlenderUtils as U

argv = sys.argv
user_args = argv[argv.index("--") + 1:]  # get all args after "--"
print("user input arguments:", user_args)

parser = argparse.ArgumentParser()

parser.add_argument("--id", "-i", type=str, default=f'run_{datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")}')
parser.add_argument("--random_seed", "-rs", type=int, default=None)

parser.add_argument("--scene_scale", "-s", type=float, default=0.5)
parser.add_argument("--global_scale", "-gs", type=float, default=2)

parser.add_argument("--gradation", "-g", type=str, default="FI23", choices=list(example_gradations.keys()))

parser.add_argument("--num_ballasts", "-n", type=int, default=50)
parser.add_argument("--max_num_meshes", "-nmesh", type=int, default=10)
parser.add_argument("--min_num_ballast_mats", "-bmin", type=int, default=2)
parser.add_argument("--max_num_ballast_mats", "-bmax", type=int, default=4)
parser.add_argument("--use_mixed_mats", "-mix", action="store_true", default=False)
parser.add_argument("--mixed_mats_z_thresh_width", "-zwidth", type=float, default=0.4)

parser.add_argument("--use_adaptive_disp", action="store_true", default=False)
parser.add_argument("--fine_density", "-d", type=float, default=0.74)

parser.add_argument("--top_fine_min_d", "-tdmin", type=float, default=0.1)
parser.add_argument("--top_fine_max_d", "-tdmax", type=float, default=MAX_FINE_DIAMETER)
parser.add_argument("--top_fine_num", "-tnum", type=int, default=400)
parser.add_argument("--top_fine_disp_shift", "-tshift", type=float, default=0.2)

parser.add_argument("--use_hair_sys", action="store_true", default=False)
parser.add_argument("--hair_sys_density", "-hd", type=int, default=10000)
parser.add_argument("--hair_sys_variaty", "-hv", type=int, default=10)

parser.add_argument("--image_width", "-iw", type=int, default=1920)
parser.add_argument("--image_height", "-ih", type=int, default=1920)

parser.add_argument("--disp_scale", "-ds", type=float, default=0.1)
parser.add_argument("--disp_uv_scale", "-uv", type=float, default=4/3)

parser.add_argument("--render_cam_traj", action="store_true", default=False)
parser.add_argument("--num_cam_images", "-in", type=int, default=1)
parser.add_argument("--camera_height", "-ch", type=float, default=0.5)
parser.add_argument("--line_scan_mode", "-line", action="store_true", default=False)

parser.add_argument("--render_result", action="store_true", default=False)
parser.add_argument("--render_labels", action="store_true", default=False)
parser.add_argument("--render_one_env_only", "-one", action="store_true", default=False)
parser.add_argument("--keep_largest_label", action="store_true", default=False)
parser.add_argument("--output_label_details", action="store_true", default=False)
parser.add_argument("--label_thresh", "-t", type=float, default=0.2)
parser.add_argument("--min_num_pixel_per_label", type=float, default=100)

parser.add_argument("--save_scene", action="store_true", default=False)
parser.add_argument("--save_gradation", action="store_true", default=True)

args = parser.parse_args(user_args)

if args.random_seed is not None:
    np.random.seed(args.random_seed)

TAG = args.id + '_' + args.gradation
input_gradation = example_gradations[args.gradation]
CAMERA_USE_TRAJECTORY = args.render_cam_traj
DISPLACEMENT_HAIR_SYSTEMS = args.use_hair_sys
RENDER_LABELS = args.render_labels
RENDER_RESULT = args.render_result

DISPLACEMENT_ON_LANDSCAPE = False

# to avoid too small particles so the simulation can be accurate
GLOBAL_SCALE = args.global_scale
SCENE_SCALE = args.scene_scale # 0.5 m x 0.5 m

SCENE_SIZE = SCENE_SCALE * GLOBAL_SCALE
BALLAST_ZONE_SIZE = SCENE_SIZE / 3
BALLAST_ZONE_HEIGHT = SCENE_SIZE * 1.5
BALLAST_ZONE_HEIGHT_START =  0.5 * SCENE_SIZE / 5

NUM_BALLAST = args.num_ballasts

MAX_FRAME_NUM = 250
BALLAST_FALLING_FRAME_PHASE_I = 250
BALLAST_FALLING_FRAME_PHASE_II = 200

IMAGE_WIDTH = args.image_width
IMAGE_HEIGHT = args.image_height

CYCLES_DEVICE = 'GPU'
CYCLES_SAMPLES_LABEL = 64
CYCLES_SAMPLES_IMAGE = 64 # 4096

LOADING_MESH_DEFAULT_SCALE = (.001, .001, .001)
LOADING_MESH_START_X = 2 * SCENE_SIZE 
HIGH_POLY_MESH_KEEP_FACE_RATIO = 0.2 # The mesh complexity
LOW_POLY_MESH_KEEP_FACES = 50

# candidates: rocks_noise, ant_turbulence
BASE_LANDSCAPE_TYPE = "rocks_noise" 
BASE_LANDSCAPE_HEIGHT =  0 
if args.gradation in ["FI30", "FI39"]:
    BASE_LANDSCAPE_HEIGHT = 0.05 * SCENE_SIZE * (input_gradation.cdf[0.375] / 0.125) # 0.05 - 7/14, 0.1 - 23, 0.15 - 30, 0.2 - 39
BASE_LANDSCAPE_Z = 0.001 * SCENE_SIZE
BASE_LANDSCAPE_NOISE_SIZE = 1

CAMERA_LOCATION = (0, 0, args.camera_height * GLOBAL_SCALE) # single result camera height
CAMERA_LOOK_AT = (0, 0, 0)
CAMERA_TRAJECTORY_NUM_IMAGES = args.num_cam_images
# camera trajectory cam height
CAMERA_TRAJECTORY_START = np.array([0, -SCENE_SIZE/16, args.camera_height * GLOBAL_SCALE])
CAMERA_TRAJECTORY_END = np.array([0, SCENE_SIZE/16, args.camera_height * GLOBAL_SCALE])

# displacement landscape
DISPLACEMENT_LANDSCAPE_TYPE = "rocks_noise"  # candidates: rocks_noise, ant_turbulence
DISPLACEMENT_LANDSCAPE_HEIGHT = 0.02 * SCENE_SIZE
DISPLACEMENT_LANDSCAPE_NOISE_SIZE = 1

# displacement texture
DISPLACEMENT_STRENTH = args.disp_scale * SCENE_SCALE

# displacement subdivision
DISPLACEMENT_ADAPTIVE_SUBDIVISION = args.use_adaptive_disp
DISPLACEMENT_SUBDIVISION_LEVEL = 9

# displacement height calculation
DISPLACEMENT_FINE_DENSITY = args.fine_density
DISPLACEMENT_MAX_BIN_SEARCH_ITER = 10
DISPLACEMENT_VOL_TOLERANCE = 0.05 # %5
DISPLACEMENT_HAIR_SYS_NUM_ROCK_TYPE = args.hair_sys_variaty
DISPLACEMENT_HAIR_SYS_PARTICLES_DENSITY = args.hair_sys_density # per square meter 
DISPLACEMENT_UV_SCALE = (args.disp_uv_scale * SCENE_SCALE, args.disp_uv_scale * SCENE_SCALE)

ADAPTIVE_SUBDIVISION_DICING_RATE = 1.36

LABEL_AREA_THRESHOLD = args.label_thresh

LABEL_NO_FINE_PATH = f"{C.ROOT_DIR}/outputs/{TAG}/label-no-fine"
LABEL_FINE_PATH = f"{C.ROOT_DIR}/outputs/{TAG}/label"
OUTPUT_PATH = f"{C.ROOT_DIR}/outputs/{TAG}"

SHAKE_TIME = 3


if __name__=='__main__':

    # add on management
    A.register()

    # set up clean scene
    U.reset_all()
    U.unselect_all()

    fine_grain_material_path = np.random.choice(glob.glob(C.FINE_MATERIAL_PATH))
    fine_grain_material_name = os.path.split(fine_grain_material_path)[-1]
    fine_grain_material = M.load_material(name=fine_grain_material_name, path=fine_grain_material_path, is_ballast=False)

    if args.use_mixed_mats:
        max_coverage_cdf = example_gradations["FI39"].cdf[0.375]
        min_coverage_cdf = example_gradations["FI7"].cdf[0.375]
        coverage_thresh_center = ((min_coverage_cdf - input_gradation.cdf[0.375]) / (max_coverage_cdf - min_coverage_cdf) + 1) * 0.95
        # print(coverage_thresh_center)
        ballast_materials = [M.load_mixed_ballast_mat(name=os.path.split(path)[-1], 
                                ballast_mat_path=path, 
                                fine_mat_path=fine_grain_material_path,
                                z_thresh_center=coverage_thresh_center, z_thresh_width=args.mixed_mats_z_thresh_width)
            # z_thresh_center: [-1, 1], -1: full coverage, 1: zero coverage
            # z_thresh_width: coverage variaty
            for path in np.random.choice(glob.glob(C.BALLAST_MATERIAL_PATH), 
                size=np.random.randint(args.min_num_ballast_mats, args.max_num_ballast_mats+1))]
    else:
        ballast_materials = [M.load_material(name=os.path.split(path)[-1], path=path, is_ballast=True)
            for path in np.random.choice(glob.glob(C.BALLAST_MATERIAL_PATH), 
                size=np.random.randint(args.min_num_ballast_mats, args.max_num_ballast_mats+1))]
    
    transparent_material = M.transparent_material()

    env_lightings = glob.glob(C.ENV_LIGHTING_PATH)
    np.random.shuffle(env_lightings)
    env_tex_node, env_mapping_node = U.add_environment_texture(env_lightings[0])

    # initialize default scene
    scene = Scene(ref=bpy.context.scene)
    scene.meta["size"] = SCENE_SIZE
    scene.meta["global_scale"] = GLOBAL_SCALE
    scene.meta["max_frames"] = MAX_FRAME_NUM

    # set world rigid body for scene
    bpy.ops.rigidbody.world_add()
    scene.ref.rigidbody_world.time_scale = 1
    scene.ref.rigidbody_world.substeps_per_frame = 10
    scene.ref.rigidbody_world.solver_iterations = 10

    # set renderer
    scene.ref.render.engine = 'CYCLES' #  'BLENDER_EEVEE', 'BLENDER_WORKBENCH', 'CYCLES'    
    scene.ref.cycles.feature_set = 'EXPERIMENTAL'
    scene.ref.cycles.samples = CYCLES_SAMPLES_IMAGE
    scene.ref.cycles.device = CYCLES_DEVICE
    if CYCLES_DEVICE == 'GPU':
        bpy.context.preferences.addons[
            "cycles"
        ].preferences.compute_device_type = "CUDA" # or "OPENCL"
        # Set the device_type
        # get_devices() to let Blender detects GPU device
        bpy.context.preferences.addons["cycles"].preferences.get_devices()
        print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
    
    # set output image shape
    scene.ref.render.resolution_x = IMAGE_WIDTH
    scene.ref.render.resolution_y = IMAGE_HEIGHT
    
    # create base landscape
    bpy.ops.mesh.primitive_plane_add(size=SCENE_SIZE)
    base = bpy.context.selected_objects[0]
    base.location = (0, 0, -.1 * SCENE_SIZE)
    U.subdivision(base, level=8)
    bpy.ops.mesh.landscape_add(
        mesh_size_x=SCENE_SIZE, 
        mesh_size_y=SCENE_SIZE, 
        refresh=True, 
        height=BASE_LANDSCAPE_HEIGHT,
        noise_type=BASE_LANDSCAPE_TYPE,
        noise_size=BASE_LANDSCAPE_NOISE_SIZE
    )
    base_landscape = bpy.context.selected_objects[0]
    base_landscape.location = (0, 0, BASE_LANDSCAPE_Z)
    base_landscape_for_height_calc = U.copy_obj(base_landscape)
    U.focus(base_landscape)
    # build base landscape mesh 
    base_landscape = U.build_mesh_from_two_square_planes(top=base_landscape, bottom=base, size=SCENE_SIZE)

    # create rigid body
    U.set_active(base_landscape)
    bpy.ops.rigidbody.object_add(type='PASSIVE')
    bpy.ops.rigidbody.shape_change(type='MESH')
    base_landscape.rigid_body.use_margin = True 
    base_landscape.rigid_body.collision_margin = 0.00001
    base_landscape.rigid_body.friction = 0.8
    
    M.assign_material(base_landscape, fine_grain_material)

    # create camera
    bpy.ops.object.camera_add(location=CAMERA_LOCATION)
    camera = bpy.context.selected_objects[0]
    if args.line_scan_mode:
        camera.data.type = "ORTHO"
        camera.data.ortho_scale = 1.0
    else:
        camera.data.type = "PERSP"
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.001, location=CAMERA_LOOK_AT)
    camera_point_to = bpy.context.selected_objects[0]
    camera_point_to.hide_render = True
    U.camera_look_at(obj_camera=camera, point=camera_point_to.location)
    scene.register("camera", camera)
    scene.register("camera_point_to", camera_point_to)

    # create transparent box to constraint particle system
    bpy.ops.mesh.primitive_cube_add(size=SCENE_SIZE, location=(0, 0, SCENE_SIZE))
    trans_box = bpy.context.selected_objects[0]
    trans_box.scale = (0.95, 0.95, 2)
    M.assign_material(trans_box, transparent_material)
    bpy.ops.rigidbody.object_add(type='PASSIVE')
    bpy.ops.rigidbody.shape_change(type='MESH')  
    trans_box.rigid_body.use_margin = True 
    trans_box.rigid_body.collision_margin = 0.00001
    
    # generate ballast
    ballast_meshes = U.traverse_path(C.BALLAST_MESH_PATH, ["obj", "fbx"])

    print(ballast_meshes)

    num_meshes = min(args.max_num_meshes, len(ballast_meshes))
    ballast_meshes = np.random.choice(ballast_meshes, num_meshes, replace=False)
    high_poly_ballasts = []
    low_poly_ballasts = []

    for i in range(num_meshes):
        mesh_file = ballast_meshes[i]

        # import high resolution meshes
        high_poly_ballast = U.import_obj(mesh_file)
        bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME')
        high_poly_ballast.scale = LOADING_MESH_DEFAULT_SCALE
        high_poly_ballast.location = (LOADING_MESH_START_X + i, 0, 0)
        scene.register(f"high_poly_ballast_{i}", high_poly_ballast)
        
        # simplify loaded_mesh
        U.simplify_mesh(high_poly_ballast, keep_face_ratio=HIGH_POLY_MESH_KEEP_FACE_RATIO)
        
        # initialize rigid body and calculate volume and mass
        bpy.ops.rigidbody.object_add(type='ACTIVE')
        bpy.ops.rigidbody.shape_change(type="CONVEX_HULL")
        scene.meta[f"high_poly_ballast_{i}"]['d'] = G.diameter_calculate(high_poly_ballast, inch=True)
        loading_scale = MU.meter2inch(0.001) / scene.meta[f"high_poly_ballast_{i}"]['d']

        print(loading_scale)
        for ii in range(3):
            high_poly_ballast.scale[ii] = LOADING_MESH_DEFAULT_SCALE[ii] * loading_scale 
        scene.meta[f"high_poly_ballast_{i}"]['d'] = G.diameter_calculate(high_poly_ballast, inch=True)
        U.focus(high_poly_ballast)
        bpy.ops.rigidbody.object_remove()

        
        high_poly_ballasts.append(high_poly_ballast)

        # convert to low resolution meshes
        low_poly_ballast = U.copy_obj(high_poly_ballast)
        U.focus(low_poly_ballast)
        bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME')
        low_poly_ballast.scale = high_poly_ballast.scale
        low_poly_ballast.location = (LOADING_MESH_START_X + i, 0, 1)
        scene.register(f"low_poly_ballast_{i}", low_poly_ballast)

        # decrease number of polygons and clean up the mesh
        U.set_active(low_poly_ballast)
        U.simplify_mesh(low_poly_ballast, keep_face_ratio = LOW_POLY_MESH_KEEP_FACES / len(low_poly_ballast.data.polygons))
        U.clean_mesh(low_poly_ballast)

        low_poly_ballasts.append(low_poly_ballast)
    
    synthetic_ballast_diameters = input_gradation.gen_ballast(total_ballast=NUM_BALLAST) # inch
    synthetic_ballast_diameters.sort(reverse=True)
    scene.meta["expected_fine_size"] = input_gradation.expected_fine_size

    num_ballasts = len(synthetic_ballast_diameters)
    ballasts = []
    ballast_types = np.random.randint(0, num_meshes, num_ballasts).tolist()

    scene.ref.frame_start = 0
    scene.ref.frame_end = MAX_FRAME_NUM
    scene.ref.frame_current = 0
    scene.ref.frame_set(0)

    # to boost the simulation and avoid mesh overlapping, simulate some frames when generate ballasts
    frame_step = max(1, round(num_ballasts / BALLAST_FALLING_FRAME_PHASE_I + 0.5))

    for i in tqdm.trange(num_ballasts):

        ballast_type = ballast_types[i]

        ballast = U.copy_obj(low_poly_ballasts[ballast_type])
        U.focus(ballast)
        
        # random location
        ballast.location = (
            np.random.uniform(-BALLAST_ZONE_SIZE, BALLAST_ZONE_SIZE), 
            np.random.uniform(-BALLAST_ZONE_SIZE, BALLAST_ZONE_SIZE), 
            np.random.uniform(BALLAST_ZONE_HEIGHT_START, BALLAST_ZONE_HEIGHT_START + BALLAST_ZONE_HEIGHT)
        )    
        
        # random rotation
        angle = np.random.uniform(0, 2*math.pi)
        axis = np.random.uniform(0, 1000, 3) + 1e-10
        axis = axis / np.linalg.norm(axis)

        ballast.rotation_mode = "AXIS_ANGLE"
        ballast.rotation_axis_angle = (angle, axis[0], axis[1], axis[2])

        # create rigid body
        bpy.ops.rigidbody.object_add(type='ACTIVE')
        bpy.ops.rigidbody.shape_change(type="CONVEX_HULL")
        ballast.rigid_body.use_margin = True 
        ballast.rigid_body.collision_margin = 0.0001
        
        d = scene.meta[f"high_poly_ballast_{ballast_type}"]['d']        
        scale = synthetic_ballast_diameters[i] * GLOBAL_SCALE / d * ballast.scale[0] 
        ballast.scale = (scale, scale, scale) 
        ballast.name = f"ballast_d_{synthetic_ballast_diameters[i]:.3f}_t_{ballast_type}_rd_{d:.3f}"


        bpy.ops.rigidbody.mass_calculate(material="Custom", density=G.BALLAST_DENSITY)   
        
        ballasts.append(ballast)

        # simulate some frame when generate ballasts
        if i % frame_step == 0 and scene.ref.frame_current < BALLAST_FALLING_FRAME_PHASE_I:
            scene.ref.frame_set(scene.ref.frame_current + 1)

    # apply phase I transformation and go to phase II
    U.unselect_all()
    for ballast in ballasts:
        ballast.select_set(True)
    U.set_active(ballasts[0])    
    bpy.ops.object.visual_transform_apply()


    scene.ref.frame_set(0)
    rand_a = np.random.uniform(-3, 3, 2)
    flag = 1
    shake_completed = 0.5
    for i in tqdm.trange(BALLAST_FALLING_FRAME_PHASE_II):

        if i % (BALLAST_FALLING_FRAME_PHASE_II // 10) == 0:
            flag = -flag
            shake_completed += 0.5

        if int(shake_completed+0.1) >= SHAKE_TIME or i > BALLAST_FALLING_FRAME_PHASE_II // 2:
            bpy.context.scene.gravity = (0, 0, -9.8)
        else:  
            bpy.context.scene.gravity[2] = flag * 5
        
        scene.ref.frame_set(i)
    bpy.ops.object.visual_transform_apply()
    scene.ref.frame_set(0)
    
    print("================== Start Filter Invalid Ballast ==================")
    # calculate valid ballast number 
    valid_ballast_num = num_ballasts
    invalid_indices = []
    for (i, ballast) in enumerate(ballasts):
        # M.assign_material(ballast, np.random.choice(ballast_materials))
        ballast.rigid_body.type = "PASSIVE"
        if ballast.location[2] < 0:
            valid_ballast_num -= 1
            invalid_indices.append(i)
    
    for invalid_i in reversed(invalid_indices):
        ballasts.pop(invalid_i)
        ballast_types.pop(invalid_i)
        synthetic_ballast_diameters.pop(invalid_i)
    
    print(f"valid percentage: {valid_ballast_num/num_ballasts*100:.2f}%")  
    num_ballasts = valid_ballast_num

    print("================== Complete Filtering Invalid Ballast ==================")   

    coarse_group = U.copy_obj(ballasts[0])
    U.unselect_all()
    U.set_active(None)
    for ballast in ballasts[1:]:
        tmp_ballast = U.copy_obj(ballast)
        ballast.select_set(False)
        tmp_ballast = tmp_ballast.select_set(True)
        coarse_group.select_set(True)
        U.set_active(coarse_group)  
        bpy.ops.object.join()

    U.focus(coarse_group)
    bpy.ops.rigidbody.object_remove()

    if not DISPLACEMENT_ON_LANDSCAPE:
        bpy.ops.mesh.primitive_plane_add(size=SCENE_SIZE)
    else:
        bpy.ops.mesh.landscape_add(
            mesh_size_x=SCENE_SIZE, 
            mesh_size_y=SCENE_SIZE, 
            refresh=True, 
            height=DISPLACEMENT_LANDSCAPE_HEIGHT,
            noise_type=DISPLACEMENT_LANDSCAPE_TYPE,
            noise_size=DISPLACEMENT_LANDSCAPE_NOISE_SIZE
        )
        
    displacement_layer = bpy.context.selected_objects[0]
    if DISPLACEMENT_ADAPTIVE_SUBDIVISION:
        fine_grain_material.cycles.displacement_method = "BOTH"
        coarse_layer = U.copy_obj(displacement_layer)
        # enable adaptive subdivision
        U.subdivision(displacement_layer, adaptive=True)


    else:
        fine_grain_material.cycles.displacement_method = "BUMP"
        coarse_layer = U.copy_obj(displacement_layer)
        
        M.add_displacement(coarse_layer, 
            disp_map_path=fine_grain_material.node_tree.nodes["tex_displacement"].image.filepath_raw,
            strength=DISPLACEMENT_STRENTH,
            subdivision_level=5)

        M.add_displacement(displacement_layer, 
            disp_map_path=fine_grain_material.node_tree.nodes["tex_displacement"].image.filepath_raw,
            strength=DISPLACEMENT_STRENTH,
            subdivision_level=DISPLACEMENT_SUBDIVISION_LEVEL)

        bpy.ops.object.shade_smooth()
        
    displacement_layer.location = (0, 0, - SCENE_SIZE)

    U.unselect_all()
    U.set_active(None)
    bpy.ops.mesh.primitive_plane_add(size=SCENE_SIZE)
    base = bpy.context.selected_objects[0]
    U.subdivision(base, level = 5)
    base.location = (0, 0, 0)

    """Binary Search to find the fine displacement layer height"""
    print("================== Start Binary Search ==================")
    input_fine_vol = input_gradation.get_fine_vol(inch=False) * (GLOBAL_SCALE**3)
    start, end = U.get_highest_vertice_z(base_landscape_for_height_calc) + DISPLACEMENT_STRENTH / 2, SCENE_SIZE
    bin_search_iters = 0
    volume_err = math.inf
    disp_height = (start + end) / 2
    while bin_search_iters < DISPLACEMENT_MAX_BIN_SEARCH_ITER and volume_err > DISPLACEMENT_VOL_TOLERANCE:
        disp_height = (start + end) / 2
        U.focus(coarse_layer)
        current_vol = G.calc_fine_volume(
            height=disp_height,
            disp_obj=coarse_layer,
            base_obj=base,
            ballasts_obj_list=[coarse_group],
            scene_size=SCENE_SIZE,
            valid_rate=DISPLACEMENT_FINE_DENSITY,
            operations=["DIFFERENCE"]
        )
        volume_diff = current_vol - input_fine_vol

        if volume_diff > 0:
            end = disp_height
        elif volume_diff < 0:
            start = disp_height
        else:
            break
        
        volume_err = np.abs(volume_diff / input_fine_vol)
        bin_search_iters += 1

        print(f"BIN SEARCH ITER {bin_search_iters} -", "Height:", disp_height, "Err:", np.round(volume_err * 100, 3))
        print(f"    Cur vol: {current_vol:.4f}, Input vol: {input_fine_vol:.4f}")
    print("================== Binary Search Completed ==================")

    U.delete_obj(base_landscape_for_height_calc)
    displacement_layer.location = (0, 0, disp_height)
    M.assign_material(displacement_layer, fine_grain_material)
    M.scale_UV(displacement_layer, DISPLACEMENT_UV_SCALE)
    M.rotate_UV(displacement_layer, np.random.uniform(low=15, high=75))
    
    # 1. displacement _layer + passive rigid body
    U.focus(displacement_layer)
    bpy.ops.rigidbody.object_add(type="PASSIVE")
    bpy.ops.rigidbody.shape_change(type="MESH")
    displacement_layer.rigid_body.use_margin = True 
    displacement_layer.rigid_body.collision_margin = 0.0001

    fine_ballasts = []
    num_fine_ballasts = args.top_fine_num
    max_fine_diameter = args.top_fine_max_d
    min_fine_diameter = args.top_fine_min_d
    current_disp_scale_z = displacement_layer.scale[2]
    displacement_shift = MU.inch2meter(args.top_fine_disp_shift) * GLOBAL_SCALE
    min_displacement_scaling = (DISPLACEMENT_STRENTH/2-displacement_shift) / (DISPLACEMENT_STRENTH / 2)
    print("min_displacement_scale:", min_displacement_scaling)
    if 0 < min_displacement_scaling < 1:
        displacement_layer.scale[2] = np.random.uniform(min_displacement_scaling, 1.0) * current_disp_scale_z
    displacement_layer.location = (0, 0, disp_height - displacement_shift)

    fine_types = np.random.randint(0, num_meshes, num_fine_ballasts).tolist()
    fine_diameters = np.random.uniform(min_fine_diameter, max_fine_diameter, num_fine_ballasts)
    scene.ref.frame_set(0)
    for i in tqdm.trange(num_fine_ballasts):
        fine_type = fine_types[i]

        fine = U.copy_obj(low_poly_ballasts[fine_type])
        U.focus(fine)
        
        # random location
        fine.location = (
            np.random.uniform(-SCENE_SIZE * 0.5, SCENE_SIZE * 0.5), 
            np.random.uniform(-SCENE_SIZE * 0.5, SCENE_SIZE * 0.5), 
            np.random.uniform(disp_height + DISPLACEMENT_STRENTH / 2, min(1.8 * SCENE_SIZE, disp_height + DISPLACEMENT_STRENTH / 2 + BALLAST_ZONE_HEIGHT))
        )    
        
        # random rotation
        angle = np.random.uniform(0, 2*math.pi)
        axis = np.random.uniform(0, 1000, 3) + 1e-10
        axis = axis / np.linalg.norm(axis)

        fine.rotation_mode = "AXIS_ANGLE"
        fine.rotation_axis_angle = (angle, axis[0], axis[1], axis[2])

        # create rigid body
        bpy.ops.rigidbody.object_add(type='ACTIVE')
        bpy.ops.rigidbody.shape_change(type="CONVEX_HULL")
        fine.rigid_body.use_margin = True 
        fine.rigid_body.collision_margin = 0.0001
        fine.name = f"top_fine_d_{fine_diameters[i]:.3f}_t_{fine_type}_rd_{d:.3f}"
        
        d = scene.meta[f"high_poly_ballast_{fine_type}"]['d']    
        scale = fine_diameters[i] * GLOBAL_SCALE / d * fine.scale[0] 
        fine.scale = (scale, scale, scale) 

        bpy.ops.rigidbody.mass_calculate(material="Custom", density=G.BALLAST_DENSITY)   
        
        fine_ballasts.append(fine)

        # simulate some frame when generate ballasts
        if i % frame_step == 0 and scene.ref.frame_current < BALLAST_FALLING_FRAME_PHASE_I:
            scene.ref.frame_set(scene.ref.frame_current + 1)

    # apply phase I transformation and go to phase II
    U.unselect_all()
    for fine in fine_ballasts:
        fine.select_set(True)
    bpy.ops.object.visual_transform_apply()
    U.set_active(fine_ballasts[0])

    scene.ref.frame_set(0)
    for i in tqdm.trange(BALLAST_FALLING_FRAME_PHASE_II):        
        scene.ref.frame_set(i)
    bpy.ops.object.visual_transform_apply()
    scene.ref.frame_set(0)  
    
    # assign back high poly meshes to ballasts, assign materials to ballasts
    for fine in fine_ballasts:
        M.assign_material(fine, np.random.choice(ballast_materials))
        fine.rigid_body.type = "PASSIVE"  

    valid_fine_num = num_fine_ballasts
    invalid_indices = []
    for (i, fine) in enumerate(fine_ballasts):
        M.assign_material(fine, np.random.choice(ballast_materials))
        fine.rigid_body.type = "PASSIVE"
        if fine.location[2] < 0:
            valid_fine_num -= 1
            invalid_indices.append(i)
    
    for invalid_i in reversed(invalid_indices):
        fine_types.pop(invalid_i)
        fine_ballasts.pop(invalid_i)

    displacement_layer.scale[2] = current_disp_scale_z
    displacement_layer.location = (0, 0, disp_height)

    print(f"valid percentage (fine): {valid_fine_num/num_fine_ballasts*100:.2f}%")  
    num_fine_ballasts = valid_ballast_num

    coarse_group_fine = U.copy_obj(fine_ballasts[0])
    U.unselect_all()
    U.set_active(None)
    for fine in fine_ballasts[1:]:
        tmp_ballast = U.copy_obj(fine)
        fine.select_set(False)
        tmp_ballast = tmp_ballast.select_set(True)
        coarse_group_fine.select_set(True)
        U.set_active(coarse_group_fine)  
        bpy.ops.object.join()

    U.focus(coarse_group_fine)
    bpy.ops.rigidbody.object_remove()

    print("Before add top fine:", current_vol)

    current_vol = G.calc_fine_volume(
        height=disp_height,
        disp_obj=coarse_layer,
        base_obj=base,
        ballasts_obj_list=[coarse_group, coarse_group_fine],
        scene_size=SCENE_SIZE,
        valid_rate=DISPLACEMENT_FINE_DENSITY,
        operations=["DIFFERENCE", "UNION"]
    )

    print("After add top fine:", current_vol)


    U.delete_obj(coarse_group)
    U.delete_obj(coarse_group_fine)
    U.delete_obj(coarse_layer)
    U.delete_obj(base)

    # get final mesh size of fines
    synthetic_fine_vol = current_vol * MU.meter2inch(1/GLOBAL_SCALE) ** 3

    # TODO: re-calculate gradation
    print("================== Start Gradation Calibration ==================")
    print("    Input gradation: ", input_gradation.cdf)
    input_gradation.calibrate_from_diameters(d_list=synthetic_ballast_diameters, fine_vol=synthetic_fine_vol)
    print("    Synthetic gradation: ", input_gradation.cdf)
    print("================== Gradation Calibration Completed ==================")

    
    # assign back high poly meshes to ballasts, assign materials to ballasts
    for (ballast_type, ballast) in zip(ballast_types, ballasts):
        ballast.select_set(True)
        ballast.data = high_poly_ballasts[ballast_type].data.copy()
        M.assign_material(ballast, np.random.choice(ballast_materials))

    # TODO: add hair particle system to displacement layer with rock generator
    if DISPLACEMENT_HAIR_SYSTEMS:
        print("================== Start Create Hair Particle System on Displacement Layer ==================")
        
        # step 1: create stone generator collection
        
        U.unselect_all()
        U.set_active(None)

        bpy.ops.mesh.add_mesh_rock(
            num_of_rocks=DISPLACEMENT_HAIR_SYS_NUM_ROCK_TYPE, 
            scale_X=(0.5, 3.25), skew_X=-0.5, scale_Y=(0.5, 3.25), skew_Y=-0.5, scale_Z=(0.5, 3.25), skew_Z=-0.5, 
            use_scale_dis=False, scale_fac=(1, 1, 1), deform=1, rough=5, detail=3, display_detail=3, 
            smooth_fac=2, smooth_it=2, use_generate=True, use_random_seed=True)
        
        bpy.ops.collection.create(name = "Fine_Gen")
        bpy.context.scene.collection.children.link(bpy.data.collections["Fine_Gen"])
        fine_collection = bpy.data.collections["Fine_Gen"]

        for (i, obj) in enumerate(bpy.context.selected_objects):
            obj.location = (8*SCENE_SIZE, 8*SCENE_SIZE + i, 0)

            U.focus(obj)
            bpy.ops.rigidbody.object_add(type="ACTIVE")
            obj.rigid_body.collision_shape = "MESH"
            d = G.diameter_calculate(ballast_obj=obj) 
            scale = scene.meta["expected_fine_size"] * GLOBAL_SCALE / d
            obj.scale = (scale, scale, scale)
            U.focus(obj)
            bpy.ops.rigidbody.object_remove()

            M.assign_material(obj, np.random.choice(ballast_materials))

        # step 2: create hair particle system

        U.focus(displacement_layer)
        bpy.ops.object.particle_system_add()
        displacement_layer.particle_systems[0].settings.type = "HAIR"
        displacement_layer.particle_systems[0].settings.count = round(DISPLACEMENT_HAIR_SYS_PARTICLES_DENSITY * (SCENE_SIZE/GLOBAL_SCALE)**2)
        displacement_layer.particle_systems[0].settings.use_advanced_hair = True
        displacement_layer.particle_systems[0].settings.use_rotations = True
        displacement_layer.particle_systems[0].settings.phase_factor = 1.0
        displacement_layer.particle_systems[0].settings.phase_factor_random = 2.0
        displacement_layer.particle_systems[0].settings.distribution = "RAND"
        displacement_layer.particle_systems[0].settings.render_type = "COLLECTION"
        displacement_layer.particle_systems[0].settings.instance_collection = fine_collection
        displacement_layer.particle_systems[0].settings.use_collection_pick_random = True
        displacement_layer.particle_systems[0].settings.use_scale_instance = True 
        displacement_layer.particle_systems[0].settings.particle_size = 2.0
        displacement_layer.particle_systems[0].settings.size_random = 1.0
        print("================== Hair Particle System on Displacement Layer Created ==================")
        
    scene.ref.camera = camera
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)  
    
    if RENDER_RESULT:
        for (i, env_light) in enumerate(env_lightings):
            parent, env_name = os.path.split(env_light)

            env_tex_node.image = bpy.data.images.load(env_light)
            env_mapping_node.inputs[2].default_value[2] = np.deg2rad(np.random.uniform(0, 360))

            if i == 0:
                R.render_scene(scene=scene, 
                    out_path=f"{OUTPUT_PATH}/{env_name[:-4]}", 
                    render_label=RENDER_LABELS, 
                    label_objects=ballasts + fine_ballasts,
                    mask_layer = displacement_layer
                )
                R.filter_labels(f"{OUTPUT_PATH}/{env_name[:-4]}",
                    fine_id_start=len(ballasts),
                    thresh=LABEL_AREA_THRESHOLD,
                    verbose=args.output_label_details,
                    keep_largest_contour=args.keep_largest_label,
                    min_num_pixel_per_label=args.min_num_pixel_per_label)
                if args.render_one_env_only:
                    break
            else:
                R.render_scene(scene=scene, out_path=f"{OUTPUT_PATH}/{env_name[:-4]}")
   
    if CAMERA_USE_TRAJECTORY:
        # camera follow a trajectory
        for (i, env_light) in enumerate(env_lightings):
            parent, env_name = os.path.split(env_light)
            scene.ref.frame_current = 0
            env_tex_node.image = bpy.data.images.load(env_light)

            if i == 0:
                R.render_scene_by_camera_trajectory(scene, 
                    start=CAMERA_TRAJECTORY_START, 
                    end=CAMERA_TRAJECTORY_END, 
                    num_images=CAMERA_TRAJECTORY_NUM_IMAGES, 
                    out_path=f"{OUTPUT_PATH}/trajectory/{env_name[:-4]}",
                    render_label=RENDER_RESULT,
                    label_objects=ballasts + fine_ballasts,
                    mask_layer = displacement_layer
                )
                R.filter_labels(f"{OUTPUT_PATH}/trajectory/{env_name[:-4]}", 
                    fine_id_start=len(ballasts), 
                    thresh=LABEL_AREA_THRESHOLD,
                    verbose=args.output_label_details,
                    keep_largest_contour=args.keep_largest_label,
                    min_num_pixel_per_label=args.min_num_pixel_per_label)
                if args.render_one_env_only:
                    break
            else:
                R.render_scene_by_camera_trajectory(scene, 
                    start=CAMERA_TRAJECTORY_START, 
                    end=CAMERA_TRAJECTORY_END, 
                    num_images=CAMERA_TRAJECTORY_NUM_IMAGES, 
                    out_path=f"{OUTPUT_PATH}/trajectory/{env_name[:-4]}",
                    render_label=False,
                    label_objects=None,
                    mask_layer=None
                )

    if args.save_scene:
        bpy.ops.wm.save_userpref()
        bpy.ops.wm.save_mainfile(filepath=f"{OUTPUT_PATH}/displacement.blend")
        json.dump(scene.object_dict, open(f"{OUTPUT_PATH}/scene_obj.json", 'w+'), indent=4)
        json.dump(scene.meta, open(f"{OUTPUT_PATH}/scene_meta.json", 'w+'), indent=4)

    if args.save_gradation:
        json.dump(input_gradation.input_cdf, open(f"{OUTPUT_PATH}/input_gradation.json", 'w+'), indent=4)
        json.dump(input_gradation.cdf, open(f"{OUTPUT_PATH}/gradation.json", 'w+'), indent=4)
        input_gradation.draw_gradation(f"{OUTPUT_PATH}/gradation.png")
    
    print(f"Synthetic data saved at {OUTPUT_PATH}.")