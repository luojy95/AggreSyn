import sys 
sys.path.append(".")

import os
import glob
import math
import json
import tqdm
import numpy as np
from threading import Thread

import bpy

from scripts import common
from scripts import math_utils as MU
from scripts.scene import Scene
from scripts.gradation_helper import MAX_DIAMETER, test_gradation
from scripts.addon_helper import AddonHelper as A
from scripts.gradation_helper import GradationHelper as G
from scripts.material_helper import MaterialHelper as M
from scripts.blender_utils import BlenderUtils as U


GLOBAL_SCALE = 10
SCENE_SIZE = 1.0 * GLOBAL_SCALE
BALLAST_ZONE_SIZE = SCENE_SIZE / 4
BALLAST_ZONE_HEIGHT =  SCENE_SIZE / 2

NUM_BALLAST = 100

MAX_FRAME_NUM = 180

BALLAST_FALLING_FRAME_PHASE_I = 200
BALLAST_FALLING_FRAME_PHASE_II = 30

MOL_FRAME_START = 1
MOL_FRAME_END = 100
MOL_FRAME_OFFSET = 25
MOL_SUBSTEP = 4
MOL_MONITORING_SEC = 15

FINE_NUM_MAX_MOL = 1000000

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

CYCLES_DEVICE = 'GPU'
CYCLES_SAMPLES_LABEL = 1
CYCLES_SAMPLES_IMAGE = 64

LOADING_MESH_DEFAULT_SCALE = (.001, .001, .001)
LOADING_MESH_START_X = 2 * SCENE_SIZE 
HIGH_POLY_MESH_KEEP_FACE_RATIO = 0.2
LOW_POLY_MESH_KEEP_FACES = 50

np.random.seed(0)

if __name__=='__main__':

    # add on management
    A.register()

    # set up clean scene
    U.reset_all()
    U.unselect_all()

    # load materials from materials.blend
    blendfile = os.path.join(common.ROOT_DIR, "materials.blend")
    section   = "/Material/"
    objects    = ["White", "Transparent"]

    for object in objects:

        filepath  = blendfile + section + object
        directory = blendfile + section
        filename  = object
        bpy.ops.wm.append(
            filepath=filepath, 
            filename=filename,
            directory=directory
        )

    ground_material = bpy.data.materials["White"]
    ballast_materials = [bpy.data.materials["White"]]
    fine_grain_material = bpy.data.materials["White"]
    transparent_material = bpy.data.materials["Transparent"]

    # initialize default scene
    scene = Scene(ref=bpy.context.scene)

    scene.meta["size"] = SCENE_SIZE
    scene.meta["global_scale"] = GLOBAL_SCALE
    scene.meta["max_frames"] = MAX_FRAME_NUM

    # set world rigid body for scene
    bpy.ops.rigidbody.world_add()
    scene.ref.rigidbody_world.time_scale = 10
    scene.ref.rigidbody_world.substeps_per_frame = 10
    scene.ref.rigidbody_world.solver_iterations = 1

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

    # create base box (solid ground)
    bpy.ops.mesh.primitive_cube_add(size=SCENE_SIZE, location=(0,0, -0.1 * SCENE_SIZE), scale=(1, 1, 0.2))
    base_box = bpy.context.selected_objects[0]    
    bpy.ops.rigidbody.object_add(type='PASSIVE')
    base_box.rigid_body.collision_shape = 'CONVEX_HULL'  
    
    # create base landscape
    bpy.ops.mesh.primitive_plane_add(size=SCENE_SIZE)
    base = bpy.context.selected_objects[0]
    base.location = (0, 0, -0.1 * SCENE_SIZE)
    bpy.ops.mesh.landscape_add(
        mesh_size_x=SCENE_SIZE, 
        mesh_size_y=SCENE_SIZE, 
        refresh=True, 
        height=0.02 * SCENE_SIZE,
        noise_type="ant_turbulence",
        noise_size=5
    )
    base_landscape = bpy.context.selected_objects[0]
    base_landscape.location = (0, 0, 0)
    M.assign_material(base_landscape, ground_material)

    # build base landscape mesh 
    base_landscape = U.build_mesh_from_two_square_planes(top=base_landscape, bottom=base, size=SCENE_SIZE)

    # create rigid body
    U.set_active(base_landscape)
    bpy.ops.rigidbody.object_add(type='PASSIVE')
    bpy.ops.rigidbody.shape_change(type='MESH')
    base_landscape.rigid_body.use_margin = True 
    base_landscape.rigid_body.collision_margin = 0.0001

    # create light source
    bpy.ops.object.light_add(type="SUN", radius=SCENE_SIZE, location=(SCENE_SIZE/4, SCENE_SIZE/4, SCENE_SIZE/2))
    light = bpy.context.selected_objects[0]

    # create camera
    bpy.ops.object.camera_add(location=(0, 0, 0.5 * GLOBAL_SCALE))
    camera = bpy.context.selected_objects[0]
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.001, location=(0, 0, 0))
    camera_point_to = bpy.context.selected_objects[0]
    camera_point_to.hide_render = True
    U.camera_look_at(obj_camera=camera, point=camera_point_to.location)

    scene.register("light", light)
    scene.register("camera", camera)

    # create transparent box to constraint particle system
    bpy.ops.mesh.primitive_cube_add(size=SCENE_SIZE, location=(0, 0, SCENE_SIZE/2))
    trans_box = bpy.context.selected_objects[0]
    M.assign_material(trans_box, transparent_material)
    bpy.ops.rigidbody.object_add(type='PASSIVE')
    bpy.ops.rigidbody.shape_change(type='MESH')  
    bpy.ops.object.modifier_add(type='COLLISION')
    trans_box.collision.damping_factor = 0.6
    trans_box.collision.friction_factor = 0.2
    trans_box.show_instancer_for_viewport = False
    
    # generate ballast
    ballast_meshes = glob.glob(os.path.join(common.ROCK_MESH_DIR, "*.obj"))
    num_meshes = len(ballast_meshes)
    high_poly_ballasts = []
    low_poly_ballasts = []

    for i in range(num_meshes):
        mesh_file = ballast_meshes[i]

        # import high resolution meshes
        high_poly_ballast = U.import_obj(mesh_file)
        high_poly_ballast.scale = (0.001, 0.001, 0.001)   
        high_poly_ballast.location = (2 * SCENE_SIZE + i, 0, 0)
        scene.register(f"high_poly_ballast_{i}", high_poly_ballast)
        
        high_poly_ballasts.append(high_poly_ballast)


        # convert to low resolution meshes
        low_poly_ballast = U.import_obj(mesh_file)
        low_poly_ballast.location = (2*SCENE_SIZE + i, 0, 1)
        low_poly_ballast.scale = (0.001, 0.001, 0.001)   
        scene.register(f"low_poly_ballast_{i}", low_poly_ballast)

        # decrease number of polygons and clean up the mesh
        U.set_active(low_poly_ballast)
        bpy.ops.object.modifier_add(type='DECIMATE')
        num_facets = len(low_poly_ballast.data.polygons)
        low_poly_ballast.modifiers['Decimate'].ratio = 50 / num_facets
        bpy.ops.object.modifier_apply(modifier=low_poly_ballast.modifiers['Decimate'].name)
        bpy.ops.object.mode_set(mode="EDIT")
        
        bpy.ops.mesh.select_mode(type='VERT')
        bpy.ops.mesh.select_loose()
        bpy.ops.mesh.delete(type='VERT')

        bpy.ops.mesh.select_mode(type='EDGE')
        bpy.ops.mesh.select_loose()
        bpy.ops.mesh.delete(type='EDGE')
        
        bpy.ops.mesh.select_mode(type='FACE')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.convex_hull()
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode="OBJECT")

        bpy.ops.object.modifier_add(type='DECIMATE')
        low_poly_ballast.modifiers['Decimate'].ratio = 0.2
        bpy.ops.object.modifier_apply(modifier=low_poly_ballast.modifiers['Decimate'].name)
        bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME')

        low_poly_ballasts.append(low_poly_ballast)
    
    ballast_diameters = test_gradation.gen_ballast(total_ballast=NUM_BALLAST) # inch

    scene.meta["expected_fine_size"] = test_gradation.expected_fine_size

    num_ballasts = len(ballast_diameters)
    ballasts = []
    ballast_types = np.random.randint(0, num_meshes, num_ballasts)

    scene.ref.frame_start = 0
    scene.ref.frame_end = MAX_FRAME_NUM
    scene.ref.frame_current = 0
    scene.ref.frame_set(0)

    # to boost the simulation and avoid mesh overlapping, simulate some frames when generate ballasts
    frame_step = max(1, round(num_ballasts / BALLAST_FALLING_FRAME_PHASE_I + 0.5))

    for i in tqdm.trange(num_ballasts):

        ballast_type = ballast_types[i]

        ballast = U.copy_obj(low_poly_ballasts[ballast_type])
        
        # random location
        ballast.location = (
            np.random.uniform(-BALLAST_ZONE_SIZE, BALLAST_ZONE_SIZE), 
            np.random.uniform(-BALLAST_ZONE_SIZE, BALLAST_ZONE_SIZE), 
            np.random.uniform(GLOBAL_SCALE * MU.inch2meter(MAX_DIAMETER), BALLAST_ZONE_HEIGHT)
        )    
        
        U.set_active(ballast)

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

        if "d" not in scene.meta[f"low_poly_ballast_{ballast_type}"]:
            d = G.diameter_calculate(ballast, inch=True)
            scene.meta[f"low_poly_ballast_{ballast_type}"]["d"] = d
        else:
            d = scene.meta[f"low_poly_ballast_{ballast_type}"]["d"] 
        
        scale = ballast_diameters[i] * GLOBAL_SCALE / d * ballast.scale[0] 
        ballast.scale = (scale, scale, scale)    
        
        U.set_active(None)

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
    for i in tqdm.trange(BALLAST_FALLING_FRAME_PHASE_II):
        scene.ref.frame_set(i)
    bpy.ops.object.visual_transform_apply()
    scene.ref.frame_set(0)

    
    # calculate valid ballast number 
    valid_ballast_num = num_ballasts
    for ballast in ballasts:
        M.assign_material(ballast, np.random.choice(ballast_materials))
        ballast.rigid_body.type = "PASSIVE"
        if ballast.location[2] < 0:
            valid_ballast_num -= 1
    print(f"valid percentage: {valid_ballast_num/num_ballasts*100:.2f}%")

    # assign back high poly meshes to ballasts, assign materials to ballasts
    for (ballast_type, ballast) in zip(ballast_types, ballasts):
        ballast.select_set(True)
        ballast.data = high_poly_ballasts[ballast_type].data.copy()
        M.assign_material(ballast, np.random.choice(ballast_materials))

    # join all ballasts to reduce the number of total meshes
    U.unselect_all()
    U.set_active(None)
    for ballast in ballasts:
        ballast.select_set(True)
    U.set_active(ballasts[0])  
    bpy.ops.object.join()

    ballast_group = ballasts[0]
    bpy.ops.rigidbody.object_remove()
    bpy.ops.object.modifier_add(type='COLLISION')
    ballast_group.collision.damping_factor = 0.7
    ballast_group.collision.friction_factor = 0.7

    U.delete_obj(base_box)

    bpy.ops.mesh.primitive_uv_sphere_add(location=(2*SCENE_SIZE, 2*SCENE_SIZE, 0))
    fine_grain_ball = bpy.context.selected_objects[0]
    M.assign_material(fine_grain_ball, fine_grain_material)
    
    num_fine = min(FINE_NUM_MAX_MOL, test_gradation.fine_grain_cnt)
    expected_fine_size = MU.inch2meter(test_gradation.expected_fine_size)
    print("fine_number:", num_fine)
    print("size:", expected_fine_size)

    scene.ref.mol_substep = MOL_SUBSTEP

    bpy.ops.mesh.primitive_cube_add(size=SCENE_SIZE, scale=(0.1, 0.8, 0.1), location=(-SCENE_SIZE * 0.5 * 0.8, 0, SCENE_SIZE * 0.7))
    emitter_left = bpy.context.selected_objects[0]
    bpy.ops.object.particle_system_add()
    emitter_left.particle_systems[0].name = "LeftEmitter"
    emitter_left.particle_systems[0].settings.name = "LeftEmitterSettings"
    emitter_left.particle_systems[0].settings.frame_start = MOL_FRAME_START
    emitter_left.particle_systems[0].settings.frame_end = MOL_FRAME_END
    emitter_left.particle_systems[0].settings.count =  num_fine // 2 
    emitter_left.particle_systems[0].settings.lifetime = 10000
    emitter_left.particle_systems[0].settings.emit_from = "VOLUME"
    emitter_left.particle_systems[0].settings.object_align_factor[0] = 5
    emitter_left.particle_systems[0].settings.factor_random = .9
    emitter_left.particle_systems[0].settings.use_multiply_size_mass = True
    emitter_left.particle_systems[0].settings.use_size_deflect = True 
    emitter_left.particle_systems[0].settings.render_type = "OBJECT"
    emitter_left.show_instancer_for_render = False
    emitter_left.particle_systems[0].settings.particle_size = expected_fine_size * GLOBAL_SCALE 
    emitter_left.particle_systems[0].settings.size_random = 0.95
    emitter_left.particle_systems[0].settings.instance_object = fine_grain_ball
    emitter_left.particle_systems[0].settings.use_scale_instance = False
    emitter_left.particle_systems[0].settings.use_rotations = True 
    emitter_left.particle_systems[0].settings.rotation_mode = "OB_X"
    emitter_left.particle_systems[0].settings.phase_factor = 0.5
    emitter_left.particle_systems[0].settings.phase_factor_random = 2
    emitter_left.particle_systems[0].settings.mol_active = True
    emitter_left.particle_systems[0].settings.mol_density_active = True
    emitter_left.particle_systems[0].settings.mol_matter = "-1"
    emitter_left.particle_systems[0].settings.mol_density = 2500
    emitter_left.particle_systems[0].settings.mol_selfcollision_active = True
    emitter_left.particle_systems[0].settings.mol_othercollision_active = True
    emitter_left.particle_systems[0].settings.mol_collision_damp = 0.5
    emitter_left.particle_systems[0].settings.mol_friction = 0.5     
    emitter_left.show_instancer_for_viewport = False


    bpy.ops.mesh.primitive_cube_add(size=SCENE_SIZE, scale=(0.1, 0.8, 0.1), location=(SCENE_SIZE * 0.5 * 0.8, 0, SCENE_SIZE * 0.8))
    emitter_right = bpy.context.selected_objects[0]
    bpy.ops.object.particle_system_add()
    emitter_right.particle_systems[0].name = "RightEmitter"
    emitter_right.particle_systems[0].settings.name = "RightEmitterSettings"
    emitter_right.particle_systems[0].settings.frame_start = MOL_FRAME_START + MOL_FRAME_OFFSET
    emitter_right.particle_systems[0].settings.frame_end = MOL_FRAME_END + MOL_FRAME_OFFSET
    emitter_right.particle_systems[0].settings.count = num_fine // 2
    emitter_right.particle_systems[0].settings.lifetime = 10000
    emitter_right.particle_systems[0].settings.emit_from = "VOLUME"
    emitter_right.particle_systems[0].settings.object_align_factor[0] = -5
    emitter_right.particle_systems[0].settings.factor_random = .9
    emitter_right.particle_systems[0].settings.use_multiply_size_mass = True
    emitter_right.particle_systems[0].settings.use_size_deflect = True 
    emitter_right.particle_systems[0].settings.render_type = "OBJECT"
    emitter_right.show_instancer_for_render = False
    emitter_right.particle_systems[0].settings.particle_size = expected_fine_size * GLOBAL_SCALE 
    emitter_right.particle_systems[0].settings.size_random = 0.95
    emitter_right.particle_systems[0].settings.instance_object = fine_grain_ball
    emitter_right.particle_systems[0].settings.use_scale_instance = False
    emitter_right.particle_systems[0].settings.use_rotations = True 
    emitter_right.particle_systems[0].settings.rotation_mode = "OB_X"
    emitter_right.particle_systems[0].settings.phase_factor = 0.5
    emitter_right.particle_systems[0].settings.phase_factor_random = 2
    emitter_right.particle_systems[0].settings.mol_active = True
    emitter_right.particle_systems[0].settings.mol_density_active = True
    emitter_left.particle_systems[0].settings.mol_matter = "-1"
    emitter_left.particle_systems[0].settings.mol_density = 2500
    emitter_right.particle_systems[0].settings.mol_selfcollision_active = True
    emitter_right.particle_systems[0].settings.mol_othercollision_active = True
    emitter_right.particle_systems[0].settings.mol_collision_damp = 0.5
    emitter_right.particle_systems[0].settings.mol_friction = 0.5     
    emitter_right.show_instancer_for_viewport = False


    scene.register("emitter-left", emitter_left)
    scene.register("emitter-right", emitter_right)

    status = bpy.ops.object.mol_simulate()

    def threaded_function(scene):
        import time
        import bpy
        
        from scripts import common

        MOL_MONITORING_SEC = 15

        while scene.ref.mol_simrun:
            print("mol simulation is running...")

            time.sleep(MOL_MONITORING_SEC)

        print("mol simulation done.")

        tmp_path = f"{common.ROOT_DIR}/tmp/molecular"

        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)

        bpy.ops.wm.save_userpref()
        bpy.ops.wm.save_mainfile(filepath=f"{tmp_path}/tmp.blend")

        print(scene.object_dict)

        json.dump(scene.object_dict, open(f"{tmp_path}/scene_obj.json", 'w+'))
        json.dump(scene.meta, open(f"{tmp_path}/scene_meta.json", 'w+'))

        print(f"tmp file saved at {tmp_path}. close the blender window and run the next step...")
    
    thread = Thread(target = threaded_function, args = (scene, ), daemon=False)
    thread.start()
