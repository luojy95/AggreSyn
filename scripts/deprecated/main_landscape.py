import sys 
sys.path.append(".")

import os
import glob
import math
import json
import tqdm
import numpy as np

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
SCENE_SIZE = 0.5 * GLOBAL_SCALE
BALLAST_ZONE_SIZE = SCENE_SIZE / 4
BALLAST_ZONE_HEIGHT =  SCENE_SIZE / 2

NUM_BALLAST = 500

MAX_FRAME_NUM = 180

BALLAST_FALLING_FRAME_PHASE_I = 200
BALLAST_FALLING_FRAME_PHASE_II = 30

FINE_NUM_MAX_HAIR = 20000

IMAGE_WIDTH = 5000
IMAGE_HEIGHT = 4000

CYCLES_DEVICE = 'GPU'
CYCLES_SAMPLES_LABEL = 1
CYCLES_SAMPLES_IMAGE = 4096

GENERATE_COMPLETE_LABEL = False

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
    bpy.ops.object.camera_add(location=(0, 0, 0.8 * GLOBAL_SCALE))
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
    ballast_meshes = glob.glob(common.ROCK_MESH_PATH)
    num_meshes = len(ballast_meshes)
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
        U.focus(high_poly_ballast)
        bpy.ops.rigidbody.object_remove()

        
        high_poly_ballasts.append(high_poly_ballast)

        # convert to low resolution meshes
        low_poly_ballast = U.copy_obj(high_poly_ballast)
        U.focus(low_poly_ballast)
        bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME')
        low_poly_ballast.scale = LOADING_MESH_DEFAULT_SCALE
        low_poly_ballast.location = (LOADING_MESH_START_X + i, 0, 1)
        scene.register(f"low_poly_ballast_{i}", low_poly_ballast)

        # decrease number of polygons and clean up the mesh
        U.set_active(low_poly_ballast)
        U.simplify_mesh(low_poly_ballast, keep_face_ratio = LOW_POLY_MESH_KEEP_FACES / len(low_poly_ballast.data.polygons))
        U.clean_mesh(low_poly_ballast)

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

        d = scene.meta[f"high_poly_ballast_{ballast_type}"]['d']        
        
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

    # Kepler's Sphere Packing (density of the symmetrical packing is 0.52)
    height_expected = max(0.01 * SCENE_SIZE, test_gradation.get_fine_vol(inch=False) / 0.52 / (SCENE_SIZE/GLOBAL_SCALE)**2 * 2)
    print(height_expected)
    # build top landscape
    bpy.ops.mesh.primitive_plane_add(size=SCENE_SIZE)
    base = bpy.context.selected_objects[0]
    base.location = (0, 0, -0.1 * SCENE_SIZE)

    bpy.ops.mesh.landscape_add(
        mesh_size_x=SCENE_SIZE, 
        mesh_size_y=SCENE_SIZE, 
        refresh=True, 
        height = height_expected,
        subdivision_x=128,
        subdivision_y=128,
        noise_type="ant_turbulence",
        noise_size=2
        # noise_type="rocks_noise"
    )
    top = bpy.context.selected_objects[0]
    top.location = (0, 0, 0.1)

    top = U.build_mesh_from_two_square_planes(top=top, bottom=base, size=SCENE_SIZE)


    # TODO: get labels for each ballast
    scene.ref.use_nodes = True
    tree = bpy.context.scene.node_tree

    # clear default nodes
    for node in tree.nodes:
        tree.nodes.remove(node)

    # create input image node
    render_node = tree.nodes.new(type='CompositorNodeRLayers')
    render_node.layer = "ViewLayer"
    render_node.location = 0,0

    # create output node
    comp_node = tree.nodes.new(type='CompositorNodeComposite')   
    comp_node.location = 400,0

    for (i, ballast) in enumerate(ballasts):
        ballast.pass_index = i + 1
    
    scene.ref.view_layers["ViewLayer"].use_pass_object_index = True
    
    scene.ref.use_nodes = True
    
    tree = bpy.context.scene.node_tree
    links = tree.links

    # clear default nodes
    for node in tree.nodes:
        tree.nodes.remove(node)
    
    # create input image node
    render_node = tree.nodes.new(type='CompositorNodeRLayers')
    render_node.layer = "ViewLayer"
    render_node.location = 0,0

    # create output node
    comp_node = tree.nodes.new(type='CompositorNodeComposite')   
    comp_node.location = 400,0

    link = links.new(render_node.outputs[0], comp_node.inputs[0])

    file_out_node = tree.nodes.new(type='CompositorNodeOutputFile')
    file_out_node.base_path = f"{common.ROOT_DIR}/outputs/label"
    file_out_node.location = 500,100
    file_out_node.layer_slots.clear()

    if not os.path.exists(file_out_node.base_path):
        os.makedirs(file_out_node.base_path)

    tmp_scenes = []
    # for i in tqdm.trange(num_ballasts):

    #     # TODO: render complete individual projections
    #     file_out_node.layer_slots.new(f"complete_ballast_individual_{i:04d}")
    #     if GENERATE_COMPLETE_LABEL:
    #         id_mask_node = tree.nodes.new(type='CompositorNodeIDMask')
    #         id_mask_node.index = i + 1
    #         bpy.ops.scene.new(type="EMPTY")
    #         scene_i = bpy.context.scene
    #         scene_i.collection.objects.link(camera)
    #         scene_i.camera = camera
    #         scene_i.view_layers["ViewLayer"].use_pass_object_index = True
    #         scene_i.collection.objects.link(light)
    #         scene_i.collection.objects.link(ballasts[i])
    #         scene_i.cycles.samples = 1
    #         render_node_i = tree.nodes.new(type='CompositorNodeRLayers')
    #         render_node_i.layer = "ViewLayer"
    #         render_node_i.scene = scene_i

    #         tmp_scenes.append(scene_i)

    #         link = links.new(render_node_i.outputs[2], id_mask_node.inputs[0])
    #         link = links.new(id_mask_node.outputs[0], file_out_node.inputs[2 * i])

    #     # render visible individual projections
    #     id_mask_node_1 = tree.nodes.new(type='CompositorNodeIDMask')
    #     id_mask_node_1.index = i + 1
    #     file_out_node.layer_slots.new(f"visible_ballast_individual_{i:04d}")
    #     link = links.new(render_node.outputs[2], id_mask_node_1.inputs[0])
    #     link = links.new(id_mask_node_1.outputs[0], file_out_node.inputs[2 * i + 1])

    file_out_node.layer_slots.new("visible_ballast_all")

    combine_node = tree.nodes.new(type="CompositorNodeCombRGBA")

    divide_1_node = tree.nodes.new(type='CompositorNodeMath')
    divide_1_node.operation = 'DIVIDE'
    divide_1_node.inputs[1].default_value = 256

    divide_2_node = tree.nodes.new(type='CompositorNodeMath')
    divide_2_node.operation = 'DIVIDE'
    divide_2_node.inputs[1].default_value = 256

    divide_3_node = tree.nodes.new(type='CompositorNodeMath')
    divide_3_node.operation = 'DIVIDE'
    divide_3_node.inputs[1].default_value = 256

    mod_1_node = tree.nodes.new(type='CompositorNodeMath')
    mod_1_node.operation = 'MODULO'
    mod_1_node.inputs[1].default_value = 256

    floor_1_node = tree.nodes.new(type='CompositorNodeMath')
    floor_1_node.operation = 'FLOOR'

    link = links.new(render_node.outputs[2], divide_1_node.inputs[0])
    link = links.new(divide_1_node.outputs[0], floor_1_node.inputs[0])
    link = links.new(floor_1_node.outputs[0], divide_2_node.inputs[0])
    
    link = links.new(render_node.outputs[2], mod_1_node.inputs[0])
    link = links.new(mod_1_node.outputs[0], divide_3_node.inputs[0])
        
    link = links.new(divide_2_node.outputs[0], combine_node.inputs[0])
    link = links.new(divide_3_node.outputs[0], combine_node.inputs[1])

    link = links.new(combine_node.outputs[0], file_out_node.inputs[-1])

    bpy.context.window.scene = scene.ref
    scene.ref.camera = camera
    bpy.ops.render.render(animation=False, scene=scene.ref.name, write_still=True)

    # clear default nodes
    for node in tree.nodes:
        tree.nodes.remove(node)
    for tmp_scene in tmp_scenes:
        bpy.context.window.scene = tmp_scene
        bpy.ops.scene.delete()

    bpy.context.window.scene = scene.ref
    scene.ref.use_nodes = False

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

    U.unselect_all()
    U.set_active(top)
    # TODO: solve problem with union the base landscape
    # # get union volume with the base_landscape
    # bpy.ops.object.modifier_add(type='BOOLEAN')
    # top.modifiers["Boolean"].operation = "UNION"
    # top.modifiers["Boolean"].object = base_landscape
    # bpy.ops.object.modifier_apply(modifier=top.modifiers["Boolean"].name)
    
    # # get union volume with the base_landscape
    # bpy.ops.object.modifier_add(type='BOOLEAN')
    # top.modifiers["Boolean"].operation = "DIFFERENCE"
    # top.modifiers["Boolean"].object = ballast_group
    # bpy.ops.object.modifier_apply(modifier=top.modifiers["Boolean"].name)

    M.assign_material(top, ground_material)

    # TODO: add hair particle system to top landscape with rock generator
    # TODO: make hair particles more and smaller
    
    U.unselect_all()
    U.set_active(None)

    bpy.ops.mesh.add_mesh_rock(preset_values='1', 
        num_of_rocks=20, scale_X=(0.5, 1.25), skew_X=-0.5, scale_Y=(0.5, 1.25), skew_Y=-0.5, scale_Z=(0.5, 1.25), skew_Z=-0.5, 
        use_scale_dis=False, scale_fac=(1, 1, 1), deform=1, rough=2, detail=3, display_detail=3, smooth_fac=2, smooth_it=2, use_generate=True, use_random_seed=True)
    
    bpy.ops.collection.create(name = "Fine_Gen")
    bpy.context.scene.collection.children.link(bpy.data.collections["Fine_Gen"])
    fine_collection = bpy.data.collections["Fine_Gen"]

    for obj in bpy.context.selected_objects:
        obj.location = (2*SCENE_SIZE, 2*SCENE_SIZE, 1)
        M.assign_material(obj, np.random.choice(ballast_materials))
    
    U.unselect_all()
    U.set_active(top)
    bpy.ops.object.particle_system_add()

    fine_num = min(FINE_NUM_MAX_HAIR, round((SCENE_SIZE/GLOBAL_SCALE/MU.inch2meter(test_gradation.expected_fine_size)) ** 2))
    top.particle_systems[0].settings.type = "HAIR"
    top.particle_systems[0].settings.count = fine_num
    top.particle_systems[0].settings.use_advanced_hair = True
    top.particle_systems[0].settings.use_rotations = True
    top.particle_systems[0].settings.phase_factor = 1.0
    top.particle_systems[0].settings.phase_factor_random = 2.0
    top.particle_systems[0].settings.render_type = "COLLECTION"
    top.particle_systems[0].settings.instance_collection = fine_collection
    top.particle_systems[0].settings.use_collection_pick_random = True
    top.particle_systems[0].settings.particle_size = MU.inch2meter(scene.meta["expected_fine_size"]) * GLOBAL_SCALE / 2
    top.particle_systems[0].settings.use_scale_instance = False
    top.particle_systems[0].settings.size_random = 1.0
    top.particle_systems[0].settings.distribution = "RAND"

    # TODO: re-calculate gradation
    # U.set_active(top)
    # bpy.ops.rigidbody.object_add(type='PASSIVE')
    # bpy.ops.rigidbody.shape_change(type='MESH')  
    # volume = (G.volume_calculate(top) - (0.1*SCENE_SIZE)*SCENE_SIZE*SCENE_SIZE) / GLOBAL_SCALE**3 * 0.52
    
    # print(volume, test_gradation.get_fine_vol(inch=False))

    U.delete_obj(base_landscape)
    U.delete_obj(base_box)

    out_path = f"{common.ROOT_DIR}/outputs/landscape"

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    scene.ref.camera = bpy.data.objects[scene.object_dict['camera']]
    scene.ref.render.filepath = os.path.join(f"{out_path}", "landscape_results.jpg")
    bpy.ops.render.render(animation=False, scene=scene.ref.name, write_still=True)

    bpy.ops.wm.save_userpref()
    bpy.ops.wm.save_mainfile(filepath=f"{out_path}/landscape.blend")

    json.dump(scene.object_dict, open(f"{out_path}/scene_obj.json", 'w+'))
    json.dump(scene.meta, open(f"{out_path}/scene_meta.json", 'w+'))

    print(f"output file saved at {out_path}. close the blender window ...")