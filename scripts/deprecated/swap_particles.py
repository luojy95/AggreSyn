import sys 
sys.path.append(".")
import os

import numpy as np

import bpy

from scripts import common
from scripts import math_utils as MU
from scripts.scene import Scene
from scripts.gradation_helper import GradationHelper as G
from scripts.material_helper import MaterialHelper as M
from scripts.blender_utils import BlenderUtils as U


OUTPUT_FN = "molecular.blend"

np.random.seed(0)

if __name__=='__main__':

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

    # select default scene
    scene = Scene(ref=bpy.context.scene)
    scene.restore(obj_path=f"{common.TMP_DIR}/molecular/scene_obj.json", meta_path=f"{common.TMP_DIR}/molecular/scene_meta.json")
    
    GLOBAL_SCALE = scene.meta["global_scale"]
    SCENE_SIZE = scene.meta["size"]
    MAX_FRAME_NUM = scene.meta["max_frames"]

    scene.ref.frame_set(MAX_FRAME_NUM-1)


    U.unselect_all()
    U.set_active(None)

    bpy.ops.collection.create(name  = "Fine_Gen")
    bpy.context.scene.collection.children.link(bpy.data.collections["Fine_Gen"])
    fine_collection = bpy.data.collections["Fine_Gen"]

    bpy.ops.mesh.add_mesh_rock(preset_values='1', 
        num_of_rocks=20, scale_X=(0.5, 1.25), skew_X=-0.5, scale_Y=(0.5, 1.25), skew_Y=-0.5, scale_Z=(0.5, 1.25), skew_Z=-0.5, 
        use_scale_dis=False, scale_fac=(1, 1, 1), deform=1, rough=2, detail=3, display_detail=3, smooth_fac=2, smooth_it=2, use_generate=True, use_random_seed=True)
    

    fines = []
    for obj in bpy.context.selected_objects:
        obj.location = (SCENE_SIZE*2, SCENE_SIZE*2, SCENE_SIZE*2)
        fine_collection.objects.link(obj)
        bpy.data.collections["Collection"].objects.unlink(obj)
        fines.append(obj)
    
    for obj in fines:
        U.unselect_all()
        U.set_active(None)
        obj.select_set(True)
        U.set_active(obj)

        M.assign_material(obj, np.random.choice(ballast_materials))
        bpy.ops.rigidbody.object_add(type="ACTIVE")
        d = G.diameter_calculate(obj, inch=False)
        scale = 2 / d
        obj.scale = (scale, scale, scale)
        U.set_active(obj)
        bpy.ops.rigidbody.object_remove() 

    emitters = []
    for key in scene.object_dict:
        if "emit" in key:
            emitters.append((key, bpy.data.objects[scene.object_dict[key]]))
    
    for (emitter_name, emitter_obj) in emitters:
        emitter_obj.show_instancer_for_viewport = False
        emitter_obj.particle_systems[0].settings.render_type = "COLLECTION"
        emitter_obj.particle_systems[0].settings.instance_collection = fine_collection
        emitter_obj.particle_systems[0].settings.use_collection_pick_random = True
        emitter_obj.particle_systems[0].settings.particle_size = MU.inch2meter(scene.meta["expected_fine_size"]) * fines[0].scale[0] * GLOBAL_SCALE
        

    print("mesh replacement done.")

    # TODO: re-calculate gradation

    #  'BLENDER_EEVEE', 'BLENDER_WORKBENCH', 'CYCLES'
    scene.ref.render.engine = 'BLENDER_EEVEE'
    # scene.ref.render.engine = 'CYCLES'
    scene.ref.render.resolution_x = 4032
    scene.ref.render.resolution_y = 3024
    scene.ref.camera = bpy.data.objects[scene.object_dict['camera']]
    scene.ref.render.filepath = os.path.join(f"{common.OUTPUT_DIR}", "mol_results.jpg")
    bpy.ops.render.render(animation=False, scene=scene.ref.name, write_still=True)

    bpy.ops.wm.save_userpref()
    bpy.ops.wm.save_mainfile(filepath=f"{common.OUTPUT_DIR}/{OUTPUT_FN}")