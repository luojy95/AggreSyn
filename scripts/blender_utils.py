import os
import glob
import sys 
sys.path.append(".")

import numpy as np

import bpy
import bmesh

class BlenderUtils:

    @classmethod
    def unselect_all(cls) -> None:
        """Unselect"""
        # for obj in bpy.data.objects:
        for obj in bpy.context.selected_objects:
            if obj.select_get():
                obj.select_set(False)

    @classmethod
    def single_select(cls, obj) -> bool:
        cls.unselect_all()
        obj.select_set(True)
        return obj.select_get()


    @classmethod
    def focus(cls, obj):
        cls.set_active(None)
        cls.single_select(obj)
        cls.set_active(obj)

    @classmethod
    def import_obj(cls, path):
        postfix = os.path.split(path)[-1].split(".")[-1]
        if postfix == "obj":
            import_func = bpy.ops.import_scene.obj
        elif postfix == "fbx":
            import_func = bpy.ops.import_scene.fbx
        else:
            return None

        msg = import_func(filepath=path)

        if 'FINISHED' not in msg:
            return None 

        obj_object = bpy.context.selected_objects[0]

        print(f"Obj object {obj_object.name} is imported to the current scene")
        return obj_object

    @classmethod
    def copy_obj(cls, src_obj):
        obj_copy = src_obj.copy()
        obj_copy.data = obj_copy.data.copy()
        bpy.context.collection.objects.link(obj_copy)
        return obj_copy

    @classmethod
    def delete_obj(cls, target_obj):
        cls.unselect_all()
        target_obj.select_set(True)
        cls.set_active(target_obj)    
        bpy.ops.object.delete()

    @classmethod
    def set_active(cls, obj = None):
        bpy.context.view_layer.objects.active = obj

    @classmethod
    def reset_all(cls):
        #  step 1 : delete all objects in scene
        for scene in bpy.data.scenes:
            for obj in scene.objects:
                bpy.context.collection.objects.unlink(obj)

        # step 2 : delete all cached data in the dataset
        for bpy_data_iter in (
                bpy.data.objects,
                bpy.data.meshes,
                bpy.data.cameras,
                bpy.data.materials
        ):
            for id_data in bpy_data_iter:
                bpy_data_iter.remove(id_data)

    @classmethod
    def add_environment_texture(cls, path):
        # Get the environment node tree of the current scene
        node_tree = bpy.context.scene.world.node_tree
        tree_nodes = node_tree.nodes

        # Clear all nodes
        tree_nodes.clear()

        text_coord_node = tree_nodes.new(type="ShaderNodeTexCoord")
        generated_coord = text_coord_node.outputs[0]

        mapping = tree_nodes.new("ShaderNodeMapping")
        mapping.inputs[2].default_value[2] = np.deg2rad(np.random.uniform(0, 360))

        # Add Background node
        node_background = tree_nodes.new(type='ShaderNodeBackground')

        # Add Environment Texture node
        node_environment = tree_nodes.new('ShaderNodeTexEnvironment')
        # Load and assign the image to the node property
        node_environment.image = bpy.data.images.load(path) # Relative path
        node_background.inputs[1].default_value = np.random.uniform(3, 5) # initial env brightness

        # Add Output node
        node_output = tree_nodes.new(type='ShaderNodeOutputWorld')   
        node_output.location = 200,0

        # Link all nodes
        links = node_tree.links
        links.new(generated_coord, mapping.inputs[0])
        links.new(mapping.outputs[0], node_environment.inputs[-1])
        link = links.new(node_environment.outputs["Color"], node_background.inputs["Color"])
        link = links.new(node_background.outputs["Background"], node_output.inputs["Surface"])
        
        return node_environment, mapping
    
    @classmethod
    def build_mesh_from_two_square_planes(cls, top, bottom, size):
        cls.unselect_all()
        cls.set_active(bottom)
        bottom.select_set(True)
        
        cls.set_active(top)
        top.select_set(True)
        bottom.select_set(True)
        bpy.ops.object.join()
        top = bpy.context.selected_objects[0]  

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_mode(type='EDGE')

        bm = bmesh.from_edit_mesh(top.data)  # Create bmesh object for easy mesh evaluation
        epsilon = 1e-5 

        for e in bm.edges:  # Check all edges
            start_pos = e.verts[0].co  # Get first vert position of this edge
            end_pos = e.verts[1].co  # Get second vert position of this edge

            # Select or deselect depending of the relative position of both vertices
            start_max_abs_coords = np.abs([start_pos.x, start_pos.y]).max()
            end_max_abs_coords = np.abs([end_pos.x, end_pos.y]).max()
            e.select_set(abs(start_max_abs_coords - size / 2) <= epsilon and abs(end_max_abs_coords - size / 2) <= epsilon )
            
        bmesh.update_edit_mesh(top.data)  # Update the mesh in edit mode
        bpy.ops.mesh.bridge_edge_loops()
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode="OBJECT")

        return top
    
    @classmethod
    def get_highest_vertice_z(cls, obj):
        cls.focus(obj)

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_mode(type='EDGE')

        bm = bmesh.from_edit_mesh(obj.data)  # Create bmesh object for easy mesh evaluation
        max_z = -float("inf")
        for v in bm.verts:
            max_z = max(max_z, v.co.z)
        bpy.ops.object.mode_set(mode="OBJECT")

        return max_z

    @classmethod
    def camera_look_at(cls, obj_camera, point):
        loc_camera = obj_camera.matrix_world.to_translation()

        direction = point - loc_camera
        # point the cameras '-Z' and use its 'Y' as up
        rot_quat = direction.to_track_quat('-Z', 'Y')

        # assume we're using euler rotation
        obj_camera.rotation_euler = rot_quat.to_euler()
    
    @classmethod
    def clean_mesh(cls, obj):
        cls.focus(obj)
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
    
    @classmethod
    def simplify_mesh(cls, obj, keep_face_ratio = 0.5):
        cls.focus(obj)
        bpy.ops.object.modifier_add(type='DECIMATE')
        obj.modifiers['Decimate'].ratio = keep_face_ratio 
        bpy.ops.object.modifier_apply(modifier=obj.modifiers['Decimate'].name)
        bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME')
    
    @classmethod
    def subdivision(cls, obj, level=5, adaptive=False, dicing=1.0):
        cls.focus(obj)
        obj.modifiers.new(f"{obj.name}_subsurf", type="SUBSURF")
        obj.modifiers[f"{obj.name}_subsurf"].subdivision_type = "SIMPLE"

        if adaptive:
            obj.cycles.use_adaptive_subdivision = True
            obj.cycles.dicing_rate = dicing
            obj.modifiers[f"{obj.name}_subsurf"].levels = level - 1

        else:
            obj.modifiers[f"{obj.name}_subsurf"].levels = level - 1
            obj.modifiers[f"{obj.name}_subsurf"].render_levels = level
        
        # bpy.ops.object.modifier_apply(modifier=f"{obj.name}_subsurf")
    
    @classmethod
    def traverse_path(cls, path, postfix_list):
        files = []

        for file in glob.glob(path):
            if os.path.isdir(file):
                files.extend(cls.traverse_path(os.path.join(file, '*'), postfix_list))
            
            else:
                postfix = os.path.split(file)[-1].split(".")[-1]
                if postfix in postfix_list:
                    files.append(file)

        return files
        
    @ classmethod
    def append_blend_file(cls, blendfile, section, objects):
        # ----------------- EXAMPLE -------------
        # load materials from materials.blend
        # blendfile = os.path.join(C.ROOT_DIR, "materials.blend")
        # section  = "/Material/"
        # objects  = {
        #     "Ballast":[
        #         "Ballast-01", "Ballast-02", "Ballast-03", "Ballast-04", "Ballast-05", "Ballast-06", "Ballast-07",
        #         "Ballast-08", "Ballast-09", "Ballast-10", "Ballast-11", "Ballast-12"
        #     ], 
        #     "Ground": [
        #         # "Ground-01", "Ground-02", "Ground-03", "Ground-04", 
        #         "Ground-05", #"Ground-01-Box"
        #     ],
        #     "Others": ["Transparent"], 
        # }

        for key in objects:
            for object in objects[key]:

                filepath  = blendfile + section + object
                directory = blendfile + section
                filename  = object
                bpy.ops.wm.append(
                    filepath=filepath, 
                    filename=filename,
                    directory=directory
                )