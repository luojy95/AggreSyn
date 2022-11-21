import sys
sys.path.append(".")

import glob
import os
from functools import reduce

import numpy as np

import bpy
from bpy.types import Object, Material

from scripts.blender_utils import BlenderUtils as U

class MaterialHelper:
    
    @classmethod
    def create_material(cls, name: str) -> Material:
        mtl = bpy.data.materials.new(name=name)
        return mtl
    
    @classmethod
    def assign_material(cls, obj: Object, material: Material):
        obj.data.materials.clear()
        obj.data.materials.append(material)

    @classmethod
    def add_displacement(cls, obj: Object, disp_map_path: str, strength: float=1.0, subdivision_level=5):
        texture = bpy.data.textures.new(f"{obj.name}_disp_texture", 'IMAGE')
        texture.image = bpy.data.images.load(disp_map_path)
        
        U.subdivision(obj, level=subdivision_level)

        obj.modifiers.new(f"{obj.name}_disp", type="DISPLACE")
        obj.modifiers[f"{obj.name}_disp"].texture = texture
        obj.modifiers[f"{obj.name}_disp"].texture_coords = "UV"
        obj.modifiers[f"{obj.name}_disp"].strength = strength

        U.focus(obj)
        # bpy.ops.object.modifier_apply(modifier=obj.modifiers[f"{obj.name}_disp"].name)
    
    @classmethod
    def scale_UV(cls, object, scale = (2, 2), pivot=None, uvMap = None):
        uvMap = object.data.uv_layers[0]

        if pivot is None:
            pivot = ((scale[0] - 1) / 2, (scale[1] - 1) / 2)
        #Scale a 2D vector v, considering a scale s and a pivot point p
        def Scale2D( v, s, p ):
            return ( p[0] + s[0]*(v[0] - p[0]), p[1] + s[1]*(v[1] - p[1]) )     

        #Scale a UV map iterating over its coordinates to a given scale and with a pivot point
        for uvIndex in range( len(uvMap.data) ):
            uvMap.data[uvIndex].uv = Scale2D( uvMap.data[uvIndex].uv, scale, pivot )

    @classmethod
    def rotate_UV(cls, object, angle, anchor=(0.5, 0.5)):
        uvMap = object.data.uv_layers[0]

        def make_rotation_transformation(angle, origin=(0, 0)):
            cos_theta, sin_theta = np.cos(angle), np.sin(angle)
            x0, y0 = origin    
            def xform(point):
                x, y = point[0] - x0, point[1] - y0
                return (x * cos_theta - y * sin_theta + x0,
                        x * sin_theta + y * cos_theta + y0)
            return xform


        rad = np.deg2rad(angle)
        rot = make_rotation_transformation(rad, anchor)
        for uvIndex in range( len(uvMap.data) ):
            uvMap.data[uvIndex].uv = rot( uvMap.data[uvIndex].uv)

    @classmethod
    def transparent_material(cls, name="transparent"):
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True 
        nodes = mat.node_tree.nodes
        bsdf = nodes["Principled BSDF"]
        bsdf.inputs[21].default_value = 0
        mat.blend_method = "CLIP"
        mat.shadow_method = "CLIP"

        return mat

    @classmethod
    def load_material(cls, name, path, is_ballast = False, displacement_strength = 0.1):
        
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True 
        nodes = mat.node_tree.nodes 

        bsdf = nodes["Principled BSDF"]
        bsdf.inputs[7].default_value = 0.1
        mat_out = nodes["Material Output"]

        tex_coord = nodes.new("ShaderNodeTexCoord")
        generated_coord = tex_coord.outputs[0]
        uv_coord = tex_coord.outputs[2]

        mapping = nodes.new("ShaderNodeMapping")

        if is_ballast:
            mat.node_tree.links.new(generated_coord, mapping.inputs[0])
        else:
            mat.node_tree.links.new(uv_coord, mapping.inputs[0])

        img_extensions = ['jpg', 'png', 'jpeg']
        img_paths = reduce(
            lambda a, b: a+b,
            [glob.glob(os.path.join(path, f"*.{ext}")) for ext in img_extensions])

        def load_texture(name, img_path):
            tex = nodes.new("ShaderNodeTexImage")
            tex.name = name
            img = bpy.data.images.load(img_path)
            tex.image = img 

            if is_ballast:
                tex.projection = "BOX"
            else:
                tex.projection = "FLAT"
            
            mat.node_tree.links.new(mapping.outputs[0], tex.inputs[0])

            return tex
                    
        for img_path in img_paths:
            name_lower = os.path.split(img_path)[-1].lower()
            # color map
            if "color" in name_lower or "albedo" in name_lower:
                tex = load_texture(name="tex_color", img_path=img_path)
                mat.node_tree.links.new(tex.outputs[0], bsdf.inputs[0])
            
            elif "rough" in name_lower:
                tex = load_texture(name="tex_rough", img_path=img_path)
                mat.node_tree.links.new(tex.outputs[0], bsdf.inputs[9])
            
            elif "normal" in name_lower and "normal" not in nodes:
                tex = load_texture(name="tex_normal", img_path=img_path)
                normal_map = nodes.new("ShaderNodeNormalMap")
                normal_map.name = "normal_map"
                mat.node_tree.links.new(tex.outputs[0], normal_map.inputs[1])
                mat.node_tree.links.new(normal_map.outputs[0], bsdf.inputs[22])
            
            elif "disp" in name_lower:
                tex = load_texture(name="tex_displacement", img_path=img_path)
                displacement = nodes.new("ShaderNodeDisplacement")
                displacement.name = "displacement"
                displacement.inputs[2].default_value = displacement_strength
                mat.node_tree.links.new(tex.outputs[0], displacement.inputs[0])
                mat.node_tree.links.new(displacement.outputs[0], mat_out.inputs[2])
        
        return mat

    @classmethod
    def load_mixed_ballast_mat(cls, name, ballast_mat_path, fine_mat_path, z_thresh_width=0.3, z_thresh_center=0):
        
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True 
        nodes = mat.node_tree.nodes 

        ballast_bsdf = nodes["Principled BSDF"]
        ballast_bsdf.inputs[7].default_value = 0.1

        fine_bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        fine_bsdf.inputs[7].default_value = 0.1

        mat_out = nodes["Material Output"]

        tex_coord = nodes.new("ShaderNodeTexCoord")
        generated_coord = tex_coord.outputs[0]
        uv_coord = tex_coord.outputs[2]

        mapping = nodes.new("ShaderNodeMapping")

        mat.node_tree.links.new(uv_coord, mapping.inputs[0])

        def load_texture_box(name, img_path):
            tex = nodes.new("ShaderNodeTexImage")
            tex.name = name
            img = bpy.data.images.load(img_path)
            tex.image = img 

            tex.projection = "FLAT"
            
            mat.node_tree.links.new(mapping.outputs[0], tex.inputs[0])

            return tex
                    
        # Step 1: load ballast box material
        for img_path in get_image_paths(ballast_mat_path):
            name_lower = os.path.split(img_path)[-1].lower()
            # color map
            if "color" in name_lower or "albedo" in name_lower:
                tex = load_texture_box(name="ballast_tex_color", img_path=img_path)
                mat.node_tree.links.new(tex.outputs[0], ballast_bsdf.inputs[0])
            
            elif "rough" in name_lower:
                tex = load_texture_box(name="ballast_tex_rough", img_path=img_path)
                mat.node_tree.links.new(tex.outputs[0], ballast_bsdf.inputs[9])
            
            elif "normal" in name_lower and "normal" not in nodes:
                tex = load_texture_box(name="ballast_tex_normal", img_path=img_path)
                normal_map = nodes.new("ShaderNodeNormalMap")
                normal_map.name = "ballast_normal_map"
                mat.node_tree.links.new(tex.outputs[0], normal_map.inputs[1])
                mat.node_tree.links.new(normal_map.outputs[0], ballast_bsdf.inputs[22])
            
            elif "disp" in name_lower:
                tex = load_texture_box(name="ballast_tex_displacement", img_path=img_path)
        
        # Step 2: load fine box material
        for img_path in get_image_paths(fine_mat_path):
            name_lower = os.path.split(img_path)[-1].lower()
            # color map
            if "color" in name_lower or "albedo" in name_lower:
                tex = load_texture_box(name="fine_tex_color", img_path=img_path)
                mat.node_tree.links.new(tex.outputs[0], fine_bsdf.inputs[0])
            
            elif "rough" in name_lower:
                tex = load_texture_box(name="fine_tex_rough", img_path=img_path)
                mat.node_tree.links.new(tex.outputs[0], fine_bsdf.inputs[9])
            
            elif "disp" in name_lower:
                tex = load_texture_box(name="fine_tex_displacement", img_path=img_path)

        
        mat.node_tree.links.new(nodes['ballast_normal_map'].outputs[0], fine_bsdf.inputs[22])

        # step 3: calcualte random cover
        obj_info = nodes.new("ShaderNodeObjectInfo")
        obj_rand = obj_info.outputs[-1] # random in [0, 1), unique for each object in the scene
    
        # step 3.1: consider face normal
        geo_info = nodes.new("ShaderNodeNewGeometry")
        normal_info = geo_info.outputs[1]
        sep_xyz = nodes.new("ShaderNodeSeparateXYZ")
        normal_z = sep_xyz.outputs[2]
        mat.node_tree.links.new(normal_info, sep_xyz.inputs[0])
        multi_node = nodes.new(type='ShaderNodeMath')
        multi_node.operation = 'MULTIPLY'
        multi_node.inputs[1].default_value = z_thresh_width
        mat.node_tree.links.new(obj_rand, multi_node.inputs[0])
        sub_node = nodes.new(type='ShaderNodeMath')
        sub_node.operation = 'SUBTRACT'
        sub_node.inputs[1].default_value = z_thresh_width / 2 - z_thresh_center
        mat.node_tree.links.new(multi_node.outputs[0], sub_node.inputs[0])
        z_thresh = sub_node.outputs[0]

        compare = nodes.new(type='ShaderNodeMath')
        compare.operation = 'GREATER_THAN'
        mat.node_tree.links.new(normal_z, compare.inputs[0])
        mat.node_tree.links.new(z_thresh, compare.inputs[1])
        mask = compare.outputs[0]

        apply_mask = nodes.new(type='ShaderNodeMath')
        apply_mask.operation = 'MULTIPLY'
        mat.node_tree.links.new(normal_z, apply_mask.inputs[0])
        mat.node_tree.links.new(mask, apply_mask.inputs[1])
        masked_z = apply_mask.outputs[0]

        z_clr_ramp = nodes.new(type="ShaderNodeValToRGB")

        mat.node_tree.links.new(masked_z, z_clr_ramp.inputs[0])
        # Removing the First Element this is not necessary but done to show how to remove color stops
        z_clr_ramp.color_ramp.elements.remove(z_clr_ramp.color_ramp.elements[0])

        # Adding new color stop at location 0.100
        z_clr_ramp.color_ramp.elements.new(0.100)

        # Setting the color for the stop that we recently created
        z_clr_ramp.color_ramp.elements[0].color = (0,0,0,1)

        #creating the second stop the same way
        z_clr_ramp.color_ramp.elements.new(0.600)
        z_clr_ramp.color_ramp.elements[1].color = (.5,.5,.5,1)
        z_clr = z_clr_ramp.outputs[0]

        # step 3.2: add noise texture

        noise = nodes.new(type="ShaderNodeTexNoise")
        noise.inputs[2].default_value = 2.0
        noise.inputs[3].default_value = np.random.uniform(7, 15)
        noise.inputs[4].default_value = np.random.uniform(0.85, 0.95)
        noise.inputs[5].default_value = 0

        mul_node = nodes.new(type='ShaderNodeMath')
        mul_node.operation = 'MULTIPLY'
        mul_node.inputs[1].default_value = 2
        mat.node_tree.links.new(noise.outputs[0], mul_node.inputs[0])
        
        sub_node = nodes.new(type='ShaderNodeMath')
        sub_node.operation = 'SUBTRACT'
        sub_node.inputs[1].default_value = 1
        mat.node_tree.links.new(mul_node.outputs[0], sub_node.inputs[0])
        
        scaled_noise = sub_node.outputs[0]

        mul_node = nodes.new(type='ShaderNodeMath')
        mul_node.operation = 'MULTIPLY'
        mat.node_tree.links.new(scaled_noise, mul_node.inputs[0])
        mat.node_tree.links.new(mask, mul_node.inputs[1])

        noise_clr_ramp = nodes.new(type="ShaderNodeValToRGB")

        mat.node_tree.links.new(mul_node.outputs[0], noise_clr_ramp.inputs[0])
        # Removing the First Element this is not necessary but done to show how to remove color stops
        noise_clr_ramp.color_ramp.elements.remove(noise_clr_ramp.color_ramp.elements[0])
        # Adding new color stop at location 0.100
        noise_clr_ramp.color_ramp.elements.new(0.000)
        # Setting the color for the stop that we recently created
        noise_clr_ramp.color_ramp.elements[0].color = (0,0,0,1)
        # creating the second stop the same way
        noise_clr_ramp.color_ramp.elements.new(0.600)
        noise_clr_ramp.color_ramp.elements[1].color = (1,1,1,1)
        noise_clr = noise_clr_ramp.outputs[0]
        

        # step 4: mix z and noise

        mix_clr = nodes.new(type="ShaderNodeMixRGB")
        mix_clr.inputs[0].default_value = 0
        mat.node_tree.links.new(noise_clr, mix_clr.inputs[1])
        mat.node_tree.links.new(z_clr, mix_clr.inputs[2])

        gamma = nodes.new(type="ShaderNodeGamma")
        gamma.inputs[1].default_value = np.random.uniform(0.01, 0.02)
        mat.node_tree.links.new(mix_clr.outputs[0], gamma.inputs[0])

        mix_factor = gamma.outputs[0]

        # step 5: mix bsdfs
        mix_shader = nodes.new(type="ShaderNodeMixShader")
        mat.node_tree.links.new(mix_factor, mix_shader.inputs[0])
        mat.node_tree.links.new(ballast_bsdf.outputs[0], mix_shader.inputs[1])
        mat.node_tree.links.new(fine_bsdf.outputs[0], mix_shader.inputs[2])
        mat.node_tree.links.new(mix_shader.outputs[0], mat_out.inputs[0])

        # step 6: mix bumps/displacements
        mix_clr = nodes.new(type="ShaderNodeMixRGB")
        mat.node_tree.links.new(mix_factor, mix_clr.inputs[0])
        mat.node_tree.links.new(nodes['ballast_tex_displacement'].outputs[0], mix_clr.inputs[1])
        mat.node_tree.links.new(nodes['fine_tex_displacement'].outputs[0], mix_clr.inputs[2])
        mixed_disp = mix_clr.outputs[0]
        
        displacement = nodes.new("ShaderNodeDisplacement")
        displacement.name = "displacement"
        mat.node_tree.links.new(mixed_disp, displacement.inputs[0])
        mat.node_tree.links.new(displacement.outputs[0], mat_out.inputs[2])

        return mat
 
def get_image_paths(root_path, img_extensions = ['jpg', 'png', 'jpeg']):
    return reduce(
            lambda a, b: a+b,
            [glob.glob(os.path.join(root_path, f"*.{ext}")) for ext in img_extensions])
        
if __name__ == '__main__':
    print("Hello material helper.")

    U.reset_all()
    bpy.ops.object.light_add(type='POINT', location=(0, 0, .2))
    bpy.ops.mesh.primitive_plane_add(size=2)
    plane = bpy.context.object
        
    mat = MaterialHelper.load_material(
        name="mat_test", 
        path="/home/kelin/Documents/Rock_Ground_Coal_Texture/combine/ground/Gravel_Natural_vmciahxg_8K_surface_ms/",
        is_ballast=False)

    MaterialHelper.assign_material(plane, mat)
    MaterialHelper.scale_UV(plane)
    
    



