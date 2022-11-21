import sys
sys.path.append(".")

from bpy.types import Object

import os 
import glob
import cv2
import json
import numpy as np
import datetime


import bpy
import mathutils

from scripts import common as common
from scripts.blender_utils import BlenderUtils as U


class RenderHelper:

    @classmethod
    def render_scene_by_camera_trajectory(cls, scene, start, end, num_images, out_path, render_label=False, label_objects = None, mask_layer=None):
        bpy.ops.curve.primitive_nurbs_path_add()
        trajectory = bpy.context.selected_objects[0]
        trajectory.location = (start + end) / 2
        trajectory.scale[0] = np.linalg.norm(start - end)
        
        camera: Object = scene.ref.camera
        U.focus(camera)
        bpy.ops.object.constraint_add(type='FOLLOW_PATH')
        camera.constraints['Follow Path'].target = trajectory
        camera.constraints['Follow Path'].use_curve_follow = True
        camera.constraints['Follow Path'].forward_axis = "FORWARD_X"
        
        bpy.ops.constraint.followpath_path_animate(constraint='Follow Path')

        camera.location = (0, 0, 0)

        # U.focus(trajectory)
        trajectory.data.use_path = True
        trajectory.data.path_duration = num_images - 1
        scene.ref.render.filepath = out_path
        scene.ref.frame_start = 1
        scene.ref.frame_end = num_images
        # bpy.ops.render.render(animation=True, scene=scene.ref.name, write_still=True)
        cls.render_scene(scene, out_path, animation=True, render_label=render_label, label_objects=label_objects, mask_layer=mask_layer)

    @classmethod
    def render_scene(cls, scene, out_path, animation=False, render_label=False, label_objects=None, mask_layer=None):
        # get labels for each ballast
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
        file_out_node = tree.nodes.new(type='CompositorNodeOutputFile')
        file_out_node.base_path = out_path
        file_out_node.location = 500,100

        file_out_node.format.file_format = "PNG" # default is "PNG"
        file_out_node.format.color_mode = "RGB"  # default is "BW"
        file_out_node.format.color_depth = "16"  # default is 8
        file_out_node.format.compression = 0     # default is 15
       

        if not os.path.exists(file_out_node.base_path):
            os.makedirs(file_out_node.base_path)
        
        links.new(render_node.outputs[0], file_out_node.inputs[0])

        if render_label:

            for (i, obj) in enumerate(label_objects):
                obj.pass_index = i + 1
        
            scene.ref.view_layers["ViewLayer"].use_pass_object_index = True

            label_out_node = tree.nodes.new(type='CompositorNodeOutputFile')
            label_out_node.base_path = out_path
            label_out_node.location = 500,200

            label_out_node.format.file_format = "PNG" # default is "PNG"
            label_out_node.format.color_mode = "BW"  # default is "BW"
            label_out_node.format.color_depth = "16"  # default is 8
            label_out_node.format.compression = 0     # default is 15

            label_out_node.layer_slots.clear()
            label_out_node.layer_slots.new("ballast_label_0_")
            label_out_node.layer_slots.new("ballast_label_1_")

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

            links.new(render_node.outputs[2], divide_1_node.inputs[0])
            links.new(divide_1_node.outputs[0], floor_1_node.inputs[0])
            links.new(floor_1_node.outputs[0], divide_2_node.inputs[0])
            links.new(render_node.outputs[2], mod_1_node.inputs[0])
            links.new(mod_1_node.outputs[0], divide_3_node.inputs[0])
            links.new(divide_2_node.outputs[0], label_out_node.inputs[0])
            links.new(divide_3_node.outputs[0], label_out_node.inputs[1])

        bpy.ops.render.render(animation=animation, scene=scene.ref.name, write_still=not animation)
        if render_label:
            mask_layer.hide_render = True
            file_out_node.base_path = os.path.join(out_path, "uncovered")
            label_out_node.base_path = os.path.join(out_path, "uncovered")
            bpy.ops.render.render(animation=animation, scene=scene.ref.name, write_still=not animation)
            mask_layer.hide_render = False

        # clear default nodes
        for node in tree.nodes:
            tree.nodes.remove(node)

        scene.ref.use_nodes = False
    
    @classmethod
    def filter_labels(cls, path, fine_id_start, thresh=0.2, verbose=True, keep_largest_contour=True, min_num_pixel_per_label=10):

        coco = {
            "info": {
                "description": "Blender Synthetic Ballast", "date_created": datetime.datetime.now().strftime("%m/%d/%Y")
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [
                {"supercategory": "object","id": 0,"name": "ballast"}
            ]
        }

        labels_0_path = glob.glob(os.path.join(path, "ballast_label_0_*.png"))
        labels_1_path = glob.glob(os.path.join(path, "ballast_label_1_*.png"))

        for (label_0_path, label_1_path) in zip(labels_0_path, labels_1_path):
            
            parent, label_0_name = os.path.split(label_0_path)
            parent, label_1_name = os.path.split(label_1_path)

            uncovered_label_0_path = os.path.join(parent, "uncovered", label_0_name)
            uncovered_label_1_path = os.path.join(parent, "uncovered", label_1_name)

            uncovered_label_0 = cv2.imread(uncovered_label_0_path, cv2.IMREAD_GRAYSCALE)
            uncovered_label_1 = cv2.imread(uncovered_label_1_path, cv2.IMREAD_GRAYSCALE)
            covered_label_0 = cv2.imread(label_0_path, cv2.IMREAD_GRAYSCALE)
            covered_label_1 = cv2.imread(label_1_path, cv2.IMREAD_GRAYSCALE)

            H, W = uncovered_label_0.shape 

            f = np.zeros([H, W], dtype=np.int32)
            c = np.zeros([H, W], dtype=np.int32)

            raw_file_name = f"Image{label_0_name.split('_')[-1]}"
            raw_file_path = os.path.join(parent, raw_file_name)

            img_dict = {
                "id": common.hash_string(raw_file_path), "width": W, "height": H, "file_name": raw_file_name, "file_path": raw_file_path, "date_captured": datetime.datetime.now().strftime("%m-%d-%Y %H:%M:%S")
            }

            coco["images"].append(img_dict)

            for i in range(H):
                for j in range(W):
                    f[i,j] = uncovered_label_0[i,j] * 256 + uncovered_label_1[i,j]
                    c[i,j] = covered_label_0[i,j] * 256 + covered_label_1[i,j]

            valid_labels = []
            for i in range(1, 1+f.max()):
                if np.sum(f == i) > 0:
                    rate = np.sum(c==i) / np.sum(f == i)
                    if rate < thresh:
                        c[c==i] = 0
                    else:
                        valid_labels.append(i)

            out = np.zeros((H, W, 3), dtype=np.uint8)
            out_contoured = out.copy()

            def fill_hole_grayscale(img):
                im_flood = img.copy()
                mask = np.zeros((H+2, W+2), np.uint8)
                isbreak = False 
                seed_point = None
                for i in range(H):
                    for j in range(W):
                        if (img[i,j] == 0):
                            seed_point = (i, j)
                            isbreak = True
                            break 
                    if isbreak:
                        break 
                if seed_point is not None:
                    cv2.floodFill(im_flood, mask, seed_point, 255)
                    im_flood_inv = cv2.bitwise_not(im_flood)
                    im_out = img | im_flood_inv if np.sum(im_flood_inv) / np.sum(img) < 2 else img
                else:
                    im_out = img
                return im_out
            
            for i in valid_labels: 
                label_i = np.zeros([H, W], dtype=np.uint8)
                label_i[c==i] = 255
                label_i = fill_hole_grayscale(label_i)

                _, thr = cv2.threshold(label_i, 127, 255, 0)
                contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                min_xy = np.array([np.inf, np.inf])
                max_xy = np.array([-np.inf, -np.inf])
                for contour in contours:
                    contour = contour[:, 0, :]
                    mins = contour.min(axis=0)
                    min_xy = np.minimum(min_xy, mins)
                    maxs = contour.max(axis=0)
                    max_xy = np.maximum(max_xy, maxs)
                
                min_xy = min_xy.astype(np.int32)
                max_xy = max_xy.astype(np.int32)

                bbox = min_xy.tolist()
                bbox_scale = (max_xy-min_xy).tolist()
                bbox_area = bbox_scale[0] * bbox_scale[1]
                bbox.extend(bbox_scale)

                if keep_largest_contour:
                    largest_area = 0
                    largest_id = -1
                    for (j, contour) in enumerate(contours):
                        lij = np.zeros([H, W], dtype=np.uint8)
                        cv2.fillPoly(lij, contour[np.newaxis, :, 0, :], 255)
                        area = (lij > 0).sum()
                        if area > largest_area and area >= max(min_num_pixel_per_label, thresh*bbox_area): 
                            largest_area = area 
                            largest_id = j 
                    if largest_id == -1:
                        continue 
                    
                    contours = contours[largest_id:largest_id+1]
                    label_i = np.zeros([H, W], dtype=np.uint8)
                    cv2.fillPoly(label_i, contours[0][np.newaxis, :, 0, :], 255)
                else:
                    invalid_label_ids = []
                    for (j, contour) in enumerate(contours):
                        lij = np.zeros([H, W], dtype=np.uint8)
                        cv2.fillPoly(lij, contour[np.newaxis, :, 0, :], 255)
                        area = (lij > 0).sum()
                        if area < max(min_num_pixel_per_label, thresh*bbox_area):
                            invalid_label_ids.append(j)
                    for invalid_j in reversed(invalid_label_ids):
                        contours = contours[:invalid_j] + contours[invalid_j+1:]
                    
                    if len(contours) == 0:
                        continue
                    
                    label_i = np.zeros([H, W], dtype=np.uint8)
                    for contour in contours:
                        cv2.fillPoly(label_i, contour[np.newaxis, :, 0, :], 255)

                out[label_i > 0, 0] = i // 256
                out[label_i > 0, 1] = i % 256

                min_xy = np.array([np.inf, np.inf])
                max_xy = np.array([-np.inf, -np.inf])
                center_xy = np.array([0, 0], dtype=np.float32)
                for contour in contours:
                    contour = contour[:, 0, :]
                    mins = contour.min(axis=0)
                    min_xy = np.minimum(min_xy, mins)
                    maxs = contour.max(axis=0)
                    max_xy = np.maximum(max_xy, maxs)
                    center_xy += contour.mean(axis=0)
                
                min_xy = min_xy.astype(np.int32)
                max_xy = max_xy.astype(np.int32)

                center_xy /= len(contours)
                bbox = min_xy.tolist()
                bbox_scale = (max_xy-min_xy).tolist()
                bbox.extend(bbox_scale)

                segmentation = [ contour.flatten().tolist() for contour in contours]
                
                annotation_dict = {
                    "id": common.hash_string(f"{raw_file_path}_{i}"), 
                    "category_id": 0, 
                    "iscrowd": 0, 
                    "obj_id": i,
                    "obj_type": "fine" if i >= fine_id_start else "ballast",
                    "segmentation": segmentation, 
                    "image_id": common.hash_string(raw_file_path), 
                    "area": sum((label_i>0).flatten().tolist()), 
                    "bbox": bbox
                }
                coco["annotations"].append(annotation_dict)

                if verbose:
                    if not os.path.exists(f"{parent}/label_details"):
                        os.makedirs(f"{parent}/label_details")
                    for (j, contour) in enumerate(contours):
                        lij = np.zeros([H, W, 3], dtype=np.uint8)
                        lij[label_i > 0] = 255
                        cv2.drawContours(lij, [contour], -1, (0,0,255), 2)
                        cv2.rectangle(lij, min_xy.tolist(), max_xy.tolist(), (255, 0, 0), 2)
                        cv2.imwrite(f"{parent}/label_details/{raw_file_name[:-4]}_label_{i}_seg_{j}.png", lij)

                cv2.drawContours(out_contoured, contours, -1, (0,0,255), 3)
                out_contoured = cv2.putText(out_contoured, f'{i}', center_xy.astype(np.int32).tolist(), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            out_contoured = out_contoured | out
                
            cv2.imwrite(f"{parent}/filtered_label_{label_0_name.split('_')[-1]}", out)
            cv2.imwrite(f"{parent}/filtered_label_with_contours_{label_0_name.split('_')[-1]}", out_contoured)
        # print(coco)

        with open(f"{parent}/filtered_coco.json", 'w+') as f:
            json.dump(coco, f)

# def update_camera(camera, focus_point=mathutils.Vector((0.0, 0.0, 0.0)), distance=10.0):
    #     """
    #     Focus the camera to a focus point and place the camera at a specific distance from that
    #     focus point. The camera stays in a direct line with the focus point.

    #     :param camera: the camera object
    #     :type camera: bpy.types.object
    #     :param focus_point: the point to focus on (default=``mathutils.Vector((0.0, 0.0, 0.0))``)
    #     :type focus_point: mathutils.Vector
    #     :param distance: the distance to keep to the focus point (default=``10.0``)
    #     :type distance: float
    #     """
    #     looking_direction = camera.location - focus_point
    #     rot_quat = looking_direction.to_track_quat('Z', 'Y')

    #     camera.rotation_euler = rot_quat.to_euler()
    #     # Use * instead of @ for Blender <2.8
    #     camera.location = rot_quat @ mathutils.Vector((0.0, 0.0, distance))
