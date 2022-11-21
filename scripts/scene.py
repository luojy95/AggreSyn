import sys 
sys.path.append(".")

import json

from scripts.blender_utils import BlenderUtils as U

class Scene:
    def __init__(self, ref) -> None:
        self.object_dict = {}
        self.meta = {}
        self.ref = ref
    
    def restore(self, obj_path, meta_path) -> None:
        self.object_dict = json.load(open(obj_path, 'r'))
        self.meta = json.load(open(meta_path, 'r'))

    def register(self, id, obj, meta_info = None):
        self.object_dict[id] = obj.name

        if meta_info is None:
          self.meta[id] = {}
    
    def clear_scene(self):
        for id in self.object_dict:
            U.remove_obj(id)
    
    def pop(self, id):
        self.object_dict.pop(id)
        self.meta.pop(id)
    
