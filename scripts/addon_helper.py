"""Addon Management"""

import sys 
sys.path.append(".")

import bpy 
import addon_utils

from scripts import common

DEFAULT_ADDONS = [
    "molecular",               # provide molecular script addon to enable self/mutual collisions in particle systems
    "ant_landscape",           # provide support for landscape generation
    "node_wrangler",           # provide shortcuts and supports for material design in node mode 
    "cycles",                  # provide realistic rendering engine
    "add_mesh_extra_objects",  # provide support for stone generator
    "real_snow",               # provide support for automatic snow-like mesh covering   
]

class AddonHelper:
    """Class for manage addons and automatically register addon"""
    @classmethod
    def register(cls):
        """Run at the beginning of every scripts to register add-ons"""

        if True not in addon_utils.check("molecular"):
            bpy.ops.preferences.addon_install(
                overwrite=False, 
                filepath=f"{common.ROOT_DIR}/thirdparty/molecular_1.1.3_linux.zip"
            )
        
        for addon in DEFAULT_ADDONS:
            addon_utils.enable(addon)
            bpy.ops.preferences.addon_enable(module=addon)

def test():
    AddonHelper.register()

if __name__ == "__main__":
    test()