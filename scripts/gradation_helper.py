import sys 
sys.path.append(".")

import math
import numpy as np
from matplotlib import pyplot as plt

import bpy

from scripts import common
from scripts import math_utils as MU
from scripts.blender_utils import BlenderUtils as U

MAX_DIAMETER = 3 # inch
MIN_DIAMETER = 0 # inch
MAX_FINE_DIAMETER = 0.375 # #3/8 inch


class Gradation:
    
    def __init__(self, keys = [], cdf = None, pdf = None) -> None:
        if cdf != None:
            pdf = MU.cdf2pdf(cdf)
        elif pdf != None:
            cdf = MU.pdf2cdf(pdf)
        
        self.cdf = { key: prob for (key, prob) in zip(keys, cdf)}
        self.input_cdf = { key: prob for (key, prob) in zip(keys, cdf)}

        self.pdf = { (keys[i], keys[i+1]): pdf[i] for i in range(len(keys)-1)}

        self.fine_ranges = []
        for (d_min, d_max) in self.pdf:
            if d_max <= MAX_FINE_DIAMETER:
                self.fine_ranges.append((d_min, d_max))
        
    def gen_ballast(self, total_ballast = 5000):

        self.count = {}
        norm_coeff = 0
        for d_range in self.pdf:
            self.count[d_range] = self.pdf[d_range] / MU.calc_avg_sphere_volume(d_range[0], d_range[1])
            if d_range not in self.fine_ranges:
                norm_coeff += self.count[d_range]
        
        for d_range in self.pdf:
            self.count[d_range] = round(total_ballast * self.count[d_range] / norm_coeff)

        ballast_diameters = []
        self.total_fine_vol = 0
        self.total_ballast_vol = 0
        self.fine_grain_cnt = sum([self.count[d_range] for d_range in self.fine_ranges])
        for d_range in self.pdf:
            if d_range not in self.fine_ranges and self.count[d_range] > 0:
                ballast_diameters.extend(np.random.uniform(d_range[0], d_range[1], self.count[d_range]).tolist())
                self.total_ballast_vol += MU.calc_avg_sphere_volume(d_range[0], d_range[1]) * self.count[d_range]
            
            if d_range in self.fine_ranges:
                self.total_fine_vol += MU.calc_avg_sphere_volume(d_range[0], d_range[1]) * self.count[d_range]
        
        self.expected_fine_size = MU.calc_sphere_diameter_from_vol(self.total_fine_vol/self.fine_grain_cnt)

        # self.calibrate_from_diameters(ballast_diameters, self.get_fine_vol())

        np.random.shuffle(ballast_diameters)

        return ballast_diameters
    
    def get_fine_vol(self, inch=True):

        if inch:
            return MU.calc_sphere_vol(diameter=self.expected_fine_size)* self.fine_grain_cnt
        else:
            return  MU.calc_sphere_vol(diameter=MU.inch2meter(self.expected_fine_size))* self.fine_grain_cnt


    def calibrate_from_diameters(self, d_list, fine_vol):

        d_list_sort = sorted(d_list)

        v_list = list(map(MU.calc_sphere_vol, d_list_sort))

        total_vol = sum(v_list) + fine_vol

        new_cdf = [0]
        new_keys = [MIN_DIAMETER]
        
        keys = sorted(list(self.cdf.keys()))

        cur_key_id = 0
        cur_vol = fine_vol

        while keys[cur_key_id] < MAX_FINE_DIAMETER:
            cur_key_id += 1
        
        new_cdf.append(cur_vol / total_vol)
        new_keys.append(keys[cur_key_id])

        for (d, v) in zip(d_list_sort, v_list):

            if d > keys[cur_key_id + 1]:
                cur_key_id += 1
                new_cdf.append(cur_vol / total_vol)
                new_keys.append(keys[cur_key_id])
            
            cur_vol += v 

        new_pdf = MU.cdf2pdf(new_cdf)

        self.cdf = { key: prob for (key, prob) in zip(new_keys, new_cdf)}

        self.pdf = { (new_keys[i], new_keys[i+1]): new_pdf[i] for i in range(len(new_keys)-1)}
    
    def draw_gradation(self, path):

        plt.plot(MU.inch2meter(np.array(list(self.input_cdf.keys()))) * 1000, list(self.input_cdf.values()), 'r', label="input")
        plt.plot(MU.inch2meter(np.array(list(self.cdf.keys()))) * 1000, list(self.cdf.values()), 'b', label="synthetic")
        plt.xscale('log')
        plt.legend()
        plt.xlabel("size (mm)")
        plt.ylabel("gradation")
        plt.ylim([0, 1])
        plt.savefig(path)



class GradationHelper:

    BALLAST_DENSITY = 2500

    @classmethod
    def volume_calculate(cls, ballast_obj) -> float:
        U.unselect_all()
        ballast_obj.select_set(True)
        
        U.set_active(ballast_obj)
        bpy.ops.rigidbody.mass_calculate(material="Custom", density=cls.BALLAST_DENSITY)
        
        U.set_active(None)
        U.unselect_all()

        
        rigid_body = ballast_obj.rigid_body

        if rigid_body:
            mass = rigid_body.mass
            volume = mass / cls.BALLAST_DENSITY

            # print("volume:", volume, "mass:", mass)
            return volume
        else:
            return -1
    

    @classmethod
    def diameter_calculate(cls, ballast_obj, inch = True) -> float:
        
        volume = cls.volume_calculate(ballast_obj=ballast_obj)

        radius = (3 / 4 * volume / math.pi) ** (1/3)

        return MU.meter2inch(2*radius) if inch else 2*radius
    
    @classmethod
    def calc_fine_volume(cls, height, disp_obj, base_obj, ballasts_obj_list, scene_size, valid_rate = 1.0, operations=["DIFFERENCE"]):
        tmp_disp = U.copy_obj(disp_obj)
        tmp_disp.location[2] = height 
        tmp_base = U.copy_obj(base_obj)
        
        disp_mesh = U.build_mesh_from_two_square_planes(
            top = tmp_disp,
            bottom = tmp_base,
            size = scene_size
        )

        U.focus(disp_mesh)

        # for ballast in ballasts_obj:
        for (ballasts_obj, op) in zip(ballasts_obj_list, operations):

            bpy.ops.object.modifier_add(type='BOOLEAN')
            disp_mesh.modifiers["Boolean"].operation = op
            disp_mesh.modifiers["Boolean"].object = ballasts_obj
            bpy.ops.object.modifier_apply(modifier=disp_mesh.modifiers["Boolean"].name)


        bpy.ops.rigidbody.object_add(type="ACTIVE")
        disp_mesh.rigid_body.collision_shape = "MESH"

        vol = cls.volume_calculate(disp_mesh) * valid_rate

        U.delete_obj(disp_mesh)

        return vol

keys = [MAX_DIAMETER, 2.5, 2, 1.5, 1, 3/4, 3/8, 0.197, 0.0787, MIN_DIAMETER]
weights = [
    0,
    4086.8-620.3, 
    12632.4-626.3, 
    10619.2-594.7, 
    11011.4-649.1, 
    6241.5-591.1+39.2,  
    7028.8-572.6, 
    3669.1-642.2, 
    2376.4-643.4, 
    2427.3-611.9 + 946-643.5 + 1082.5-635.5 + 1009.4-607.7 + 1975.6-652.0
]
keys.reverse()
test_gradation = Gradation(keys, cdf = MU.pdf2cdf(list(reversed(weights))))

FI_keys = [
    MAX_DIAMETER, 
    2.5, 
    2, 
    1.5, 
    1, 
    3/4, 
    3/8, 
    MU.meter2inch(0.00236), #8
    MU.meter2inch(0.00118), #16
    MU.meter2inch(0.0006), #30
    MU.meter2inch(0.0003), #50
    MU.meter2inch(0.00015), #100
    MU.meter2inch(0.000075), #200
    MIN_DIAMETER #PAN
]
FI_keys.reverse()

weights_FI7 = [
    0, 
    9050.7114,
    39547.275,
    66418.025,
    94893.423,
    33450.068,
    4451.0976,
    3904.6225,
    4279.8536,
    2727.2727,
    1569.1525,
    929.5609,
    1986.0786,
    1583.2974
]
weights_FI14 = [
    0, 
    9050.7114,
    39547.275,
    66418.025,
    94893.423,
    33450.068,
    4451.0976,
    8447,
    9258.75,
    5900,
    3394.6,
    2010.95,
    4296.55,
    3425.2
]
weights_FI23 = [
    0, 
    9050.7114,
    39547.275,
    66418.025,
    94893.423,
    33450.068,
    4451.0976,
    15204.6,
    16665.75,
    10620,
    6110.28,
    3619.71,
    7733.79,
    6165.36
]
weights_FI30 = [
    0, 
    9050.7114,
    39547.275,
    66418.025,
    94893.423,
    33450.068,
    4451.0976,
    21962.2,
    24072.75,
    15340,
    8825.96,
    5228.47,
    11171.03,
    8905.52
]
weights_FI39 = [
    0, 
    9050.7114,
    39547.275,
    66418.025,
    94893.423,
    33450.068,
    4451.0976,
    32699.7,
    34631.85,
    22005.7,
    12374.26,
    7286.27,
    14761.53,
    13871.82
]
grad_keys = [
    MAX_DIAMETER, 
    2.5, 
    2, 
    1.5, 
    1, 
    3/4, 
    3/8, 
    MU.meter2inch(0.00475), #4
    MU.meter2inch(0.002), #10
    MU.meter2inch(0.000425), #40
    MU.meter2inch(0.0003), #50
    MU.meter2inch(0.00015), #100
    MU.meter2inch(0.000075), #200
    MIN_DIAMETER #PAN
]
grad_keys.reverse()
Grad_1 = [
    0,
    0, 
    0,
    0,
    554.4+1191.8+523.4,
    488+807.2+699.6,
    841.5+820.7+1169.3,
    360.7+466.1+810.8,
    199.7+243.7+545.7,
    306.6+352.8+1085.7,
    111.3+124.2+376.9,
    119.4+134.6+198.8,
    60+70.9+147.7,
    74.5+70.9+48.2
]
Grad_2 = [
    0, 
    0,
    0,
    325.5+152.5+193.5,
    1342.4+1136.6+1005.7+895.4,
    747.3+766.8+788.2+651.1,
    868.1+1063.3+995.8+1050.3,
    413.6+452+546+679.2,
    240.5+217.3+262.1+437,
    362.8+257.1+319.9+617,
    126.8+87.2+100.2+200.5,
    144.2+96.2+111.3+207.3,
    68.2+48.9+52.9+97.3,
    136.4+103.8+110.8+218.2,
]


FI7_gradation = Gradation(FI_keys, cdf = MU.pdf2cdf(list(reversed(weights_FI7))))
FI14_gradation = Gradation(FI_keys, cdf = MU.pdf2cdf(list(reversed(weights_FI14))))
FI23_gradation = Gradation(FI_keys, cdf = MU.pdf2cdf(list(reversed(weights_FI23))))
FI30_gradation = Gradation(FI_keys, cdf = MU.pdf2cdf(list(reversed(weights_FI30))))
FI39_gradation = Gradation(FI_keys, cdf = MU.pdf2cdf(list(reversed(weights_FI39))))
Grad1_gradation = Gradation(grad_keys, cdf = MU.pdf2cdf(list(reversed(Grad_1))))
Grad2_gradation = Gradation(grad_keys, cdf = MU.pdf2cdf(list(reversed(Grad_2))))

example_gradations = {
    "FI7": FI7_gradation,
    "FI14": FI14_gradation,
    "FI23": FI23_gradation,
    "FI30": FI30_gradation,
    "FI39": FI39_gradation,
    "Grad1": Grad1_gradation,
    "Grad2": Grad2_gradation,
}

if __name__ == "__main__":

    # keys = [MAX_DIAMETER, 2.5, 2, 1.5, 1, 3/4, 3/8, 0.197, 0.0787, MIN_DIAMETER]
    # weights = [
    #     0,
    #     4086.8-620.3, 
    #     12632.4-626.3, 
    #     10619.2-594.7, 
    #     11011.4-649.1, 
    #     6241.5-591.1+39.2,  
    #     7028.8-572.6, 
    #     3669.1-642.2, 
    #     2376.4-643.4, 
    #     2427.3-611.9 + 946-643.5 + 1082.5-635.5 + 1009.4-607.7 + 1975.6-652.0
    # ]
    # cdf = MU.pdf2cdf(list(reversed(weights)))
    # pdf = MU.cdf2pdf(cdf)

    # keys.reverse()
    
    # gradation = Gradation(keys, cdf)

    gradation = FI23_gradation

    print(gradation.cdf)

    gradation.gen_ballast(total_ballast=500)

    print(gradation.expected_fine_size)

    print(gradation.cdf)

    print(gradation.count)
    print(gradation.total_ballast_vol, gradation.total_fine_vol)
