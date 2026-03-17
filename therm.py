# this file uses click to parse a thermal config, a deepflow config, and a deepflow output file.
# it then performs thermal analysis.

# TODO: Before pushing this anywhere remove the token from thermal_simulators/anemoi_sim.py!!!!

import click
import yaml
import sys
import math
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from thermal_simulators.factory import SimulatorFactory
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from rearrange import *
from rearrange import *
from therm_xml_parser import *
import time
from bonding_xml_parser import *
from heatsink_xml_parser import *
import pickle
import os
import subprocess
import re
import shutil
from collections import defaultdict

import numpy as np
try:
    import seaborn as sns
except Exception:
    sns = None

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
except Exception:
    LinearRegression = None
    r2_score = None

import re
from pathlib import Path
from typing import List, Tuple
import csv

if sns is not None:
    sns.set()

# how many times*min_dist should we move
MOVE_MULTIPLIER = 1
# how much should we inflate for overlap checking
INFLATION = 1

conductivity_values = {
    "Air": 0.025,
    "FR-4": 0.1,
    "Cu-Foil": 400,
    "Si": 105,
    "Aluminium": 205,
    "TIM001": 100,
    "Glass": 1.36,
    "TIM": 100,
    "SnPb 67/37": 36,
    "Epoxy, Silver filled": 1.6,
    "SiO2": 1.1,
    "AlN": 237,
    "EpAg": 1.6,
    "Infill_material": 19,
    "Polymer1": 675,
    "TIM0p5": 5.0 # 0.5 # 100.0 # 
}
# EpAg is Epoxy, Silver filled used in layer_definitions.xml for bonding layers 5nm_HBM2HBM_metal.

class Pin():
    def __init__(self,name,parent_name):
        self.name = name
        self.parent_name = parent_name
        self.edge_name = None

    def assign_to_edge(self,edge_name):
        self.edge_name = edge_name

    def is_assigned(self):
        return self.edge_name is not None

# def parse_pin_map(pin_map,gpu_coords):

#     # print("GPU coords received:",gpu_coords)
#     # this is a pin map
#     #   pin_map: [
#     #     "PIXXX
#     #      XIXXX
#     #      XIDIP
#     #      XIXXX
#     #      PIXXX"
#     #   ]
#     # P is pin, D is device, X is air, I is ISOLATOR
#     # lower left corner of the GPU is (gpu_coords[0],gpu_coords[1]).
#     # top right is (gpu_coords[2],gpu_coords[3])
#     # the pin map is a string. We need to parse it and assign coordinates to each pin.
#     pin_coords = []
#     # the pin map could be any size, but has to be odd x odd number. check that first
#     pin_map = pin_map[0].split(' ')
#     for i in range(len(pin_map)):
#         # strip whitespace
#         pin_map[i] = pin_map[i].strip()
#     num_rows = len(pin_map)
#     num_cols = len(pin_map[0])
#     if num_rows % 2 == 0 or num_cols % 2 == 0:
#         raise ValueError("Pin map must be odd x odd")
#     # now parse the pin map
#     die_length = gpu_coords[3] - gpu_coords[1]
#     die_width = gpu_coords[2] - gpu_coords[0]
#     # now subdivide that into num_rows x num_cols
#     # for example, if 3x3 and die_length is 10, I want points at:
#     # 0, 5, 10
#     row_height = die_length / (num_rows-1)
#     col_width = die_width / (num_cols-1)
#     for i in range(num_rows):
#         for j in range(num_cols):
#             if pin_map[i][j] == 'P':
#                 pin_coords.append((gpu_coords[0]+j*col_width,gpu_coords[1]+(num_rows-1-i)*row_height))

#     # I now also want to extract isolator coordinates.
#     # the isolators are thin strips, so we want to merge isolators across columns and rows.
#     isolator_coords = []
#     for i in range(num_rows):
#         for j in range(num_cols):
#             if pin_map[i][j] == 'I':
#                 # check if this is a new isolator
#                 if (i == 0 or pin_map[i-1][j] != 'I') and (j == 0 or pin_map[i][j-1] != 'I'):
#                     # this is the start of a new isolator
#                     start_x = gpu_coords[0]+(j-1)*col_width
#                     start_y = gpu_coords[1]+(num_rows-1-i)*row_height
#                     # now find the end of the isolator
#                     end_x = start_x
#                     end_y = start_y
#                     j_acc = j
#                     while j < num_cols and pin_map[i][j_acc] == 'I':
#                         j_acc += 1
#                         if j_acc == num_cols:
#                             break
#                         end_x += col_width
#                     i_acc = i
#                     while i < num_rows and pin_map[i_acc][j] == 'I':
#                         i_acc += 1
#                         if i_acc == num_rows:
#                             break
#                         end_y -= row_height
#                     # now, remember that isolators are not boxes but lines.
#                     # so we need to ensure that start_x == end_x
#                     # but do we set start_x = end_x or end_x = start_x?
#                     # this depends on whether the isolator is to the left or to the right
#                     middle_x = gpu_coords[0]+(num_cols-1)/2*col_width
#                     if end_x <= middle_x:
#                         # left of gpu
#                         start_x = gpu_coords[0]
#                         end_x = start_x
#                     else:
#                         # right of gpu
#                         end_x = gpu_coords[2]
#                         start_x = end_x
#                     isolator_coords.append((start_x,start_y,end_x,end_y))

#     # print("Isolator coords:",isolator_coords)
#     return pin_coords, isolator_coords

def draw_fig(boxes,out_dir,out_name,limits):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    for box in boxes:
        if box.name.endswith("interposer"):
            ax.add_patch(plt.Rectangle((box.start_x, box.start_y), box.width, box.length, fill=True,color='black'))
        elif box.name[:-1].endswith('HBM'):
            ax.add_patch(plt.Rectangle((box.start_x, box.start_y), box.width, box.length, fill=True,color='blue'))
        elif box.name.endswith('wafer'):
            ax.add_patch(plt.Rectangle((box.start_x, box.start_y), box.width, box.length, fill=True,color='green'))
        else:
            ax.add_patch(plt.Rectangle((box.start_x, box.start_y), box.width, box.length, fill=True,color='red'))
        # ax.text(box.start_x+box.width/2,box.start_y+box.length/2,box.name,ha='center',va='center', color='white')
    plt.xlim(limits[0],limits[1])
    plt.ylim(limits[2],limits[3])
    plt.savefig(out_dir+'/'+out_name+'.png')
    plt.close()

# def draw_fig_3D_alt(boxes, out_dir, out_name):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
    
#     collections = []

#     for box in boxes:
#         x = [box.start_x, box.start_x + box.width, box.start_x + box.width, box.start_x, box.start_x]
#         y = [box.start_y, box.start_y, box.start_y + box.length, box.start_y + box.length, box.start_y]
#         z = [box.start_z] * 5  # base z level
#         z_top = [box.start_z + box.height] * 5  # top z level

#         verts = [list(zip(x, y, z)),
#                 list(zip(x, y, z_top))]

#         # Sides of the box
#         for i in range(4):
#             verts.append([(x[i], y[i], z[i]), (x[i+1], y[i+1], z[i+1]), 
#                         (x[i+1], y[i+1], z_top[i+1]), (x[i], y[i], z_top[i])])

#         # set zpos according to x position
#         # closer means higher zpos

#         zpos = box.start_x

#         if box.name == 'interposer':
#             color = 'black'
#         elif 'HBM' in box.name:
#             color = 'blue'
#         else:
#             color = 'red'


#         poly = Poly3DCollection(verts, facecolors=color, edgecolors='k', alpha=0.7)  # Adjust alpha for better visibility
#         poly.set_sort_zpos(zpos)
#         collections.append((poly, box))

#     for poly, box in collections:
#         ax.add_collection3d(poly)
#         # ax.text(box.start_x + box.width / 2, box.start_y + box.length / 2, box.start_z + box.height / 2, box.name, ha='center', va='center', color='white')

#     ax.set_xlim(-15, 75)
#     ax.set_ylim(-15, 75)
#     ax.set_zlim(0, 90) 
#     plt.savefig(out_dir + '/' + out_name + '3D.png')
#     plt.close()

def draw_fig_3D_zoom(boxes, out_dir, out_name, limits):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    collections = []

    for box in boxes:
        x = [box.start_x, box.start_x + box.width, box.start_x + box.width, box.start_x, box.start_x]
        y = [box.start_y, box.start_y, box.start_y + box.length, box.start_y + box.length, box.start_y]
        z = [box.start_z] * 5  # base z level
        z_top = [box.start_z + box.height] * 5  # top z level

        verts = [list(zip(x, y, z)),
                 list(zip(x, y, z_top))]

        # Sides of the box
        for i in range(4):
            verts.append([(x[i], y[i], z[i]), (x[i+1], y[i+1], z[i+1]), 
                          (x[i+1], y[i+1], z_top[i+1]), (x[i], y[i], z_top[i])])

        zpos = box.start_x

        if box.name.endswith('interposer'):
            color = 'black'
        elif box.name[:-1].endswith('HBM'):
            color = 'blue'
        elif box.name.endswith('wafer'):
            color = 'green'
        else:
            color = 'red'

        poly = Poly3DCollection(verts, facecolors=color, edgecolors='k', alpha=0.7)
        poly.set_sort_zpos(zpos)
        collections.append((poly, box))

    for poly, box in collections:
        ax.add_collection3d(poly)
        # Uncomment to add labels
        # ax.text(box.start_x + box.width / 2, box.start_y + box.length / 2, box.start_z + box.height / 2, box.name, ha='center', va='center', color='white')

    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[0], limits[1])
    ax.set_zlim(limits[0], int(limits[1]//2))  

    ax.view_init(elev=20, azim=30)  # Adjust the viewing angle
    ax.dist = 1  # Decrease this value to zoom in

    plt.savefig(out_dir + '/' + out_name + '3D.png')
    plt.close()

def determine_draw_lim(boxes):
    # find the min and max x and y
    min_x = 100000
    max_x = -100000
    min_y = 100000
    max_y = -100000
    for box in boxes:
        if box.start_x < min_x:
            min_x = box.start_x
        if box.start_x + box.width > max_x:
            max_x = box.start_x + box.width
        if box.start_y < min_y:
            min_y = box.start_y
        if box.start_y + box.length > max_y:
            max_y = box.start_y + box.length

    return min_x-5,max_x+5,min_y-5,max_y+5


# dedeepyo : 25-Feb-2025 : Implementing power source representation.
def recursively_lift_box(chiplet, box_list, height):
    box = chiplet.get_box_representation()
    box.start_z = box.start_z + height
    # print("Lifting box " + box.name + " to z : " + str(box.start_z))
    for child in chiplet.get_child_chiplets():
        recursively_lift_box(child, box_list, height)
    return

def create_power_source_backside(boxes, efficiency = 0.9):
    # ps = Box(x_coord,y_coord,z_coord,width,length,height,power,stackup,0,"Power Source")
    # recursively_lift_box(boxes[0].chiplet_parent, boxes, ps.height)
    # boxes.append(ps)
    total_power = 0
    for box in boxes:
        total_power += box.power
    
    ps = [box for box in boxes if box.chiplet_parent.get_chiplet_type() == "Power_Source"][0] # Assuming only one power source. For now, it is at bottom. backside power delivery.
    ps.power = (1 - efficiency) * total_power / efficiency
    ps.chiplet_parent.set_power(ps.power)
    # print("Power source power : " + str(ps.power))
# dedeepyo : 25-Feb-2025 #

# dedeepyo : 30-Jan-2025 : Bonding layer addition 
# height : Thickness of the bonding layer; determined based on input from Krutikesh.
# For Box, both z and dz are in mm. For PCBLayer, thickness is in um. For PCB, z is in mm.
# height here is in um.
def calculate_ratio(bonding, box):
    radius = bonding.get_diameter() / 2
    n_min_x = max(math.ceil((radius - bonding.get_offset()) / bonding.get_pitch()), 0)
    n_max_x = math.floor((1000 * box.width - bonding.get_offset() - radius) / bonding.get_pitch())
    n_x = n_max_x - n_min_x + 1
    # print("n_min_x : " + str(n_min_x) + " n_max_x : " + str(n_max_x))
    n_min_y = max(math.ceil((radius - bonding.get_offset()) / bonding.get_pitch()), 0)
    n_max_y = math.floor((1000 * box.length - bonding.get_offset() - radius) / bonding.get_pitch())
    n_y = n_max_y - n_min_y + 1
    # print("n_min_y : " + str(n_min_y) + " n_max_y : " + str(n_max_y))
    # print(n_x, n_y, box.width, box.length, radius, bonding.get_offset(), bonding.get_pitch())
    if(n_x <= 0 or n_y <= 0):
        return 0.00
    else:
        if(bonding.get_shape() == "sphere"):
            unit_volume = 4 * math.pi * (radius ** 3) / 3
        elif(bonding.get_shape() == "cylinder"):
            unit_volume = math.pi * (radius ** 2) * bonding.get_height()
        elif(bonding.get_shape() == "cuboid"):
            unit_volume = bonding.get_cross_section_area() * bonding.get_height()
        else:
            raise Exception("Invalid bonding shape")
        
        # print("Unit volume is : " + str(unit_volume))
        material_volume = unit_volume * n_x * n_y
        # print("Material volume is : " + str(material_volume))
        # print("Bonding layer volume is : " + str(1000 * box.width * 1000 * box.length * bonding.get_height()))
        ratio = material_volume / (1000 * box.width * 1000 * box.length * bonding.get_height())
        return ratio

# dedeepyo : 12-Feb-2025 : Implementing real children finder to avoid creating bonding for fake chiplets.
def get_real_children_recursive(chiplet_list):
    real_children = []
    fake_children = []
    for chiplet in chiplet_list:
        child_chiplets = chiplet.get_child_chiplets()
        for child_chiplet in child_chiplets:
            if(child_chiplet.get_fake() == False):
                real_children.append(child_chiplet)
            else:
                fake_children.append(child_chiplet)

    if(fake_children != []):
        real_children.extend(get_real_children_recursive(fake_children))

    return real_children
# dedeepyo : 12-Feb-2025 #

def create_all_bonding(box_list, name_type_dict, bonding_list):
    bonding_box_list = []

    bondings = {b.get_name():b for b in bonding_list}
    bonding_dict = {}
    for key in name_type_dict:
        x, y = key.split("#")
        if x not in bonding_dict:
            bonding_dict[x] = {}
        if y not in bonding_dict[x]:
            bonding_dict[x][y] = bondings[name_type_dict[key]]

    bm = [box for box in box_list if box.start_z == 0.000000]
    for b in bm:
        chiplet_tree_root = b.chiplet_parent
        create_bonding(box_list, chiplet_tree_root, bonding_dict, bonding_box_list)
    
    return bonding_box_list

# Assuming bonding only between interposer and GPU or HBM, not between stacked chiplets. # This assumption is no longer valid.
def create_bonding(box_list, base_chiplet, bonding_dict, bonding_box_list):
    children_of_base = base_chiplet.get_child_chiplets()
    for chiplet in children_of_base:
        try:
            bonding = bonding_dict[base_chiplet.get_chiplet_type()].get(chiplet.get_chiplet_type())
        except:
            bonding = None

        # dedeepyo : 03-Dec-2025 : Testing bonding for GPU on top.
        # print(f"Bonding for {base_chiplet.get_chiplet_type()} with {chiplet.get_chiplet_type()} above it.")
        # dedeepyo : 03-Dec-2025

        if bonding is not None:
            material = bonding.get_material()
            height = bonding.get_height()
            height_mm = height / 1000

            height_mm = round(height_mm, 6)
            box = chiplet.get_box_representation()
            x = round(box.start_x, 6)
            y = round(box.start_y, 6)
            z = round(box.start_z, 6)
            dx = round(box.width, 6)
            dy = round(box.length, 6)

            ratio = 100 * calculate_ratio(bonding, box)
            bonding_box = Box(x, y, z, dx, dy, height_mm, 0.0, f"1:{material}:{ratio},Epoxy, Silver filled:{100 - ratio}", 0.0, f"{chiplet.name}_bonding")
            recursively_lift_box(chiplet, box_list, height_mm)
            bonding_box_list.append(bonding_box)

        # if(bonding_dict.get(chiplet.get_chiplet_type()) is not None):
        #     create_bonding(box_list, chiplet, bonding_dict, bonding_box_list)

        create_bonding(box_list, chiplet, bonding_dict, bonding_box_list)

def recursively_lift_box(chiplet, box_list, height):
    box = chiplet.get_box_representation()
    box.start_z = box.start_z + height
    # print("Lifting box " + box.name + " to z : " + str(box.start_z))
    for child in chiplet.get_child_chiplets():
        recursively_lift_box(child, box_list, height)
    return

# dedeepyo : 9-Apr-25 : Copying over TIM creation.
def create_TIM_to_heatsink(box_list, material = "TIM0p5", min_TIM_height = 0.1, system_type = None):
    TIM_boxes = []
    z_min = max([box.end_z for box in box_list])
    z = z_min
    # top_box = [b for b in box_list if b.end_z == z_min][0]
    # print("Highest box is : " + top_box.name + " which ends at z : " + str(z_min))
    # print("The bottom of heat sink is at z : " + str(z))
    if system_type != "3D_1GPU_top":
        for box in box_list:
            # dedeepyo : 12-Feb-25 : A fake chiplet always has child chiplets; that is the purpose why it is made. Also, a box here is never fake.
            if(box.chiplet_parent.get_child_chiplets() == []):
                tim_height = z - box.end_z
                # if(tim_height != 0):
                if(tim_height > 0.0001):
                    # print("TIM height is : " + str(tim_height) + " for box " + box.name + " which ends at z : " + str(box.end_z))
                    tim_box = Box(box.start_x, box.start_y, box.end_z, box.width, box.length, tim_height, 0.0, f"1:{material}", 0.0, f"{box.name}_TIM")
                    TIM_boxes.append(tim_box)
                    # print(tim_data["name"] + " starts from z : " + tim_data["z"] + " and has height : " + tim_data["dz"])
                # else:
                #     print("The top of PCB " + box.name + " is at the same level as the bottom of heat sink!")
        
    intp_list = [b for b in box_list if b.chiplet_parent.get_chiplet_type() == "interposer" or b.chiplet_parent.get_chiplet_type() == "substrate"]
    intp = intp_list[0]
    for i in intp_list:
        if(i.end_z > intp.end_z):
            intp = i

    tim_box = Box(intp.start_x, intp.start_y, z_min, intp.width, intp.length, min_TIM_height, 0.0, f"1:{material}", 0.0, f"{intp.name}_TIM")
    TIM_boxes.append(tim_box)
    return TIM_boxes

# dedeepyo : 21-Jun-25 : Creating multiple heat sinks for a chiplet tree.
def calculate_GPU_HBM_HTC(box_list, power_dict, hc):
    # Assumption that we already have temperature data with default HTC hc. We use those temperatures to calculate the individual HTC for GPU and HBM.
    GPU_HTC = 15236.73003 # hc
    HBM_HTC = 2729.690335 # hc
    return GPU_HTC, HBM_HTC

def create_multiple_heat_sinks(box_list, heatsink_list, heatsink_name, power_dict, min_TIM_height = 0.01): # , scale_factor_x = 0, scale_factor_y = 0, area_scale_factor = 0):
    # dedeepyo : 7-Feb-2025 : Heatsink object creation.
    heatsinks = [h for h in heatsink_list if h.get_name() == heatsink_name]
    if(heatsinks is None):
        raise Exception("Heatsink not found")
    
    fin_height = heatsinks[0].get_fin_height()
    fin_thickness = heatsinks[0].get_fin_thickness()
    material = heatsinks[0].get_material()
    fin_number = heatsinks[0].get_fin_count()
    hc = heatsinks[0].get_hc()
    fin_offset = heatsinks[0].get_fin_offset()
    dz = heatsinks[0].get_base_thickness()
    bind_to_ambient = heatsinks[0].get_bind_to_ambient()

    if(fin_number == 0):
        if(fin_thickness > 0):
            fin_number = int(dx / (2 * fin_thickness))
    
    excluded_types = ["interposer", "substrate", "PCB", "Power_Source"]
    box_list_min = [box for box in box_list if box.chiplet_parent.get_chiplet_type() not in excluded_types]
    z_min = max([box.end_z for box in box_list_min])
    z = z_min + min_TIM_height
    
    # x_min = min([box.start_x for box in box_list_min])
    # x_max = max([box.end_x for box in box_list_min])
    # y_min = min([box.start_y for box in box_list_min])
    # y_max = max([box.end_y for box in box_list_min])
        
    # if(area_scale_factor != 0):
    #     dimension_scale_factor = math.sqrt(area_scale_factor)
    #     dx = dimension_scale_factor * (x_max - x_min)
    #     dy = dimension_scale_factor * (y_max - y_min)
    # elif(scale_factor_x == 0):
    #     if(scale_factor_y == 0):
    #         dy = heatsinks[0].get_base_length()
    #     else:
    #         dy = scale_factor_y * (y_max- y_min)
    #     dx = heatsinks[0].get_base_width()
    # elif(scale_factor_y == 0):
    #     dy = heatsinks[0].get_base_length()
    #     dx = scale_factor_x * (x_max - x_min)
    # else:
    #     dx = scale_factor_x * (x_max - x_min)
    #     dy = scale_factor_y * (y_max- y_min)

    # x = (x_max + x_min - dx) / 2
    # y = (y_max + y_min - dy) / 2

    GPU_HTC = hc
    HBM_HTC = hc
    GPU_HTC, HBM_HTC = calculate_GPU_HBM_HTC(box_list, power_dict, hc)

    HTC_dict = {
        "GPU_HTC": GPU_HTC / 1000,  # Convert to kW/m^2-K
        "HBM_HTC": HBM_HTC / 1000
    }
    power_dict.update(HTC_dict)
    heatsink_data_list = []
    chiplet_HTC = {
        "GPU": "GPU_HTC",
        "HBM": "HBM_HTC"
    }
    box_list_top = []
    for box in box_list:
        if(box.chiplet_parent.get_child_chiplets() == []):
            box_list_top.append(box)
    for box in box_list_top:
        chiplet_type = box.chiplet_parent.get_chiplet_type()[0:3]
        hc = chiplet_HTC.get(chiplet_type, None)
        if(hc is not None):
            # print(f"HTC is {hc} for chiplet type {chiplet_type}")
            x = box.start_x
            y = box.start_y
            dx = box.width
            dy = box.length
            heatsink_data = {
                "name": f"HS_top_{box.name}",
                "index": 0,
                "material": material,  # Cu-Foil
                "x": str(round(x, 6)),
                "y": str(round(y, 6)),
                "base_dx": str(round(dx, 6)),
                "base_dy": str(round(dy, 6)),
                "z": str(round(z, 6)),
                "base_dz": str(round(dz, 6)),
                "fin_height": str(fin_height),
                "fin_thickness": str(fin_thickness),
                "fin_count": str(fin_number),
                "fin_axis": "Y",
                "hc": str(hc),
                "bound": bind_to_ambient
            }
            heatsink_data_list.append(heatsink_data)
    
    return heatsink_data_list, power_dict
# dedeepyo : 21-Jun-25

def create_heat_sink(box_list, heatsink_list, heatsink_name, min_TIM_height = 0.01, scale_factor_x = 0, scale_factor_y = 0, area_scale_factor = 0):
    # dedeepyo : 7-Feb-2025 : Heatsink object creation.
    heatsinks = [h for h in heatsink_list if h.get_name() == heatsink_name]
    if(heatsinks is None):
        raise Exception("Heatsink not found")
    
    # excluded_types = ["interposer", "substrate", "PCB", "Power_Source"]
    excluded_types = []
    box_list_min = [box for box in box_list if box.chiplet_parent.get_chiplet_type() not in excluded_types]
    x_min = min([box.start_x for box in box_list_min])
    x_max = max([box.end_x for box in box_list_min])
    y_min = min([box.start_y for box in box_list_min])
    y_max = max([box.end_y for box in box_list_min])
        
    if(area_scale_factor != 0):
        dimension_scale_factor = math.sqrt(area_scale_factor)
        dx = dimension_scale_factor * (x_max - x_min)
        dy = dimension_scale_factor * (y_max - y_min)
    elif(scale_factor_x == 0):
        if(scale_factor_y == 0):
            dy = heatsinks[0].get_base_length()
        else:
            dy = scale_factor_y * (y_max- y_min)
        dx = heatsinks[0].get_base_width()
    elif(scale_factor_y == 0):
        dy = heatsinks[0].get_base_length()
        dx = scale_factor_x * (x_max - x_min)
    else:
        dx = scale_factor_x * (x_max - x_min)
        dy = scale_factor_y * (y_max- y_min)

    x = (x_max + x_min - dx) / 2
    y = (y_max + y_min - dy) / 2
    z_min = max([box.end_z for box in box_list_min])
    # print(str([box.end_z for box in box_list_min]))
    # print(str([box.name for box in box_list_min if box.end_z == z_min]))
    # print("z_min is : " + str(z_min))
    z = z_min + min_TIM_height
    # print("z is : " + str(z))
    # x = 0
    # dedeepyo : 31-Oct-2024 : Implementing checker for all top chiplets touching heat_sink

    # dedeepyo : 7-Feb-2025 : Heatsink object creation.
    fin_height = heatsinks[0].get_fin_height()
    fin_thickness = heatsinks[0].get_fin_thickness()
    material = heatsinks[0].get_material()
    fin_number = heatsinks[0].get_fin_count()
    hc = heatsinks[0].get_hc()
    fin_offset = heatsinks[0].get_fin_offset()
    dz = heatsinks[0].get_base_thickness()
    bind_to_ambient = heatsinks[0].get_bind_to_ambient()

    if(fin_number == 0):
        if(fin_thickness > 0):
            fin_number = int(dx / (2 * fin_thickness))

    heatsink_data = {
        "name": f"HS_top",
        "index": 0, # 10000000, # 
        # "material": self.return_material_table_id(material),  # Cu-Foil
        "material": material,  # Cu-Foil
        "x": str(round(x, 6)),
        "y": str(round(y, 6)),
        "base_dx": str(round(dx, 6)),
        "base_dy": str(round(dy, 6)),
        "z": str(round(z, 6)),
        "base_dz": str(round(dz, 6)),
        "fin_height": str(fin_height),
        "fin_thickness": str(fin_thickness),
        "fin_count": str(fin_number),
        "fin_axis": "Y",
        "hc": str(hc),
        "bound": bind_to_ambient
    }
    return heatsink_data
# dedeepyo : 09-Apr-2025 #

# dedeepyo : 03-Dec-2025 : Implementing DFS.
def find_deepest_node(chiplet_tree):
    if not chiplet_tree or len(chiplet_tree) == 0:
        return None
    
    root = chiplet_tree[0]
    deepest_node = root
    max_depth = 0
    
    def traverse(node, depth):
        nonlocal deepest_node, max_depth
        if depth > max_depth:
            max_depth = depth
            deepest_node = node
        
        for child in node.get_child_chiplets():
            traverse(child, depth + 1)
    
    traverse(root, 0)
    return deepest_node
# dedeepyo : 03-Dec-2025

@click.command("standalone")
@click.option('--therm_conf', help='The thermal config file')
# @click.option('--deep_conf', help='The deepflow config file')
@click.option('--out_dir', help='The output directory')
# @click.option('--simtype', default="Anemoi", help='The simulator type')
@click.option('--heatsink_conf', help='The heatsink config file')
@click.option('--bonding_conf', help='The bonding config file')
@click.option('--heatsink', help='The heatsink name')
# @click.option('--bonding', help='The bonding name')
@click.option('--project_name', help='The project name')
@click.option('--is_repeat', default = False, help='Is this a repeat run?')
@click.option('--hbm_stack_height', default = 1, help='Number of dies per HBM stack')
@click.option('--system_type', default = "2p5D", help='The system type')
@click.option('--dummy_si', default = False, help='Does 3D have dummy Si?')
@click.option('--tim_cond_list', default = [10.0], multiple = True, help='The TIM conductivity list')
@click.option('--infill_cond_list', default = [1.6], multiple = True, help='The infill conductivity list')
@click.option('--underfill_cond_list', default = [1.6], multiple = True, help='The underfill conductivity list')
def therm(therm_conf, heatsink_conf, bonding_conf, heatsink, out_dir, project_name, simtype = "Anemoi", is_repeat = False, hbm_stack_height = 1, system_type = "2p5D", dummy_si = False, tim_cond_list = (5, 10, 50), infill_cond_list = (1.6, 19), underfill_cond_list = (1.6, 19)):

    chiplet_tree = parse_all_chiplets(therm_conf)
    (w_top, l_top) = recursive_chiplet_sizing(chiplet_tree[0], None)

    # first chip at 0,0
    next_level_assignements = []
    # traverse chiplet tree. Handle each level separately.D

    def generate_placements_from_floorplan(floorplan, floorplan_dict, chiplet_tree):
        # if floorplan or floorplan_dict = "", then check if the chiplet_tree has only 1 element.
        # if it does, then assign it to the center of the floorplan.
        # if it has more, then exception
        if floorplan == "" or floorplan_dict == "":
            if len(chiplet_tree) == 1:
                grid = [[chiplet_tree[0]]]
                return grid, []
            else:
                raise ValueError("Chiplet tree has more than 1 element with no floorplan definitions")
            
        # if floorplan and floorplan_dict are not empty, then we need to parse them.
        # first, parse the floorplan
        # example floorplan (in XML):
        # floorplan="
        #     HXH
        #     XDX
        #     HXH"
        # example floorplan dict:
        # floorplan_dict="
        # D:(GPU)*
        # H:(HBM0,HBM1,HBM2,HBM3)
        # "
        # first, parse the floorplan dict
        # I want a python dict that goes D:GPU, H:(HBM0,HBM1,HBM2,HBM3)
        # because GPU has an * after it, I want to remove it and instead consider the GPU a fixed chiplet.
        fixed_chiplets = []
        # print("Before split", floorplan_dict)
        floorplan_dict = floorplan_dict.split(' ')
        # now eliminate all substrings that are empty whitespace
        floorplan_dict = [x for x in floorplan_dict if x != ""]
        # now split by :
        floorplan_dict = [x.split(':') for x in floorplan_dict]
        # print("After split", floorplan_dict)
        # now split by *
        # handle asterisk
        for i in range(len(floorplan_dict)):
            if floorplan_dict[i][1][-1] == '*':
                floorplan_dict[i][1] = floorplan_dict[i][1][:-1]
                fix_name = floorplan_dict[i][1]
                # remove "(" and ")" from fix name
                fix_name = fix_name.split('(')[1]
                fix_name = fix_name.split(')')[0]
                fixed_chiplets.append(fix_name)
        # floorplan_dict = {x[0]:x[1] for x in floorplan_dict}  
        # we want to do the above line while checking that x[1] is a list
        # if its a string or not a list, basically not in parentheses, then we want to error out
        for i in range(len(floorplan_dict)):
            if floorplan_dict[i][1][0] != '(':
                raise ValueError("Floorplan dict must have values in parentheses")
            floorplan_dict[i][1] = floorplan_dict[i][1][1:-1]
            floorplan_dict[i][1] = floorplan_dict[i][1].split(',')
        floorplan_dict = {x[0]:x[1] for x in floorplan_dict}

        # ensure the values in the dict exist in the chiplet tree
        for key in floorplan_dict:
            # if key is a single string
            if len(floorplan_dict[key]) == 1:
                # if floorplan_dict[key][0] not in chiplet_tree:
                #     print("Chiplet tree:",chiplet_tree)
                #     raise ValueError("Chiplet {} not found in chiplet tree".format(floorplan_dict[key][0]))
                found = False
                for chiplet in chiplet_tree:
                    if chiplet.get_chiplet_type() == floorplan_dict[key][0]:
                        found = True
                        break
                if not found:
                    raise ValueError("Chiplet {} not found in chiplet tree".format(floorplan_dict[key][0]))
            else:
                for chiplet in floorplan_dict[key]:
                    # if chiplet not in chiplet_tree:
                    #     raise ValueError("Chiplet {} not found in chiplet tree".format(chiplet))
                    for chiplet_tree_chiplet in chiplet_tree:
                        if chiplet_tree_chiplet.get_chiplet_type() == chiplet:
                            found = True
                            break
                    if not found:
                        raise ValueError("Chiplet {} not found in chiplet tree".format(chiplet))


    
        # now parse the floorplan
        # print("Pre split", floorplan)
        # floorplan = floorplan.split('\n')
        # floorplan = [x.strip() for x in floorplan] # dedeepyo : 27-Feb-25
        # floorplan = [x for x in floorplan if x != ""] # dedeepyo : 27-Feb-25
        # floorplan = [list(x) for x in floorplan] # dedeepyo : 27-Feb-25
        # print("After processing", floorplan)
        # floorplan is now a flat list. We need to ensure it is a square.
        # floorpl_len = len(floorplan) # dedeepyo : 27-Feb-25
        # check that it is a square number
        # like literally. Is it 4,9,16,..
        # if math.sqrt(floorpl_len) != int(math.sqrt(floorpl_len)): # dedeepyo : 27-Feb-25
        #     raise ValueError("Floorplan must be a square") # dedeepyo : 27-Feb-25
        # now format it into a square
        # floorplan = [floorplan[i:i+int(math.sqrt(floorpl_len))] for i in range(0,floorpl_len,int(math.sqrt(floorpl_len)))] # dedeepyo : 27-Feb-25

        # dedeepyo : 27-Feb-25 : Fixing the floorplan parsing. Adding rectangular floorplanning.
        floorplan = floorplan.strip("\n").strip().split(' ')
        floorplan = [x for x in floorplan if x != ""]
        floorplan = [list(x) for x in floorplan]
        r = len(floorplan)
        for i in range(r):
            floorplan[i] = [list(x) for x in floorplan[i]]
        # print("Floorplan:",floorplan)
        # print(len(floorplan),len(floorplan[0]))
        # dedeepyo : 27-Feb-25 #
        # # ensure floorplan is an odd square.
        # if len(floorplan) % 2 == 0 or len(floorplan[0]) % 2 == 0:
        #     raise ValueError("Floorplan must be an odd square")
            

        # create a grid with the same dimensions as the floorplan
        grid = [[None for i in range(len(floorplan[0]))] for j in range(len(floorplan))]
        # now, for each element in the floorplan, assign the corresponding chiplet
        # print floorplan and grid shapes
        # print("Floorplan shape:",len(floorplan),len(floorplan[0]))
        # print("Grid shape:",len(grid),len(grid[0]))
        # print("Floorplan:",floorplan)
        # print("Floorplan dict:",floorplan_dict)
        # print(grid)
        for i in range(len(floorplan)):
            for j in range(len(floorplan[0])):
                if floorplan[i][j][0] == 'X':
                    grid[i][j] = None
                else:
                    # print("Checking",floorplan[i][j][0])
                    # try to find an appropriate assignment within the floorplan dict
                    if floorplan[i][j][0] not in floorplan_dict:
                        raise ValueError("Chiplet {} not found in floorplan dict".format(floorplan[i][j][0]))
                    # else, we need to find the first chiplet in the chiplet tree that matches the floorplan dict
                    chosen_chiplet = None
                    for chiplet in chiplet_tree:
                        # print("Checking",chiplet.get_chiplet_type())
                        # print("In",floorplan_dict[floorplan[i][j][0]])
                        if chiplet.get_chiplet_type() in floorplan_dict[floorplan[i][j][0]] and not chiplet.is_assigned_floorplan():
                            chosen_chiplet = chiplet
                            chiplet.set_assigned_floorplan(True)
                            break

                    if chosen_chiplet is None:
                        raise ValueError("Chiplet defined by {} not found in chiplet tree".format(floorplan[i][j][0]))
                    grid[i][j] = chosen_chiplet

                    # grid[i][j] = floorplan_dict[floorplan[i][j][0]][0]
                    # print("Assigned",grid[i][j])

        return grid, fixed_chiplets

    # dedeepyo : 17-Dec-2024 : Implementing copy of placement recursion.
    def copy_placements_recursive(boxes,parent,chiplet_tree, min_dist = 0.0):
        return
    # dedeepyo : 17-Dec-2024

    # dedeepyo : 28-Jan-2025 : Implementing recursive fake chiplet sizing.
    # Assuming any level has either all fake chiplets or all real chiplets.
    # Assuming the base chiplet is never fake.
    def determine_sizing_recursive(boxes,parent,chiplet_tree, min_dist = 0.0, fake_chiplet_size_dict = {}):
        if len(chiplet_tree) == 0:
            return
        
        if not parent:
            grid = [[chiplet_tree[0]]]
            x_coord = 0
            y_coord = 0
            z_coord = 0
            stackup = chiplet_tree[0].get_stackup()
            material = "Si"
            power = chiplet_tree[0].get_power()
            area = chiplet_tree[0].get_core_area()
            aspect_ratio = chiplet_tree[0].get_aspect_ratio()
            height = chiplet_tree[0].get_height()
            width = math.sqrt(area*aspect_ratio)
            length = area/width
            box = Box(x_coord,y_coord,z_coord,width,length,height,power,stackup,0,chiplet_tree[0].get_name())
            box.assign_chiplet_parent(chiplet_tree[0])
            boxes.append(box)
            determine_sizing_recursive(boxes,box,chiplet_tree[0].get_child_chiplets(), box.chiplet_parent.assembly_process.get_die_separation(), fake_chiplet_size_dict)
        else:
            grid, fixed_chiplets = generate_placements_from_floorplan(parent.chiplet_parent.floorplan,parent.chiplet_parent.floorplan_dict,chiplet_tree)
            parent_coords = parent.get_2d_coords()
            parent_width = parent.width
            parent_length = parent.length
            num_rows = len(grid)
            num_cols = len(grid[0])
            # Assuming fake chiplets are only iterated over once.
            fake_chiplet_dict = {}
            largest_width = 0
            largest_length = 0
            for i in range(num_rows):
                for j in range(num_cols):
                    if grid[i][j] is not None:
                        chiplet = grid[i][j]
                        chiplet_area = chiplet.get_core_area()
                        chiplet_aspect_ratio = chiplet.get_aspect_ratio()
                        chiplet_width = math.sqrt(chiplet_area*chiplet_aspect_ratio)
                        chiplet_length = chiplet_area/chiplet_width
                        # dedeepyo : 28-Jan-2025 : Implementing fake chiplet addition to dictionary.
                        chiplet.set_assigned_floorplan(False)
                        if((chiplet.get_fake() == True) and (fake_chiplet_dict.get(chiplet.get_chiplet_type()) == None)):
                            fake_chiplet_dict[chiplet.get_chiplet_type()] = 0
                        # dedeepyo : 28-Jan-2025 #
                        if chiplet_width > largest_width:
                            largest_width = chiplet_width
                        if chiplet_length > largest_length:
                            largest_length = chiplet_length

            side_length = largest_length
            side_length += min_dist #
            # side_length += min_dist * (1 + INFLATION)
            side_width = largest_width
            side_width += min_dist #
            # side_width += min_dist * (1 + INFLATION)
            grid_length = side_length*num_rows
            grid_width = side_width*num_cols
            grid_x = parent_coords[0] + parent_width/2 - grid_width/2
            grid_y = parent_coords[1] + parent_length/2 - grid_length/2

            children_boxes = []         
            for i in range(num_rows):
                for j in range(num_cols):
                    if grid[i][j] is not None:
                        chiplet = grid[i][j]
                        if(chiplet.get_fake() == True):
                            if(fake_chiplet_dict[chiplet.get_chiplet_type()] == 1):
                                continue
                        x_coord = grid_x + j*side_length + side_length/2
                        y_coord = grid_y + i*side_length + side_length/2
                        z_coord = parent.start_z + parent.height
                        stackup = chiplet.get_stackup()
                        material = "Si"
                        power = chiplet.get_power()
                        aspect_ratio = chiplet.get_aspect_ratio()
                        height = chiplet.get_height()
                        area = chiplet.get_core_area()
                        width = math.sqrt(area*aspect_ratio)
                        length = area/width
                        
                        x_coord = x_coord - width/2
                        y_coord = y_coord - length/2                        
                        box = Box(x_coord,y_coord,z_coord,width,length,height,power,stackup,0,chiplet.get_name())
                        box.assign_chiplet_parent(chiplet)
                        chiplet.set_box_representation(box)
                        min_dist_for_children = 0.0
                        if(chiplet.get_fake() == True):
                            min_dist_for_children = parent.chiplet_parent.assembly_process.get_die_separation()
                            fake_chiplet_dict[chiplet.get_chiplet_type()] = 1
                        else:
                            min_dist_for_children = chiplet.assembly_process.get_die_separation()
                            boxes.append(box)                        
                        children_boxes.append(box)
                        determine_sizing_recursive(boxes,box,chiplet.get_child_chiplets(), min_dist_for_children, fake_chiplet_size_dict)

            def move_box(box, x, y):
                box.start_x += x #
                box.start_y += y #
                for chiplet in box.chiplet_parent.get_child_chiplets(): #
                    if chiplet.get_box_representation(): #
                        move_box(chiplet.get_box_representation(), x, y) #

            # parent_center = parent.get_2d_center()
            box_destination_pairs = []
            for box in children_boxes:
                if box.chiplet_parent.get_chiplet_type() in fixed_chiplets:
                    continue
                curr_conn = box.chiplet_parent.connections
                coordinates = box.get_2d_center()
                connection_coords = []
                for conn in curr_conn:
                    conn_type = conn.split(".")[-1].split("#")[0]
                    # conn_type = ''.join([i for i in conn_type if not i.isdigit()])
                    # conn_type = conn_type.split("#")[0]
                    # print("Connection type:", conn_type)
                    if conn_type in fixed_chiplets:
                        for fixed_box in boxes:
                            if fixed_box.name == conn:
                                connection_coords.append(fixed_box.get_2d_center())
                avg_x = 0
                avg_y = 0
                for coord in connection_coords:
                    avg_x += coord[0]
                    avg_y += coord[1]
                if len(connection_coords) == 0:
                    continue
                else:
                    avg_x = avg_x / len(connection_coords)
                    avg_y = avg_y / len(connection_coords)
                box_destination_pairs.append((box,(avg_x,avg_y)))

            # f = open("box_movement_log.txt", "a")
            i = 0
            old_boxes = box_destination_pairs.copy()
            N = 1010
            while (True):
                i += 1
                if i > N:
                    break
                if i % 10 == 0:
                    # TODO: Fix this low priority
                    same = True
                    for j in range(len(box_destination_pairs)):
                        new_box = box_destination_pairs[j]
                        old_box = old_boxes[j]
                        def is_equal(box1, box2):
                            bound = min_dist
                            return abs(box1[0] - box2[0]) < bound and abs(box1[1] - box2[1]) < bound

                        if not is_equal(new_box[1], old_box[1]):
                            same = False
                            break
                    if not same:
                        break
                old_boxes = box_destination_pairs.copy()

                for box, coordinates in box_destination_pairs:
                    curr_x, curr_y = box.get_2d_center()
                    dist_x = curr_x - coordinates[0]
                    dist_y = curr_y - coordinates[1]
                    multiplier_x = 1
                    multiplier_y = 1
                    if abs(dist_x) > abs(dist_y):
                        # multiplier_x = 1
                        multiplier_y = abs(dist_y) / abs(dist_x)
                    elif abs(dist_x) < abs(dist_y):
                        multiplier_x = abs(dist_x) / abs(dist_y)
                    #     multiplier_y = 1
                    # else:   
                    #     multiplier_x = 1
                    #     multiplier_y = 1

                    delta_x = -min_dist*multiplier_x if dist_x > 0 else min_dist*multiplier_x
                    delta_y = -min_dist*multiplier_y if dist_y > 0 else min_dist*multiplier_y
                    if abs(dist_x) < min_dist:
                        delta_x = 0
                    if abs(dist_y) < min_dist:
                        delta_y = 0
                    
                    # line = "Moving box " + box.name + " from (" + str(curr_x) + "," + str(curr_y) + ") to (" + str(coordinates[0]) + "," + str(coordinates[1]) + " by dist_x of " + str(dist_x) + "and dist_y of " + str(dist_y) + ") with deltas (" + str(delta_x) + "," + str(delta_y) + ")\n"
                    # f.write(line)
                    # f.close()

                    overlap_count = check_all_overlaps_3d(children_boxes, box, min_dist*INFLATION)
                    move_box(box,0,delta_y)
                    overlap_count_new = check_all_overlaps_3d(children_boxes, box, min_dist*INFLATION)
                    if overlap_count_new > overlap_count:
                        move_box(box,0,-delta_y)
                    
                    # f.write("overlap_count of " + str(box.name) + " is " + str(overlap_count) + "\n")
                    # f.write("overlap_count_new of " + str(box.name) + " after y-movement and before x-movement is " + str(overlap_count_new) + "\n")
                    move_box(box,delta_x,0)
                    overlap_count_new = check_all_overlaps_3d(children_boxes, box, min_dist*INFLATION)
                    if overlap_count_new > overlap_count:
                        move_box(box,-delta_x,0)
                    # f.write("overlap_count_new of " + str(box.name) + " after y-movement and after x-movement is " + str(overlap_count_new) + "\n")

            # f.close()
            if(parent.chiplet_parent.get_fake() == True):
                fakes = [c for c in children_boxes if c.chiplet_parent.get_fake() == True]
                if(len(fakes) == 0):
                    min_x = min([c.start_x for c in children_boxes])
                    max_x = max([c.end_x for c in children_boxes])
                    min_y = min([c.start_y for c in children_boxes])
                    max_y = max([c.end_y for c in children_boxes])
                    # parent.start_x = min_x
                    # parent.start_x -= min_dist * INFLATION
                    # parent.start_y = min_y
                    # parent.start_y -= min_dist * INFLATION
                    parent.width = max_x - min_x
                    parent.length = max_y - min_y
                    parent.width += 2 * min_dist * INFLATION
                    parent.length += 2 * min_dist * INFLATION
                    # for cbox in children_boxes:
                        # print("Child box:", cbox.name, "Width:", cbox.width, "Length:", cbox.length, "Start X:", cbox.start_x, "Start Y:", cbox.start_y, "End X:", cbox.end_x, "End Y:", cbox.end_y)
                elif(len(fakes) == len(children_boxes)):
                    # Assuming all fakes are of same type.
                    c  = fakes[0]
                    parent.width = num_cols * c.width 
                    parent.length = num_rows * c.length
                    parent.width += 2 * min_dist * INFLATION
                    parent.length += 2 * min_dist * INFLATION
                else:
                    # TODO : Handle this case
                    print("#TODO : Handle this case")
                
                parent.chiplet_parent.set_core_area(parent.width * parent.length)
                parent.chiplet_parent.set_aspect_ratio(parent.width / parent.length)
                fake_chiplet_size_dict[parent.chiplet_parent.get_chiplet_type()] = (parent.chiplet_parent.get_core_area(), parent.chiplet_parent.get_aspect_ratio())
    # dedeepyo : 28-Jan-2025 #

    def determine_placements_recursive(boxes,parent,chiplet_tree, min_dist = 0.0):
        # parent argument is a Box() class argument that references the original Chiplet() class.
        # if chiplet_tree is empty, return
        if len(chiplet_tree) == 0:
            return

        # dedeepyo : 18-Nov-2024 : This is the recursive function that will determine the placements of the chiplets.
        # We will start with the root chiplet and then recursively determine the placements of the child chiplets.
        # We will use a grid based approach to determine the placements of the chiplets.
        # The grid will be projected onto the parent chiplet.
        # The grid will have a side length of the largest chiplet in the grid.
        # The grid will be centered on the center of the parent chiplet.
        # The grid will have a minimum distance between chiplets based on the assembly process.
        # We will also handle overlaps by moving the chiplets towards the center of mass of their connections.
        # We will also handle fixed chiplets by not moving them at all.
        # We assume the first (or root) chiplet in the XML chiplet tree is not a fake chiplet.
        # We assume the definition of any referenced xml tag (eg. interposer, GPU, HBM, etc.) is present within the definition of the XML chiplet tree which references it.
        # We assume the XML chiplet tree is a tree and not a graph. This means there are no cycles in the XML chiplet tree.
        # We assume the XML chiplet tree is a valid tree. This means there is only one root chiplet and all other chiplets have exactly one parent.
        # We assume the XML chiplet tree is a valid tree. This means there are no disconnected components in the XML chiplet tree.
        # We assume the XML chiplet tree is a valid tree. This means there are no two chiplets in the chiplet XML tree with the exact same parameters. 
        # If there are duplicate definitions for the same chiplet, they must differ in atleast one parameter.
        # We assume "set" of chiplets has an area.
        # We assume the first chiplet at the base of all the chiplets is not a "set".
        # We assume the fake chiplet has an empty stackup string.
        # dedeepyo : 18-Nov-2024

        # this is only the case for the root chiplet (interposer)
        if not parent:
            # print("Parse start")
            # we are starting. Assume coordinates are 0,0
            # just assign the first chiplet to the center
            # this is interposer or PCB so we don't need to get fancy at all.
            grid = [[chiplet_tree[0]]]
            x_coord = 0
            y_coord = 0
            z_coord = 0
            # we need to make a box
            # we dont know exact material. Do a rough guess based on stackup.
            stackup = chiplet_tree[0].get_stackup()
            # stackup example:
            #         stackup="1:5nm_active,2:5nm_advanced_metal,2:5nm_intermediate_metal,2:5nm_global_metal"
            # actually just assume Si.
            material = "Si"
            power = chiplet_tree[0].get_power()
            area = chiplet_tree[0].get_core_area()
            aspect_ratio = chiplet_tree[0].get_aspect_ratio()
            height = chiplet_tree[0].get_height()
            width = math.sqrt(area*aspect_ratio)
            length = area/width
            box = Box(x_coord,y_coord,z_coord,width,length,height,power,stackup,0,chiplet_tree[0].get_name())
            box.assign_chiplet_parent(chiplet_tree[0])
            boxes.append(box)
            determine_placements_recursive(boxes,box,chiplet_tree[0].get_child_chiplets(), box.chiplet_parent.assembly_process.get_die_separation())
        else:
            grid, fixed_chiplets = generate_placements_from_floorplan(parent.chiplet_parent.floorplan,parent.chiplet_parent.floorplan_dict,chiplet_tree)
            parent_coords = parent.get_2d_coords()
            # now, we need to determine the grid based on the parent chiplet
            # the grid is projected onto the parent chiplet
            parent_width = parent.width
            parent_length = parent.length

            # how many rows/columns in grid?
            num_rows = len(grid)
            # num_cols = num_rows
            num_cols = len(grid[0])

            # the grid coordinate box size will be based on the largest chiplet dimensions in the grid
            # we need to find the largest width and largest length separately. The larger one of the two will define side length
            largest_width = 0
            largest_length = 0
            for i in range(num_rows):
                for j in range(num_cols):
                    if grid[i][j] is not None:
                        chiplet = grid[i][j]
                        chiplet_area = chiplet.get_core_area()
                        chiplet_aspect_ratio = chiplet.get_aspect_ratio()
                        chiplet_width = math.sqrt(chiplet_area * chiplet_aspect_ratio)
                        chiplet_length = chiplet_area/chiplet_width
                        if chiplet_width > largest_width:
                            largest_width = chiplet_width
                        if chiplet_length > largest_length:
                            largest_length = chiplet_length

            # now, we need to determine the side length of the grid box
            # we will use the larger of the two
            side_length = largest_length
            # side_length += min_dist * (1 + INFLATION) #
            side_length += min_dist #
            # dedeepyo : 29-Jan-2025 : Sizing length and width of grid separately.
            side_width = largest_width
            side_width += min_dist #
            # side_width += min_dist * (1 + INFLATION)
            # dedeepyo : 29-Jan-2025 #
            # print("Side length:",side_length)
            # now, we need to determine the x and y coordinates of the grid box
            # the grid box will be centered on the parent chiplet
            # the grid box will have a side length of side_length
            grid_length = side_length*num_rows
            grid_width = side_width*num_cols
            # the grid will be centered on the center of the parent chiplet
            grid_x = parent_coords[0] + parent_width/2 - grid_width/2
            grid_y = parent_coords[1] + parent_length/2 - grid_length/2
            # now, we need to assign coordinates to each chiplet in the grid
            # we assign them to the center of the grid subbox
            # print("Parent:",parent.name)
            # print("Grid:",grid)
            children_boxes = []         
            for i in range(num_rows):
                for j in range(num_cols):
                    if grid[i][j] is not None:
                        chiplet = grid[i][j]
                        x_coord = grid_x + j*side_width + side_width/2
                        y_coord = grid_y + i*side_length + side_length/2
                        # parent z + parent height
                        z_coord = parent.start_z + parent.height

                        # we dont know exact material. Do a rough guess based on stackup.
                        stackup = chiplet.get_stackup()
                        # stackup example:
                        #         stackup="1:5nm_active,2:5nm_advanced_metal,2:5nm_intermediate_metal,2:5nm_global_metal"
                        # actually just assume Si.
                        material = "Si"
                        power = chiplet.get_power()
                        aspect_ratio = chiplet.get_aspect_ratio()
                        height = chiplet.get_height()
                        area = chiplet.get_core_area()
                        width = math.sqrt(area*aspect_ratio)
                        length = area/width
                        
                        # we need to adjust the box so that it is centered on the grid point.
                        x_coord = x_coord - width/2
                        y_coord = y_coord - length/2                        
                        box = Box(x_coord,y_coord,z_coord,width,length,height,power,stackup,0,chiplet.get_name())
                        box.assign_chiplet_parent(chiplet)
                        chiplet.set_box_representation(box)

                        # dedeepyo : 18-Nov-2024 : Implementing checker and side calculation for fake chiplet.
                        # If the chiplet is a fake chiplet, we need to remove it from the list of boxes.
                        # We update the min_dist for the current chiplet and pass it for its children sitting on top of it unless the current one is a fake chiplet.
                        min_dist_for_children = 0.0
                        if(chiplet.get_fake() == True):
                            # print("Fake chiplet found: ",chiplet.get_name())
                            min_dist_for_children = parent.chiplet_parent.assembly_process.get_die_separation()
                        else:
                            min_dist_for_children = chiplet.assembly_process.get_die_separation()
                            boxes.append(box)
                        # dedeepyo : 18-Nov-2024
                        # print("Created box: ",box.name)
                        children_boxes.append(box)
                        determine_placements_recursive(boxes,box,chiplet.get_child_chiplets(), min_dist_for_children)

            # dedeepyo : 17-Dec-2024 : Implementing copy of placement recursion.
            # determined_placements = {}
            # if(determined_placements.get(grid[i][j].get_chiplet_type()) == None):
            #     determined_placements[grid[i][j].get_chiplet_type()] = box
            # else:
            #     # print("Copying placements for: ",box.name)
            #     copy_placements_recursive(boxes,box,chiplet.get_child_chiplets(), min_dist_for_children, determined_placements[grid[i][j].get_chiplet_type()])
                # print("Copied placements for: ",box.name)
            # dedeepyo : 17-Dec-2024
                    
            def move_box(box, x, y):
                box.start_x += x #
                box.start_y += y #
                # box.end_x += x
                # box.end_y += y
                for chiplet in box.chiplet_parent.get_child_chiplets(): #
                    if chiplet.get_box_representation(): #
                        move_box(chiplet.get_box_representation(), x, y) #

            # dedeepyo : 17-Dec-2024 : Implementing copying placements.            
            # if(reference_box != None):
            #     del_x = reference_box.start_x - parent.start_x
            #     del_y = reference_box.start_y - parent.start_y
            #     move_box(parent, del_x, del_y)
            #     return
            # dedeepyo : 17-Dec-2024
            
            # shrinking step
            # print("FIX:",fixed_chiplets)
            parent_center = parent.get_2d_center()
            box_destination_pairs = []
            for box in children_boxes:
                # print(box.name,":",box.chiplet_parent.connections)
                if box.chiplet_parent.get_chiplet_type() in fixed_chiplets:
                    continue
                curr_conn = box.chiplet_parent.connections
                coordinates = box.get_2d_center()
                # This list is a list of coordinates of boxes that this current box should move towards.
                connection_coords = []
                for conn in curr_conn:
                    conn_type = conn.split(".")[-1]
                    # conn_type = ''.join([i for i in conn_type if not i.isdigit()])
                    conn_type = conn_type.split("#")[0]
                    if conn_type in fixed_chiplets:
                        for fixed_box in boxes:
                            if fixed_box.name == conn:
                                connection_coords.append(fixed_box.get_2d_center())
                # average out to get the "center of mass"
                # print("Box:",box.name,"Connections:",connection_coords)
                avg_x = 0
                avg_y = 0
                for coord in connection_coords:
                    avg_x += coord[0]
                    avg_y += coord[1]
                if len(connection_coords) == 0:
                    # dedeepyo : 26-Jan-2025 : Implementing movement of fake chiplets towards each other to reduce space wastage till they abut (distance between neighbours is more than minimum die separation).
                    # Spreading out real chiplets away from each other till they do not overlap.
                    # Before computing the dimensions of the parent box of the parent chiplet, we need to move the current chiplets, which may be fake, closer to each other.
                    # Assuming either current chiplets are not fake chiplets or all are fake chiplets and none of the grid[i][j] are None or fixed. Otherwise, we have area wastage.
                    # if(box.chiplet_parent.get_fake() == True):
                    #     avg_x = parent_center[0]
                    #     avg_y = parent_center[1]
                    # dedeepyo : 26-Jan-2025 #
                    # else:
                    #     continue
                    continue
                else:
                    avg_x = avg_x / len(connection_coords)
                    avg_y = avg_y / len(connection_coords)
                box_destination_pairs.append((box,(avg_x,avg_y)))

            # f = open("box_placement_movement_log.txt", "a")
            # now, we need to greedily move the box towards the center of mass while respecting overlaps
            i = 0
            old_boxes = box_destination_pairs.copy()
            N = 1010
            # dedeepyo : 27-Jan-2025 : Assigning a higher number of steps for fake chiplets.
            # Assuming either all chiplets in the current set are fake chiplets or none of them are fake chiplets. Else, wastage of iterations.
            # if(children_boxes[0].chiplet_parent.get_fake() == True):
            #     N = 1500
            # dedeepyo : 27-Jan-2025 #
            while (True):
                i += 1
                # number of steps
                if i > N:
                    break
                # THIS EARLY EXIT STUFF DOES NOT WORK
                if i % 10 == 0:
                    # TODO: Fix this low priority
                    same = True
                    for j in range(len(box_destination_pairs)):
                        new_box = box_destination_pairs[j]
                        old_box = old_boxes[j]
                        def is_equal(box1, box2):
                            bound = min_dist
                            return abs(box1[0] - box2[0]) < bound and abs(box1[1] - box2[1]) < bound

                        if not is_equal(new_box[1], old_box[1]):
                            same = False
                            break
                    if not same:
                        # print("early exit")
                        break
                old_boxes = box_destination_pairs.copy()

                for box, coordinates in box_destination_pairs:
                    curr_x, curr_y = box.get_2d_center()
                    dist_x = curr_x - coordinates[0]
                    dist_y = curr_y - coordinates[1]
                    # attempt to greedily reduce dist x and dist y by the min dist
                    # we only want to move the box min_dist at a time
                    # multiplier = 1

                    multiplier_x = 1
                    multiplier_y = 1

                    # dedeepyo : 18-Feb-25 : Implementing a more efficient way to move the boxes, to ensure dist_x / dist_y remains constant.
                    if abs(dist_x) > abs(dist_y):
                    #     multiplier_x = 1
                        multiplier_y = abs(dist_y) / abs(dist_x)
                    elif abs(dist_x) < abs(dist_y):
                        multiplier_x = abs(dist_x) / abs(dist_y)
                    #     multiplier_y = 1
                    # else:   
                    #     multiplier_x = 1
                    #     multiplier_y = 1
                    # dedeepyo : 18-Feb-25 #

                    delta_x = -min_dist*multiplier_x if dist_x > 0 else min_dist*multiplier_x
                    delta_y = -min_dist*multiplier_y if dist_y > 0 else min_dist*multiplier_y
                    if abs(dist_x) < min_dist:
                        delta_x = 0
                    if abs(dist_y) < min_dist:
                        delta_y = 0
                    
                    # dedeepyo : 18-Dec-2024 : Implementing overlap check with only current box.
                    # overlap_count = check_all_overlaps_3d(boxes, box, min_dist*INFLATION)
                    # move_box(box,0,delta_y)
                    # overlap_count_new = check_all_overlaps_3d(boxes, box, min_dist*INFLATION)
                    # if overlap_count_new > overlap_count:
                    #     move_box(box,0,-delta_y)
                    # move_box(box,delta_x,0)
                    # overlap_count_new = check_all_overlaps_3d(boxes, box, min_dist*INFLATION)
                    # if overlap_count_new > overlap_count:
                    #     move_box(box,-delta_x,0)
                    # dedeepyo : 27-Jan-2025 #

                    overlaps = check_all_overlaps_3d(children_boxes, box, min_dist*INFLATION)
                    move_box(box,0,delta_y)
                    overlaps_new = check_all_overlaps_3d(children_boxes, box, min_dist*INFLATION)
                    if overlaps_new > overlaps:
                        move_box(box,0,-delta_y)

                    # f.write("overlap_count of " + str(box.name) + " is " + str(overlaps) + "\n")
                    # f.write("overlap_count_new of " + str(box.name) + " after y-movement and before x-movement is " + str(overlaps_new) + "\n")
                    move_box(box,delta_x,0)
                    overlaps_new = check_all_overlaps_3d(children_boxes, box, min_dist*INFLATION)
                    if overlaps_new > overlaps:
                        move_box(box,-delta_x,0)

                    # f.write("overlap_count_new of " + str(box.name) + " after y-movement and before x-movement is " + str(overlaps_new) + "\n")
                    
            # dedeepyo : 10-Oct-25 : Implementing dummy Si. #TODO: Comment for 2.5D or anything other than 3D. Uncommen`t for 3D.`
            if(system_type == "3D_1GPU" or system_type == "3D_waferscale" or system_type == "3D_1GPU_top"):
                dummy_Si_height = 0.63 # in mm # 0.62 for 8-high, 0.63 for 16-high.
                min_x = min([c.start_x for c in children_boxes]) - min_dist
                max_x = max([c.end_x for c in children_boxes]) + min_dist
                min_y = min([c.start_y for c in children_boxes]) - min_dist
                max_y = max([c.end_y for c in children_boxes]) + min_dist
                if(parent.start_x < min_x):
                    if(parent.start_y < min_y):
                        if(parent.end_x > max_x):
                            if(parent.end_y > max_y):
                                box = Box(parent.start_x, parent.start_y, parent.end_z, min_x - parent.start_x, parent.length, dummy_Si_height, 0.0, "1:dummySi_HBM", 0, parent.name + ".Dummy_Si_above_1")
                                chiplet = Chiplet(name = parent.name + ".Dummy_Si_above_1", core_area = box.width * box.length, aspect_ratio = box.width / box.length, assembly_process = parent.chiplet_parent.get_assembly_process(), stackup = "1:dummySi_HBM", height = dummy_Si_height)
                                r = parent.chiplet_parent.add_child_chiplet(chiplet)
                                boxes.append(box)
                                box.assign_chiplet_parent(chiplet)
                                chiplet.set_box_representation(box)
                                
                                box = Box(max_x, parent.start_y, parent.end_z, parent.end_x - max_x, parent.length, dummy_Si_height, 0.0, "1:dummySi_HBM", 0, parent.name + ".Dummy_Si_above_2")
                                chiplet = Chiplet(name = parent.name + ".Dummy_Si_above_2", core_area = box.width * box.length, aspect_ratio = box.width / box.length, assembly_process = parent.chiplet_parent.get_assembly_process(), stackup = "1:dummySi_HBM", height = dummy_Si_height)
                                r = parent.chiplet_parent.add_child_chiplet(chiplet)
                                boxes.append(box)
                                box.assign_chiplet_parent(chiplet)
                                chiplet.set_box_representation(box)
                                
                                box = Box(min_x, parent.start_y, parent.end_z, max_x - min_x, min_y - parent.start_y, dummy_Si_height, 0.0, "1:dummySi_HBM", 0, parent.name + ".Dummy_Si_above_3")
                                chiplet = Chiplet(name = parent.name + ".Dummy_Si_above_3", core_area = box.width * box.length, aspect_ratio = box.width / box.length, assembly_process = parent.chiplet_parent.get_assembly_process(), stackup = "1:dummySi_HBM", height = dummy_Si_height)
                                r = parent.chiplet_parent.add_child_chiplet(chiplet)
                                boxes.append(box)
                                box.assign_chiplet_parent(chiplet)
                                chiplet.set_box_representation(box)
                                
                                box = Box(min_x, max_y, parent.end_z, max_x - min_x, parent.end_y - max_y, dummy_Si_height, 0.0, "1:dummySi_HBM", 0, parent.name + ".Dummy_Si_above_4")
                                chiplet = Chiplet(name = parent.name + ".Dummy_Si_above_4", core_area = box.width * box.length, aspect_ratio = box.width / box.length, assembly_process = parent.chiplet_parent.get_assembly_process(), stackup = "1:dummySi_HBM", height = dummy_Si_height)
                                r = parent.chiplet_parent.add_child_chiplet(chiplet)
                                boxes.append(box)
                                box.assign_chiplet_parent(chiplet)
                                chiplet.set_box_representation(box)
                            #TODO: Other cases not handled. 
            # dedeepyo : 10-Oct-25 #

            # f.close()
            # dedeepyo : 26-Jan-2025 : Implementing set sizing based on child chiplet dimensions.
            # Assuming all the minimum possible dimensions of the present set of chiplets are already known / calculated.
            # if(parent.chiplet_parent.get_fake() == True):
            #     min_x = min([c.start_x for c in children_boxes])
            #     max_x = max([c.end_x for c in children_boxes])
            #     min_y = min([c.start_y for c in children_boxes])
            #     max_y = max([c.end_y for c in children_boxes])
            #     parent.start_x = min_x
            #     parent.start_y = min_y
            #     parent.width = max_x - min_x
            #     parent.length = max_y - min_y
            #     parent.chiplet_parent.set_core_area(parent.width * parent.length)
            #     parent.chiplet_parent.set_aspect_ratio(parent.width / parent.length)
            # dedeepyo : 26-Jan-2025

    # dedeepyo : 28-Jan-2025 : Assigning recursive fake chiplet sizing.
    fake_chiplet_size_dict = {}
    start_time = time.time()
    print("Starting sizing at ", start_time)
    boxes_unique = []
    determine_sizing_recursive(boxes_unique, None, chiplet_tree, 0.0, fake_chiplet_size_dict)
    end_time = time.time()
    print("Sizing done at ", end_time)
    print("Time taken: ", end_time - start_time)
    # time.sleep(5)
    # dedeepyo : 29-Jan-2025
    # print(fake_chiplet_size_dict)
    # fake_chiplet_size_dict = {
    #     'set_primary' : (1420.092338, 0.4578712122),
    #     'set_secondary' : (5680.369353, 0.4578712122)
    # }
    # for box in boxes_unique:
    #     print(box.name + " " + str(box.start_z) + " " + str(box.end_z) + " " + str(box.height) + " " + str(box.width) + " " + str(box.length) + " " + str(box.start_x) + " " + str(box.start_y))
    # return #TODO: Comment out later.
    # dedeepyo : 29-Jan-2025 : Sizing fake chiplets.
    recursively_copy_chiplet_sizes(fake_chiplet_size_dict, chiplet_tree[0])
    # dedeepyo : 29-Jan-2025 #

    start_time = time.time()
    print("Starting placement at ", start_time)
    boxes = []
    determine_placements_recursive(boxes, None, chiplet_tree, 0.0)
    end_time = time.time()
    print("Placement done at ", end_time)
    print("Time taken: ", end_time - start_time)

    # dedeepyo : 18-Nov-2024 : Implementing checker for fake chiplet.
    # If the chiplet is a fake chiplet, we need to remove it from the list of boxes.
    # We create and delete a box for the chiplet that is actually a fake chiplet.
    # temp_box_names = [b.name for b in boxes if b.chiplet_parent.get_fake() == True]
    # for box_name in temp_box_names:
    #     for box in boxes:
    #         if box.name == box_name:
    #             if(box.chiplet_parent.get_fake() == True):
    #                 boxes.remove(box)
                    # print("Removed fake chiplet box: ", box.name + " for chiplet: " + box.chiplet_parent.get_name())
    # for box in boxes:
    #     print(box.name + " " + str(box.start_z) + " " + str(box.end_z) + " " + str(box.height) + " " + str(box.width) + " " + str(box.length) + " " + str(box.start_x) + " " + str(box.start_y) + " " + str(box.end_x) + " " + str(box.end_y))
    # dedeepyo : 18-Nov-2024

    # dedeepyo : 29-Jan-2025 : Testing dimensions.
    # print(min([c.start_x for c in boxes if c.name != "interposer" or c.name != "Power_Source"]),max([c.end_x for c in boxes if c.name != "interposer" or c.name != "Power_Source"]))
    # print(min([c.start_y for c in boxes if c.name != "interposer" or c.name != "Power_Source"]),max([c.end_y for c in boxes if c.name != "interposer" or c.name != "Power_Source"]))
    # dedeepyo : 29-Jan-2025 #

    # now, draw everything
    limits = determine_draw_lim(boxes)
    draw_fig(boxes,out_dir,"post",limits)
    draw_fig_3D_zoom(boxes,out_dir,"post",limits)
    print("Placement finished, plots generated")
    # print(str(boxes))
    # return #TODO: Comment out later.

    # for box in boxes:
    #     if(box.chiplet_parent.get_chiplet_type() == "GPU" or box.chiplet_parent.get_chiplet_type() == "HBM_l1"):
    #         print("Created : ", box.name + " at (" + str(box.start_x) + "," + str(box.start_y) + "," + str(box.start_z) + ") with width " + str(box.width) + " and length " + str(box.length) + " and height " + str(box.height))

    # return #TODO: Comment out later.

    layers = parse_Layer_netlist("configs/thermal-configs/layer_definitions.xml")
    heatsink_list = heatsink_definition_list_from_file(heatsink_conf)
    bonding_list = bonding_definition_list_from_file(bonding_conf)
    heatsink_name = heatsink
    bonding_name_type_dict = {"GPU#HBM_l4": "bonding_Cu_pillar", "HBM_l4#GPU": "bonding_Cu_pillar", "GPU#HBM_l12": "bonding_Cu_pillar", "HBM_l12#GPU": "bonding_Cu_pillar", "GPU#HBM_l16": "bonding_Cu_pillar", "HBM_l16#GPU": "bonding_Cu_pillar", "GPU#HBM_l8": "bonding_Cu_pillar", "HBM_l8#GPU": "bonding_Cu_pillar", "GPU#HBM": "bonding_Cu_pillar", "HBM#GPU": "bonding_Cu_pillar", "GPU#interposer": "bonding_Cu_pillar", "interposer#GPU": "bonding_Cu_pillar", "interposer#HBM": "bonding_Cu_pillar", "HBM#interposer": "bonding_Cu_pillar", "interposer#PCB": "bonding_bga_ball", "PCB#interposer": "bonding_bga_ball", "interposer#substrate": "bonding_bga_ball", "substrate#interposer": "bonding_bga_ball", "substrate#Power_Source": "bonding_bga_ball", "Power_Source#substrate": "bonding_bga_ball", "interposer#Power_Source": "bonding_bga_ball", "Power_Source#interposer": "bonding_bga_ball", "GPU#substrate": "bonding_Cu_pillar", "substrate#GPU": "bonding_Cu_pillar", "substrate#HBM": "bonding_Cu_pillar", "HBM#substrate": "bonding_Cu_pillar", "GPU#PCB": "bonding_Cu_pillar", "PCB#GPU": "bonding_Cu_pillar", "PCB#HBM": "bonding_Cu_pillar", "HBM#PCB": "bonding_Cu_pillar", "PCB#Power_Source": "bonding_bga_ball", "Power_Source#PCB": "bonding_bga_ball"}
    # bonding_name = bonding_name_type_dict
    recursively_remove_fake_chiplets(chiplet_tree[0])
    # recursively_find_fakes(chiplet_tree[0])
    is_repeat = is_repeat # False
    min_TIM_height = 0.1 # 0.02 # 0.1 # 0.01, 0.02, 0.05, 0.1
    suffix = ""
    
    # print(str(boxes))
    # print("After creating bonding, TIM and heatsink:")
    # for box in boxes:
    #     print(box.name + " " + str(box.start_z) + " " + str(box.end_z) + " " + str(box.height) + " " + str(box.width) + " " + str(box.length) + " " + str(box.start_x) + " " + str(box.start_y) + " " + str(box.end_x) + " " + str(box.end_y))

    # dedeepyo : 01-Dec-25 : Implementing GPU on top.
    # Box(x_coord,y_coord,z_coord,width,length,height,power,"1:5nm_GPU_active_3D,20:5nm_GPU_metal",0,"Power_Source.substrate.HBM.GPU")
    # dedeepyo : 03-Dec-25 : Implementing GPU on top.
    if(system_type == "3D_1GPU_top"):
        # <chip name="GPU"
            # bb_area="$core_area"
            # bb_cost=""
            # bb_quality=""
            # bb_power=""
            # aspect_ratio="0.787"
            # x_location=""
            # y_location=""
        
            # core_area="0.0"
            # fraction_memory="0.0"
            # fraction_logic="1.0"
            # fraction_analog="0.0"
            # gate_flop_ratio="1.0"
            # reticle_share="1.0"
            # buried="False"
            # assembly_process="silicon_individual_bonding"
            # test_process="KGD_free_test"
            # stackup="1:5nm_GPU_active_3D,20:5nm_GPU_metal"
            # wafer_process="process_1"
            # v_rail="5,1.8"
            # reg_eff="1.0,0.9"
            # reg_type="none,side"
            # core_voltage="1.0"
            # power="$core_power"
            # quantity="1000000"
            # fake="False"

            # floorplan=""
            # floorplan_dict=""></chip>
        deepest_node = find_deepest_node(chiplet_tree)
        deepest_node_box = deepest_node.get_box_representation()
        z_coord = deepest_node_box.start_z + deepest_node_box.height

        GPU_stackup = "1:5nm_GPU_active_3D,20:5nm_GPU_metal"
        height = 0
        stackup_list = GPU_stackup.split(",")
        for stackup in stackup_list:
            layer_num, layer_name = stackup.split(":")
            for layer in layers:
                if layer.get_name() == layer_name:
                    height += (int(layer_num) * layer.get_thickness())

        height = round(height, 3) #CHECK: 3 decimal places. 
        
        GPU_chiplet = Chiplet(name=deepest_node.get_name() + ".GPU", core_area=826.2, aspect_ratio= 0.787, fraction_memory=0.0, fraction_logic=1.0, fraction_analog=0.0, assembly_process="silicon_individual_bonding", stackup=stackup, power=270.0, floorplan="", floorplan_dict="", fake=False, height=height)
        width = math.sqrt(GPU_chiplet.get_core_area() * GPU_chiplet.get_aspect_ratio())
        length = GPU_chiplet.get_core_area() / width
        GPU_box = Box(0.0,0.0,z_coord,width,length,GPU_chiplet.get_height(),GPU_chiplet.get_power(),GPU_chiplet.get_stackup(),0,GPU_chiplet.get_name())
        GPU_box.assign_chiplet_parent(GPU_chiplet)
        GPU_chiplet.set_box_representation(GPU_box)
        boxes.append(GPU_box) #TODO

        deepest_node.add_child_chiplet(GPU_chiplet)
    # dedeepyo : 01-Dec-25 #

    bonding_box_list = create_all_bonding(box_list = boxes, name_type_dict = bonding_name_type_dict, bonding_list = bonding_list) #        
    TIM_boxes = create_TIM_to_heatsink(box_list = boxes, material = "TIM0p5", min_TIM_height = min_TIM_height, system_type = system_type)
    heatsink_obj = create_heat_sink(box_list = boxes, heatsink_list = heatsink_list, heatsink_name = heatsink_name, min_TIM_height = min_TIM_height, scale_factor_x = 0, scale_factor_y = 0, area_scale_factor = 1)
    create_power_source_backside(boxes) #
    power_dict = initialize_power_dict_values(boxes)
    
    # DEBUG: Print actual power values being used
    print("\n===== POWER CONFIGURATION =====")
    print("Power dict:", power_dict)
    for box in boxes:
        if box.chiplet_parent.get_chiplet_type() == "GPU":
            print(f"GPU power from box: {box.power} W")
        elif box.chiplet_parent.get_chiplet_type() == "HBM":
            print(f"HBM power from box: {box.power} W")
    print("================================\n")

    # print("After creating bonding, TIM and heatsink:")
    # for box in boxes:
    #     print(box.name + " " + str(box.start_z) + " " + str(box.end_z) + " " + str(box.height) + " " + str(box.width) + " " + str(box.length) + " " + str(box.start_x) + " " + str(box.start_y) + " " + str(box.end_x) + " " + str(box.end_y))
    
    # print(str(boxes))    
    # return #TODO: Comment out later.

    # dedeepyo : 21-Jun-25 : Implmenting multiple heatsinks. 
    # heatsink_list till now stored definitions of heatsinks. From now on, it will store the actual heatsink objects used with coordinates.
    multiple_heatsinks = False # True if multiple heatsinks are used, False if only one heatsink is used.
    if not multiple_heatsinks:
        heatsink_list = []
    else:
        heatsink_list_new, power_dict_new = create_multiple_heat_sinks(box_list = boxes, heatsink_list = heatsink_list, heatsink_name = heatsink_name, min_TIM_height = min_TIM_height, power_dict = power_dict) # , scale_factor_x = 0, scale_factor_y = 0, area_scale_factor = 1)
        heatsink_list = heatsink_list_new
        power_dict = power_dict_new
    # dedeepyo : 21-Jun-25

    # dedeepyo : 01-Dec-25 #
    # Below is only Anemoi, above is tool-agnostic.
    # dedeepyo : 01-Dec-25 #
    data = {
                'boxes' : boxes,
                'heatsink_list' : heatsink_list,
                'heatsink_name' : heatsink_name,
                'bonding_box_list' : bonding_box_list,
                'heatsink_obj' : heatsink_obj,
                'TIM_boxes' : TIM_boxes,
                'suffix' : suffix,
                'is_repeat' : is_repeat,
                'min_TIM_height' : min_TIM_height,
                'layers' : layers
    }
    with open('data_dray1_051425.pkl', 'wb') as f:
        pickle.dump(data, f)

    # return
    # GPU_peak_temperature_list = []
    # HBM_peak_temperature_list = []
    GPU_time_frac_idle_list = []
    GPU_min_peak_temperature_list = []
    HBM_min_peak_temperature_list = []
    
    # box_temperatures = {box.name : [] for box in boxes}
    # print(box_temperatures)
    # results = simulator.simulate(boxes, bonding_box_list, TIM_boxes, heatsink_obj = heatsink_obj, heatsink_list = heatsink_list, heatsink_name = heatsink_name, bonding_list = bonding_list, bonding_name_type_dict = bonding_name_type_dict, is_repeat = is_repeat,  min_TIM_height = min_TIM_height, layers = layers) #
    anemoi_parameter_ID = {} # Uncomment
    # anemoi_parameter_ID = {'interposer_power': 1949, 'substrate_power': 1950, 'PCB_power': 1951, 'GPU_power': 1946, 'Power_Source_power': 1952, 'HBM_power': 1947, 'GPU_HTC_power': 1953, 'HBM_l_power': 1948, 'HBM_HTC_power': 1954}
    # print("Power dict initialized: ", power_dict)

    if(is_repeat == False):
        simulation_start_time = time.time()
        print("Starting simulation at ", simulation_start_time)
        
        is_repeat = is_repeat # False # False # True if the simulation is repeated with different powers, False if only one simulation is run.
        results = simulator_simulate(boxes, bonding_box_list, TIM_boxes, heatsink_obj = heatsink_obj, heatsink_list = heatsink_list, heatsink_name = heatsink_name, bonding_list = bonding_list, bonding_name_type_dict = bonding_name_type_dict, is_repeat = is_repeat,  min_TIM_height = min_TIM_height, power_dict = power_dict, anemoi_parameter_ID = anemoi_parameter_ID, layers = layers) #
        
        # Print hierarchical temperature statistics
        print("\n===== THERMAL ANALYSIS RESULTS =====\n")
        print_chiplet_temperatures(chiplet_tree[0], results, boxes)
        print("\n=====================================\n")
        
        print("\n===== DEBUG: OBJECT STRUCTURES =====")


        # if boxes:
        #     print("\nBOX example:")
        #     print(type(boxes[0]))
        #     print(vars(boxes[0]))

        # if bonding_box_list:
        #     print("\nBONDING BOX example:")
        #     print(type(bonding_box_list[0]))
        #     print(vars(bonding_box_list[0]))

        # if TIM_boxes:
        #     print("\nTIM BOX example:")
        #     print(type(TIM_boxes[0]))
        #     print(vars(TIM_boxes[0]))

        # if heatsink_obj:
        #     print("\nHEATSINK example:")
        #     print(type(heatsink_obj))
        #     print(heatsink_obj)

        # if layers:
        #     print("\nLAYER example:")
        #     print(type(layers[0]))
        #     print(vars(layers[0]))

        print("\n====================================\n")
        
        # print("Simulation results:")
        # for name, vals in results.items():
        #     print(name, "->", vals)

        simulation_end_time = time.time()
        print("Simulation finished at ", simulation_end_time)
        print("Time taken for simulation: ", simulation_end_time - simulation_start_time)

    # dedeepyo : 4-Jun-25

        # for box in boxes:
        #     # dedeepyo : 2-Jun-25 : Parameterizing box power.
        #     # Assuming, all boxes of a particular chiplet_type have same power.
        #     if box.power is None:
        #         box.power = 0.0
        #     box_power_in_mW = round(box.power * 1000, 6)
        #     try:
        #         power_num_in_mW = power_values[box.chiplet.get_chiplet_type()]
        #         # if(power_num_in_mW != box_power_in_mW): # Assuming this asymmetry is never encountered.
        #         #     self.papi.project_parameter_update(self.id, box.chiplet.get_chiplet_type() + "_power", box_power_in_mW)
        #         power_parameter_in_mW = anemoi_parameters[box.chiplet.get_chiplet_type() + "_power"]
        #     except KeyError:
        #         power_num_in_mW = box_power_in_mW
        #         power_parameter_in_mW = None
        #         for key, value in power_values.items():
        #             if value == power_num_in_mW:
        #                 power_parameter_in_mW = anemoi_parameters[key + "_power"]
        #                 power_values[box.chiplet.get_chiplet_type()] = power_num_in_mW
        #                 anemoi_parameters[box.chiplet.get_chiplet_type() + "_power"] = power_parameter_in_mW
        #         if power_parameter_in_mW is None:
        #             power_parameter_in_mW = box.chiplet.get_chiplet_type() + "_power"
        #             self.papi.project_parameter_create(self.id, power_parameter_in_mW, box.chiplet.get_chiplet_type() + "_power", power_num_in_mW)
        #             anemoi_parameters[power_parameter_in_mW] = power_parameter_in_mW     
        #             power_values[box.chiplet.get_chiplet_type()] = power_num_in_mW       
        #     # dedeepyo : 2-Jun-25

# def get_temperatures_from_results(file):
#     f = open(file, 'r')
#     lines = f.readlines()
#     f.close()
#     temperatures = {}

# dedeepyo : 4-Jun-25 : Implementing helper functions for iterations.
def get_GPU_count(boxes):
    GPU_count = 0
    for box in boxes:
        if box.chiplet_parent.get_chiplet_type() == "GPU":
            GPU_count += 1
    return GPU_count

def GPU_throttling(GPU_power = 275, GPU_time_frac_idle = 0.2, GPU_idle_power = 47):
  GPU_power_throttled = GPU_power * (1 - GPU_time_frac_idle) + GPU_idle_power * GPU_time_frac_idle
  return GPU_power_throttled

# def HBM_throttled_performance(bandwidth, latency, HBM_peak_temperature = 74):
#     if((HBM_peak_temperature > 85)):
#         bandwidth *= 0.732
#         latency *= 1.714
#     elif((HBM_peak_temperature > 75) and (HBM_peak_temperature <= 85)):
#         bandwidth *= 0.912
#         latency *= 1.238
    
#     return bandwidth, latency

def HBM_throttled_performance(bandwidth, latency, HBM_peak_temperature = 74):
    if((HBM_peak_temperature > 74)):
        bandwidth *= (2.82 - 0.018 * HBM_peak_temperature) # Linear interpolation between (75, 0.912) and (85, 0.732)
    if((HBM_peak_temperature > 85)):
        # bandwidth *= 0.732
        latency *= 1.714
    elif((HBM_peak_temperature > 75) and (HBM_peak_temperature <= 85)):
        # bandwidth *= 0.912
        latency *= 1.238
    
    return bandwidth, latency

def HBM_throttled_power(bandwidth_throttled, HBM_power, bandwidth_reference = 1986, HBM_peak_temperature = 74):
    refresh_energy = 0.12 * HBM_power
    non_refresh_energy = HBM_power - refresh_energy
    if(bandwidth_throttled < bandwidth_reference):
        non_refresh_energy *= (bandwidth_throttled / bandwidth_reference)
    if(HBM_peak_temperature > 85):
        refresh_energy *= 4.004
    elif(HBM_peak_temperature > 75 and HBM_peak_temperature <= 85):
        refresh_energy *= 2.0
    return refresh_energy + non_refresh_energy
    
def update_power_source_backside(boxes, power_dict, efficiency = 0.9):
    total_power = 0
    for box in boxes:
        if(box.chiplet_parent.get_chiplet_type() != "Power_Source"):
            if(box.chiplet_parent.get_chiplet_type()[0:5] == "HBM_l"):
                box.power = power_dict["HBM_l"]
            elif(box.chiplet_parent.get_chiplet_type() == "HBM"):
                box.power = power_dict["HBM"]
            elif(box.chiplet_parent.get_chiplet_type() == "GPU"):
                box.power = power_dict["GPU"]
            elif(box.power > 0.00):
                print(f"Chiplet type: {box.chiplet_parent.get_chiplet_type()} has power {box.power}W\n")

            total_power += box.power
    
    ps = [box for box in boxes if box.chiplet_parent.get_chiplet_type() == "Power_Source"][0] # Assuming only one power source. For now, it is at bottom. backside power delivery.
    ps.power = (1 - efficiency) * total_power / efficiency
    ps.chiplet_parent.set_power(ps.power)
    return ps.power

def initialize_power_dict_values(boxes):
    power_dict = {}
    for box in boxes: # Assuming all boxes of a particular type have same power.
        if(box.chiplet_parent.get_chiplet_type() == "GPU"):
            power_dict["GPU"] = box.power
        elif(box.chiplet_parent.get_chiplet_type() == "HBM"):
            power_dict["HBM"] = box.power
        elif(box.chiplet_parent.get_chiplet_type()[0:5] == "HBM_l"):
            power_dict["HBM_l"] = box.power
        elif(box.chiplet_parent.get_chiplet_type() == "Power_Source"):
            power_dict["Power_Source"] = box.power
    return power_dict

# dedeepyo : 4-Jun-25

def read_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                if len(parts) == 4:
                    data.append([float(x) for x in parts])
    return np.array(data)

def interpolate_and_report(data, col2_values, file_handle, system_name, HTC, TIM_conductivity, infill_conductivity, underfill_conductivity, HBM_stack_height, dummy_Si):
    if LinearRegression is None or r2_score is None:
        raise ImportError("scikit-learn is required for interpolate_and_report()")

    slope_intercept_dict = {}
    for val in col2_values:
        slope_intercept_dict[val] = {'peak_GPU_temp': (0.0, 0.0), 'peak_HBM_temp': (0.0, 0.0)}
        mask = np.isclose(data[:,1], val)
        subset = data[mask]
        if subset.shape[0] < 2:
            continue  # Need at least 2 points for regression
        x = subset[:,0].reshape(-1, 1)
        for col_idx, col_name in zip([2,3], ['peak_GPU_temp', 'peak_HBM_temp']):
            y = subset[:,col_idx]
            model = LinearRegression()
            model.fit(x, y)
            y_pred = model.predict(x)
            r2 = r2_score(y, y_pred)
            slope_intercept_dict[val][col_name] = (f"{model.coef_[0]:.3f}", f"{model.intercept_:.2f}")
            # file_handle.write(f"Interpolation for col={val}, {col_name}:")
            # file_handle.write(f"  Slope: {model.coef_[0]:.6f}, Intercept: {model.intercept_:.6f}, R^2: {r2:.6f}\n")

    # "system_name,HBM_power(W),HTC(W/(m2K)),TIM_conductivity(W/(mK)),infill_conductivity(W/(mK)),underfill_conductivity(W/(mK)),HBM_stack_height,dummy_Si"

    for key1 in slope_intercept_dict:
        file_handle.write(f"{system_name},{key1},{HTC},{TIM_conductivity},{infill_conductivity},{underfill_conductivity},{HBM_stack_height},{dummy_Si},{slope_intercept_dict[key1]['peak_GPU_temp'][0]},{slope_intercept_dict[key1]['peak_GPU_temp'][1]},{slope_intercept_dict[key1]['peak_HBM_temp'][0]},{slope_intercept_dict[key1]['peak_HBM_temp'][1]}\n")
    #     # print(f"{slope_intercept_dict[key1]['peak_GPU_temp'][0]}, {slope_intercept_dict[key1]['peak_HBM_temp'][0]}")
    #     # print(f"{slope_intercept_dict[key1]['peak_GPU_temp'][1]}, {slope_intercept_dict[key1]['peak_HBM_temp'][1]}")
    #     # print("calibrate_GPU")
    #     file_handle.write(f"calibrate_GPU :: {key1} : ({slope_intercept_dict[key1]['peak_GPU_temp'][0]}, {slope_intercept_dict[key1]['peak_GPU_temp'][1]})\n")
    #     # print("calibrate_HBM")
    #     file_handle.write(f"calibrate_HBM :: {key1} : ({slope_intercept_dict[key1]['peak_HBM_temp'][0]}, {slope_intercept_dict[key1]['peak_HBM_temp'][1]})\n")

def write_calibration_to_csv(
    system_name,
    HBM_power,
    HTC,
    TIM_cond,
    infill_cond,
    underfill_cond,
    HBM_stack_height,
    dummy_Si,
    calibrate_GPU_slope,
    calibrate_GPU_intercept,
    calibrate_HBM_slope,
    calibrate_HBM_intercept,
    csv_file_path="calibration_data.csv"
):
    """
    Write calibration data to CSV file.
    
    Args:
        system_name: Name of the system (e.g., "2p5D_1GPU")
        HBM_power: HBM power in Watts
        HTC: Heat Transfer Coefficient in W/(m^2*K)
        TIM_cond: TIM conductivity in W/(m*K)
        infill_cond: Infill conductivity in W/(m*K)
        underfill_cond: Underfill conductivity in W/(m*K)
        HBM_stack_height: HBM stack height (number of dies)
        dummy_Si: Boolean indicating if dummy Si is present
        calibrate_GPU_slope: GPU calibration slope
        calibrate_GPU_intercept: GPU calibration intercept
        calibrate_HBM_slope: HBM calibration slope
        calibrate_HBM_intercept: HBM calibration intercept
        csv_file_path: Path to the CSV file (default: "calibration_data.csv")
    """
    # Ensure dummy_Si is a boolean and convert to string for CSV
    dummy_Si_str = str(bool(dummy_Si))
    
    # Prepare the row data
    row_data = [
        system_name,
        str(HBM_power),
        str(HTC),
        str(TIM_cond),
        str(infill_cond),
        str(underfill_cond),
        str(HBM_stack_height),
        dummy_Si_str,
        str(calibrate_GPU_slope),
        str(calibrate_GPU_intercept),
        str(calibrate_HBM_slope),
        str(calibrate_HBM_intercept)
    ]
    
    # Check if file exists to determine if we need to write header
    file_exists = os.path.exists(csv_file_path)
    
    # Open file in append mode
    with open(csv_file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header if file doesn't exist
        if not file_exists:
            header = [
                "system_name",
                "HBM_power(W)",
                "HTC(W/(m2K))",
                "TIM_conductivity(W/(mK))",
                "infill_conductivity(W/(mK))",
                "underfill_conductivity(W/(mK))",
                "HBM_stack_height",
                "dummy_Si",
                "calibrate_GPU_slope",
                "calibrate_GPU_intercept",
                "calibrate_HBM_slope",
                "calibrate_HBM_intercept"
            ]
            writer.writerow(header)
        
        # Write the data row
        writer.writerow(row_data)

def parse_calibration_block(block: str) -> Tuple[str, List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Extract condition line and calibration tuples for GPU and HBM from a text block."""
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Empty calibration block encountered")

    condition_with_suffix = lines[0]
    colon_index = condition_with_suffix.find(":")
    if colon_index == -1:
        raise ValueError(f"Condition line missing ':' separator: {condition_with_suffix!r}")
    condition_line = condition_with_suffix[: colon_index + 1]

    gpu_entries: List[Tuple[str, str]] = []
    hbm_entries: List[Tuple[str, str]] = []
    entry_pattern = re.compile(r"calibrate_(GPU|HBM) :: ([^:]+) : \(([^)]+)\)")

    for line in lines[1:]:
        match = entry_pattern.search(line)
        if not match:
            continue
        target, setpoint, values = match.groups()
        if target.upper() == "GPU":
            gpu_entries.append((setpoint.strip(), values.strip()))
        else:
            hbm_entries.append((setpoint.strip(), values.strip()))

    if not gpu_entries and not hbm_entries:
        raise ValueError(f"No calibration entries found for block starting: {condition_line}")

    return condition_line, gpu_entries, hbm_entries

def format_condition_block(condition: str, entries: List[Tuple[str, str]]) -> List[str]:
    """Render a condition block without a calibrate header."""
    if not entries:
        return []

    # output = [condition, '    temperature_dict["2p5D_1GPU"] = {']
    output = [condition, '    temperature_dict["3D_1GPU"] = {']
    for index, (setpoint, values) in enumerate(entries):
        trailing = "," if index < len(entries) - 1 else ""
        output.append(f"        {setpoint} : ({values}){trailing}")
    output.append("    }")
    output.append("")
    return output

def convert(source: Path, destination: Path, hbm_stack_height = 1) -> None:
    raw_text = source.read_text()
    blocks = [block.strip() for block in raw_text.split("\n\n") if block.strip()]

    gpu_sections: List[List[str]] = []
    hbm_sections: List[List[str]] = []

    for block in blocks:
        condition, gpu_entries, hbm_entries = parse_calibration_block(block)
        gpu_block = format_condition_block(condition, gpu_entries)
        if gpu_block:
            gpu_sections.append(gpu_block)
        hbm_block = format_condition_block(condition, hbm_entries)
        if hbm_block:
            hbm_sections.append(hbm_block)

    output_lines: List[str] = []
    output_lines.append("hbm_stack_height = {}".format(hbm_stack_height))

    if gpu_sections:
        output_lines.append("calibrate_GPU")
        for block in gpu_sections:
            output_lines.extend(block)

    if hbm_sections:
        if output_lines and output_lines[-1] != "":
            output_lines.append("")
        output_lines.append("calibrate_HBM")
        for block in hbm_sections:
            output_lines.extend(block)

    while output_lines and output_lines[-1] == "":
        output_lines.pop()

    destination.write_text("\n".join(output_lines) + "\n")


# ZUONING BEGIN

# ============================================================
# Fine-grained thermal solver for simulator_simulate()
# Paste this ABOVE: if __name__ == '__main__': therm()
# ============================================================

from dataclasses import dataclass
from collections import defaultdict

try:
    from scipy.sparse import lil_matrix, csr_matrix
    from scipy.sparse.linalg import spsolve
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

AMBIENT_TEMP_C = 45.0
EPS = 1e-18


def use_pyspice_by_default() -> bool:
    """
    Read the optional backend override from the environment.
    Defaults to enabled to preserve the existing behavior when available.
    """
    return os.environ.get("THERM_USE_PYSPICE", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


USE_PYSPICE_DEFAULT = use_pyspice_by_default()


@dataclass
class ThermalCell:
    idx: int
    i: int
    j: int
    k: int
    x0: float
    x1: float
    y0: float
    y1: float
    z0: float
    z1: float
    dx_mm: float
    dy_mm: float
    dz_mm: float
    center_x: float
    center_y: float
    center_z: float
    owner_name: str
    owner_box: object
    k_eff: float
    power_w: float
    is_original_box: bool


def mm_to_m(x_mm: float) -> float:
    """
    Convert millimeters to meters.
    """
    return x_mm * 1e-3


def safe_float(x, default=None):
    """
    Try to convert x to float.
    If it fails, return default.
    """
    try:
        return float(x)
    except Exception:
        return default


def dedup_sorted(values, tol=1e-9):
    """
    Sort values and remove duplicates (within tolerance).
    """
    if not values:
        return []
    values = sorted(values)
    out = [values[0]]
    for v in values[1:]:
        if abs(v - out[-1]) > tol:
            out.append(v)
    return out


def normalize_material_name(name: str) -> str:
    """
    Clean up a material/layer name string.
    """
    if name is None:
        return ""
    s = str(name).strip()
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def infer_material_from_name(name: str) -> str:
    """
    Guess a known material from a layer/material name.
    """
    if name is None:
        return "Si"

    n = normalize_material_name(name).lower()

    if "epoxy, silver filled" in n or "epoxy silver filled" in n:
        return "Epoxy, Silver filled"
    if "epag" in n:
        return "EpAg"
    if "tim0p5" in n:
        return "TIM0p5"
    if "tim001" in n:
        return "TIM001"
    if "tim" in n:
        return "TIM"
    if "aln" in n:
        return "AlN"
    if "glass" in n:
        return "Glass"
    if "fr-4" in n or "fr4" in n or "pcb" in n:
        return "FR-4"
    if "air" in n:
        return "Air"
    if "aluminium" in n or "aluminum" in n:
        return "Aluminium"
    if "snpb" in n or "solder" in n or "bga" in n:
        return "SnPb 67/37"
    if "sio2" in n:
        return "SiO2"
    if "cu" in n or "copper" in n or "metal" in n:
        return "Cu-Foil"
    if "dummysi" in n:
        return "Si"
    if "silicon" in n or n == "si" or " active" in n or "interposer" in n or "substrate" in n or "wafer" in n:
        return "Si"
    if "infill" in n or "underfill" in n:
        return "Infill_material"

    return "Si"


def material_to_k(material_name: str) -> float:
    """
    Convert material name to conductivity k.
    """
    if material_name in conductivity_values:
        return float(conductivity_values[material_name])

    inferred = infer_material_from_name(material_name)
    if inferred in conductivity_values:
        return float(conductivity_values[inferred])

    return float(conductivity_values.get("Si", 105.0))


def parse_material_effective_k(material_str: str) -> float:
    """
    Parse a material string and return effective conductivity.

    Handles:
      1) single material:
         "Si"
         "Cu-Foil"

      2) mixture:
         "Cu-Foil:0.5,Si:0.5"
         "Cu-Foil:0.25,SiO2:0.75"
         "Cu-Foil:0.1,FR-4:0.9"

    Weighted average:
        k_eff = sum(frac_i * k_i) / sum(frac_i)
    """
    if material_str is None or str(material_str).strip() == "":
        return material_to_k("Si")

    s = str(material_str).strip()

    # protect commas inside names if needed
    protected_map = {
        "Epoxy, Silver filled": "Epoxy__Silver__filled"
    }
    for old, new in protected_map.items():
        s = s.replace(old, new)

    tokens = [tok.strip() for tok in s.split(",") if tok.strip()]
    tokens = [tok.replace("Epoxy__Silver__filled", "Epoxy, Silver filled") for tok in tokens]

    # If no ':' appears, treat as single material
    if all(":" not in tok for tok in tokens):
        return material_to_k(s)

    weighted_sum = 0.0
    weight_total = 0.0

    for tok in tokens:
        parts = [p.strip() for p in tok.split(":")]

        # mixture token should look like material:weight
        if len(parts) == 2:
            mat = parts[0]
            frac = safe_float(parts[1], None)

            if frac is None:
                # fallback: interpret as single material
                k_val = material_to_k(mat)
                weighted_sum += k_val
                weight_total += 1.0
            else:
                k_val = material_to_k(mat)
                weighted_sum += frac * k_val
                weight_total += frac

        else:
            # fallback
            k_val = material_to_k(tok)
            weighted_sum += k_val
            weight_total += 1.0

    if weight_total <= 0:
        return material_to_k("Si")

    return weighted_sum / weight_total

def build_layer_lookup(layers):
    """
    Build:
        layer_name -> {thickness, material, k}

    This uses the layer objects passed into simulator_simulate().
    """
    lookup = {}

    if layers is None:
        return lookup

    for layer in layers:
        layer_name = None
        if hasattr(layer, "get_name"):
            layer_name = layer.get_name()
        elif hasattr(layer, "name"):
            layer_name = layer.name

        if layer_name is None:
            continue

        thickness = None
        if hasattr(layer, "get_thickness"):
            thickness = safe_float(layer.get_thickness(), 0.0)
        elif hasattr(layer, "thickness"):
            thickness = safe_float(layer.thickness, 0.0)
        else:
            thickness = 0.0

        material = None
        if hasattr(layer, "get_material"):
            material = layer.get_material()
        elif hasattr(layer, "material"):
            material = layer.material

        if material is None or str(material).strip() == "":
            material = infer_material_from_name(layer_name)

        k_val = parse_material_effective_k(material)

        lookup[str(layer_name)] = {
            "thickness": float(thickness),
            "material": material,
            "k": float(k_val),
        }

    return lookup


def parse_stackup_effective_k(stackup: str, layer_lookup: dict) -> float:
    """
    Compute effective conductivity k for one box.

    Supports two styles:
      1) layered stackup:
         "1:5nm_GPU_active_3D,20:5nm_GPU_metal"

      2) mixture stackup:
         "1:Cu-Foil:60,Epoxy, Silver filled:40"

    Uses weighted average k.
    """
    if stackup is None or str(stackup).strip() == "":
        return material_to_k("Si")

    s = str(stackup).strip()

    # protect commas inside material names
    protected_map = {
        "Epoxy, Silver filled": "Epoxy__Silver__filled"
    }
    for old, new in protected_map.items():
        s = s.replace(old, new)

    raw_tokens = [tok.strip() for tok in s.split(",") if tok.strip()]
    tokens = []
    for tok in raw_tokens:
        for old, new in protected_map.items():
            tok = tok.replace(new, old)
        tokens.append(tok)

    weighted_sum = 0.0
    weight_total = 0.0

    # detect likely mixture format
    likely_mixture = False
    for tok in tokens:
        parts = [p.strip() for p in tok.split(":")]
        if len(parts) >= 3:
            last_num = safe_float(parts[-1], None)
            if last_num is not None and 0.0 <= last_num <= 100.0:
                likely_mixture = True

    if likely_mixture:
        for tok in tokens:
            parts = [p.strip() for p in tok.split(":")]

            if len(parts) == 2:
                mat = parts[0]
                frac = safe_float(parts[1], None)
                if frac is None:
                    continue
                k_val = material_to_k(mat)
                weighted_sum += frac * k_val
                weight_total += frac

            elif len(parts) >= 3:
                mat = parts[-2]
                frac = safe_float(parts[-1], None)
                if frac is None:
                    continue
                k_val = material_to_k(mat)
                weighted_sum += frac * k_val
                weight_total += frac

        if weight_total > 0:
            return weighted_sum / weight_total

    # layered interpretation
    for tok in tokens:
        parts = [p.strip() for p in tok.split(":")]

        if len(parts) == 2:
            count_str, layer_name = parts
            count = safe_float(count_str, None)

            if count is None:
                # fallback
                k_val = material_to_k(layer_name)
                weighted_sum += 1.0 * k_val
                weight_total += 1.0
                continue

            if layer_name in layer_lookup:
                thickness = layer_lookup[layer_name]["thickness"]
                k_val = layer_lookup[layer_name]["k"]
            else:
                thickness = 1.0
                k_val = material_to_k(layer_name)

            w = count * thickness
            weighted_sum += w * k_val
            weight_total += w

        elif len(parts) >= 3:
            # mixture-like token
            count = safe_float(parts[0], None)
            last_num = safe_float(parts[-1], None)

            if count is not None and last_num is not None:
                mat_name = parts[-2]
                frac = last_num
                k_val = material_to_k(mat_name)
                weighted_sum += frac * k_val
                weight_total += frac
            else:
                mat_name = parts[-1]
                k_val = material_to_k(mat_name)
                weighted_sum += 1.0 * k_val
                weight_total += 1.0

    if weight_total > 0:
        return weighted_sum / weight_total

    return material_to_k(infer_material_from_name(s))


def make_heatsink_box(heatsink_obj):
    """
    Convert heatsink base into a Box.
    We model the heatsink base as geometry.
    Fin effect is captured by HTC boundary condition.
    """
    if heatsink_obj is None:
        return None

    x = float(heatsink_obj["x"])
    y = float(heatsink_obj["y"])
    z = float(heatsink_obj["z"])
    dx = float(heatsink_obj["base_dx"])
    dy = float(heatsink_obj["base_dy"])
    dz = float(heatsink_obj["base_dz"])

    if dz <= 1e-9:
        dz = 0.1
    material = heatsink_obj.get("material", "Cu-Foil")
    name = heatsink_obj.get("name", "HS_top")

    return Box(x, y, z, dx, dy, dz, 0.0, f"1:{material}", 0.0, name)


def get_box_chiplet_type(box) -> str:
    """
    Best-effort extraction of the chiplet type for a box.
    """
    try:
        return str(box.chiplet_parent.get_chiplet_type())
    except Exception:
        return str(getattr(box, "name", ""))


def is_center_plane_power_box(box) -> bool:
    """
    The project statement specifies center-plane power injection for GPU and HBM dies.
    """
    chiplet_type = get_box_chiplet_type(box)
    return (
        chiplet_type == "GPU" or
        chiplet_type == "HBM" or
        chiplet_type.startswith("HBM_l")
    )


def assign_cell_powers(cells):
    """
    Distribute each box's power onto its owned cells.

    Project-specific rule:
      - GPU / HBM boxes inject power only on the vertical center plane.
      - Other powered boxes default to volumetric distribution.
    """
    cells_by_owner = defaultdict(list)
    for cell in cells:
        cells_by_owner[cell.owner_name].append(cell)

    for owner_cells in cells_by_owner.values():
        owner = owner_cells[0].owner_box
        owner_power = 0.0 if owner.power is None else float(owner.power)

        if abs(owner_power) <= EPS:
            for cell in owner_cells:
                cell.power_w = 0.0
            continue

        if is_center_plane_power_box(owner):
            z_center = 0.5 * (owner.start_z + owner.end_z)
            selected_cells = [
                cell for cell in owner_cells
                if cell.z0 - 1e-9 <= z_center <= cell.z1 + 1e-9
            ]

            if not selected_cells:
                best_distance = min(
                    abs(0.5 * (cell.z0 + cell.z1) - z_center)
                    for cell in owner_cells
                )
                selected_cells = [
                    cell for cell in owner_cells
                    if abs(0.5 * (cell.z0 + cell.z1) - z_center) <= best_distance + 1e-12
                ]

            weights = [
                max(mm_to_m(cell.dx_mm) * mm_to_m(cell.dy_mm), EPS)
                for cell in selected_cells
            ]
            weight_total = max(sum(weights), EPS)

            for cell in owner_cells:
                cell.power_w = 0.0

            for cell, weight in zip(selected_cells, weights):
                cell.power_w = owner_power * weight / weight_total
        else:
            weights = [
                max(
                    mm_to_m(cell.dx_mm) * mm_to_m(cell.dy_mm) * mm_to_m(cell.dz_mm),
                    EPS
                )
                for cell in owner_cells
            ]
            weight_total = max(sum(weights), EPS)
            for cell, weight in zip(owner_cells, weights):
                cell.power_w = owner_power * weight / weight_total


def point_inside_box(cx, cy, cz, box, tol=1e-9):
    """
    Check whether point (cx, cy, cz) lies inside box.
    """
    return (
        box.start_x - tol <= cx <= box.end_x + tol and
        box.start_y - tol <= cy <= box.end_y + tol and
        box.start_z - tol <= cz <= box.end_z + tol
    )


def find_owner_box(cx, cy, cz, solver_boxes):
    """
    Find the box that owns a voxel center.
    If multiple boxes contain the point, choose the smallest-volume box.
    """
    candidates = []
    for box in solver_boxes:
        if point_inside_box(cx, cy, cz, box):
            vol = max(box.width * box.length * box.height, EPS)
            candidates.append((vol, box))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def build_axis_lines(solver_boxes, axis='x', pitch_mm=1.0):
    """
    Build sorted unique grid boundaries for one axis.
    Includes:
      - all box start/end coordinates
      - regular pitch cuts
    """
    if axis == 'x':
        starts = [b.start_x for b in solver_boxes]
        ends = [b.end_x for b in solver_boxes]
    elif axis == 'y':
        starts = [b.start_y for b in solver_boxes]
        ends = [b.end_y for b in solver_boxes]
    elif axis == 'z':
        starts = [b.start_z for b in solver_boxes]
        ends = [b.end_z for b in solver_boxes]
    else:
        raise ValueError("axis must be x/y/z")

    lo = min(starts)
    hi = max(ends)

    values = []
    values.extend(starts)
    values.extend(ends)

    curr = math.floor(lo / pitch_mm) * pitch_mm
    while curr <= hi + 1e-12:
        values.append(round(curr, 9))
        curr += pitch_mm

    values.append(lo)
    values.append(hi)

    return dedup_sorted(values, tol=1e-9)


def build_voxel_grid(
    solver_boxes,
    original_box_names,
    layer_lookup,
    xy_pitch_mm=1.0,
    z_pitch_mm=0.1
):
    """
    Build fine-grained voxel cells.

    Each voxel:
      - belongs to one owner box
      - gets that box's effective k
      - gets a fraction of that box's power
    """
    x_lines = build_axis_lines(solver_boxes, axis='x', pitch_mm=xy_pitch_mm)
    y_lines = build_axis_lines(solver_boxes, axis='y', pitch_mm=xy_pitch_mm)
    z_lines = build_axis_lines(solver_boxes, axis='z', pitch_mm=z_pitch_mm)

    cells = []
    cell_by_ijk = {}
    cell_idx = 0

    for i in range(len(x_lines) - 1):
        x0 = x_lines[i]
        x1 = x_lines[i + 1]
        dx_mm = x1 - x0
        if dx_mm <= 1e-12:
            continue

        for j in range(len(y_lines) - 1):
            y0 = y_lines[j]
            y1 = y_lines[j + 1]
            dy_mm = y1 - y0
            if dy_mm <= 1e-12:
                continue

            for k in range(len(z_lines) - 1):
                z0 = z_lines[k]
                z1 = z_lines[k + 1]
                dz_mm = z1 - z0
                if dz_mm <= 1e-12:
                    continue

                cx = 0.5 * (x0 + x1)
                cy = 0.5 * (y0 + y1)
                cz = 0.5 * (z0 + z1)

                owner = find_owner_box(cx, cy, cz, solver_boxes)
                if owner is None:
                    continue

                k_eff = parse_stackup_effective_k(owner.stackup, layer_lookup)

                is_original = owner.name in original_box_names

                cell = ThermalCell(
                    idx=cell_idx,
                    i=i, j=j, k=k,
                    x0=x0, x1=x1,
                    y0=y0, y1=y1,
                    z0=z0, z1=z1,
                    dx_mm=dx_mm, dy_mm=dy_mm, dz_mm=dz_mm,
                    center_x=cx, center_y=cy, center_z=cz,
                    owner_name=owner.name,
                    owner_box=owner,
                    k_eff=float(k_eff),
                    power_w=0.0,
                    is_original_box=is_original
                )

                cells.append(cell)
                cell_by_ijk[(i, j, k)] = cell
                cell_idx += 1

    assign_cell_powers(cells)

    return cells, cell_by_ijk, x_lines, y_lines, z_lines


def add_conduction_between_cells(G, cell_a: ThermalCell, cell_b: ThermalCell, axis: str):
    """
    Add conduction between two neighboring cells.

    Resistance formula:
        R = (L_a/2)/(k_a A) + (L_b/2)/(k_b A)

    Then conductance:
        g = 1/R
    """
    if axis == 'x':
        area_m2 = mm_to_m(cell_a.dy_mm) * mm_to_m(cell_a.dz_mm)
        La = 0.5 * mm_to_m(cell_a.dx_mm)
        Lb = 0.5 * mm_to_m(cell_b.dx_mm)
    elif axis == 'y':
        area_m2 = mm_to_m(cell_a.dx_mm) * mm_to_m(cell_a.dz_mm)
        La = 0.5 * mm_to_m(cell_a.dy_mm)
        Lb = 0.5 * mm_to_m(cell_b.dy_mm)
    elif axis == 'z':
        area_m2 = mm_to_m(cell_a.dx_mm) * mm_to_m(cell_a.dy_mm)
        La = 0.5 * mm_to_m(cell_a.dz_mm)
        Lb = 0.5 * mm_to_m(cell_b.dz_mm)
    else:
        raise ValueError("axis must be x/y/z")

    area_m2 = max(area_m2, EPS)
    ka = max(cell_a.k_eff, EPS)
    kb = max(cell_b.k_eff, EPS)

    R = La / (ka * area_m2) + Lb / (kb * area_m2)
    g = 1.0 / max(R, EPS)

    ia = cell_a.idx
    ib = cell_b.idx

    G[ia, ia] += g
    G[ib, ib] += g
    G[ia, ib] -= g
    G[ib, ia] -= g


def add_convection_to_ambient(G, cell: ThermalCell, htc_w_m2k: float, face='top'):
    """
    Add convection from a cell face to ambient.

    g = h * A
    """
    if face in ('top', 'bottom'):
        area_m2 = mm_to_m(cell.dx_mm) * mm_to_m(cell.dy_mm)
    elif face == 'x':
        area_m2 = mm_to_m(cell.dy_mm) * mm_to_m(cell.dz_mm)
    elif face == 'y':
        area_m2 = mm_to_m(cell.dx_mm) * mm_to_m(cell.dz_mm)
    else:
        raise ValueError("face must be top/bottom/x/y")

    h = max(float(htc_w_m2k), 0.0)
    area_m2 = max(area_m2, 0.0)

    if h <= 0.0 or area_m2 <= 0.0:
        return

    g = h * area_m2
    G[cell.idx, cell.idx] += g


def solve_thermal_system(num_cells, cells, cell_by_ijk, heatsink_box=None, heatsink_obj=None):
    """
    Solve steady-state thermal system:
        G * dT = P
    where dT is temperature rise above ambient.
    """
    if num_cells == 0:
        return np.array([])

    use_sparse = SCIPY_AVAILABLE

    if use_sparse:
        G = lil_matrix((num_cells, num_cells), dtype=float)
    else:
        G = np.zeros((num_cells, num_cells), dtype=float)

    P = np.zeros(num_cells, dtype=float)

    for cell in cells:
        P[cell.idx] = cell.power_w

    # neighbor conduction
    for cell in cells:
        i, j, k = cell.i, cell.j, cell.k

        nbr = cell_by_ijk.get((i + 1, j, k), None)
        if nbr is not None:
            add_conduction_between_cells(G, cell, nbr, 'x')

        nbr = cell_by_ijk.get((i, j + 1, k), None)
        if nbr is not None:
            add_conduction_between_cells(G, cell, nbr, 'y')

        nbr = cell_by_ijk.get((i, j, k + 1), None)
        if nbr is not None:
            add_conduction_between_cells(G, cell, nbr, 'z')

    # convection from exposed top of heatsink
    conv_count = 0
    heatsink_cell_count = 0
    top_heatsink_cell_count = 0
    if heatsink_box is not None and heatsink_obj is not None:
        htc = float(heatsink_obj.get("hc", 0.0))
        z_top = heatsink_box.end_z
        tol = 1e-9

        for cell in cells:
            if cell.owner_name != heatsink_box.name:
                continue

            heatsink_cell_count += 1

            if abs(cell.z1 - z_top) < tol:
                if htc > 0.0:
                    add_convection_to_ambient(G, cell, htc, face='top')
                    conv_count += 1
                    top_heatsink_cell_count += 1
    else:
        # fallback anchor if heatsink is absent
        weak_h = 5.0
        for cell in cells:
            if cell_by_ijk.get((cell.i, cell.j, cell.k + 1), None) is None:
                add_convection_to_ambient(G, cell, weak_h, face='top')

    print("Heatsink voxel count =", heatsink_cell_count)
    print("Top heatsink voxel count =", top_heatsink_cell_count)
    print("Number of convection cells =", conv_count)
    print("Heatsink name =", heatsink_box.name if heatsink_box is not None else None)
    print("HTC =", heatsink_obj.get("hc") if heatsink_obj is not None else None)
    if conv_count == 0:
        raise RuntimeError(
            "Thermal network has no ambient boundary condition. "
            "Check heatsink hc settings."
        )
    # small numerical stabilization
    for idx in range(num_cells):
        G[idx, idx] += 1e-12

    if use_sparse:
        G = csr_matrix(G)
        dT = spsolve(G, P)
    else:
        if num_cells > 3000:
            raise RuntimeError(
                "Grid too large for dense solve and scipy is unavailable. "
                "Install scipy or use coarser grid."
            )
        dT = np.linalg.solve(G, P)

    T = dT + AMBIENT_TEMP_C
    return T


def solve_thermal_system_pyspice(cells, cell_by_ijk, heatsink_box=None, heatsink_obj=None):
    """
    Solve the thermal network using a real PySpice operating-point analysis.
    """
    from PySpice.Spice.Netlist import Circuit

    circuit = Circuit('Thermal Network')

    for cell in cells:
        node_name = f'N{cell.idx}'
        # Inject thermal power into the node so a simple 1 A / 2 ohm test gives +2 C, not -2 C.
        circuit.CurrentSource(f'P{cell.idx}', circuit.gnd, node_name, float(cell.power_w))

    res_count = 0
    for cell in cells:
        i, j, k = cell.i, cell.j, cell.k
        for axis, di, dj, dk in [('x', 1, 0, 0), ('y', 0, 1, 0), ('z', 0, 0, 1)]:
            nbr = cell_by_ijk.get((i + di, j + dj, k + dk), None)
            if nbr is None or cell.idx >= nbr.idx:
                continue

            if axis == 'x':
                area_m2 = mm_to_m(cell.dy_mm) * mm_to_m(cell.dz_mm)
                La = 0.5 * mm_to_m(cell.dx_mm)
                Lb = 0.5 * mm_to_m(nbr.dx_mm)
            elif axis == 'y':
                area_m2 = mm_to_m(cell.dx_mm) * mm_to_m(cell.dz_mm)
                La = 0.5 * mm_to_m(cell.dy_mm)
                Lb = 0.5 * mm_to_m(nbr.dy_mm)
            else:
                area_m2 = mm_to_m(cell.dx_mm) * mm_to_m(cell.dy_mm)
                La = 0.5 * mm_to_m(cell.dz_mm)
                Lb = 0.5 * mm_to_m(nbr.dz_mm)

            area_m2 = max(area_m2, EPS)
            ka = max(cell.k_eff, EPS)
            kb = max(nbr.k_eff, EPS)
            resistance = max(La / (ka * area_m2) + Lb / (kb * area_m2), EPS)

            circuit.R(
                f'C{res_count}',
                f'N{cell.idx}',
                f'N{nbr.idx}',
                resistance
            )
            res_count += 1

    conv_count = 0
    if heatsink_box is not None and heatsink_obj is not None:
        htc = float(heatsink_obj.get("hc", 0.0))
        z_top = heatsink_box.end_z
        tol = 1e-9

        for cell in cells:
            if cell.owner_name != heatsink_box.name:
                continue
            if abs(cell.z1 - z_top) >= tol or htc <= 0.0:
                continue

            area_m2 = mm_to_m(cell.dx_mm) * mm_to_m(cell.dy_mm)
            if area_m2 <= 0.0:
                continue

            g = htc * area_m2
            r_conv = 1.0 / max(g, EPS)
            circuit.R(f'V{conv_count}', f'N{cell.idx}', circuit.gnd, r_conv)
            conv_count += 1
    else:
        weak_h = 5.0
        for cell in cells:
            if cell_by_ijk.get((cell.i, cell.j, cell.k + 1), None) is not None:
                continue

            area_m2 = mm_to_m(cell.dx_mm) * mm_to_m(cell.dy_mm)
            if area_m2 <= 0.0:
                continue

            g = weak_h * area_m2
            r_conv = 1.0 / max(g, EPS)
            circuit.R(f'V{conv_count}', f'N{cell.idx}', circuit.gnd, r_conv)
            conv_count += 1

    if conv_count == 0:
        raise RuntimeError(
            "PySpice thermal network has no ambient boundary condition. "
            "Check heatsink hc settings."
        )

    simulator = circuit.simulator(
        temperature=AMBIENT_TEMP_C,
        nominal_temperature=AMBIENT_TEMP_C
    )
    analysis = simulator.operating_point()

    temperatures = np.zeros(len(cells), dtype=float)
    for cell in cells:
        node_name = f'N{cell.idx}'
        temperatures[cell.idx] = float(analysis[node_name]) + AMBIENT_TEMP_C

    return temperatures


def compute_box_directional_resistances(box, layer_lookup):
    """
    Compute simple directional thermal resistances for one original box:
        Rx = Lx / (k A_yz)
        Ry = Ly / (k A_xz)
        Rz = Lz / (k A_xy)
    """
    k_eff = parse_stackup_effective_k(box.stackup, layer_lookup)
    k_eff = max(k_eff, EPS)

    wx = max(mm_to_m(box.width), EPS)
    ly = max(mm_to_m(box.length), EPS)
    hz = max(mm_to_m(box.height), EPS)

    A_yz = max(ly * hz, EPS)
    A_xz = max(wx * hz, EPS)
    A_xy = max(wx * ly, EPS)

    Rx = wx / (k_eff * A_yz)
    Ry = ly / (k_eff * A_xz)
    Rz = hz / (k_eff * A_xy)

    return float(Rx), float(Ry), float(Rz)


def aggregate_results(original_boxes, cells, temperatures, layer_lookup):
    """
    Aggregate voxel temperatures back to original input boxes only.

    Returns:
        {
            box.name: (peak_temp, avg_temp, Rx, Ry, Rz)
        }
    """
    cell_temps_by_owner = defaultdict(list)
    cell_vols_by_owner = defaultdict(list)

    for cell in cells:
        if not cell.is_original_box:
            continue
        temp = float(temperatures[cell.idx])
        vol = mm_to_m(cell.dx_mm) * mm_to_m(cell.dy_mm) * mm_to_m(cell.dz_mm)
        cell_temps_by_owner[cell.owner_name].append(temp)
        cell_vols_by_owner[cell.owner_name].append(vol)

    results = {}

    for box in original_boxes:
        temps = cell_temps_by_owner.get(box.name, [])
        vols = cell_vols_by_owner.get(box.name, [])

        if len(temps) == 0:
            peak_temp = AMBIENT_TEMP_C
            avg_temp = AMBIENT_TEMP_C
        else:
            peak_temp = max(temps)
            vol_sum = max(sum(vols), EPS)
            avg_temp = sum(t * v for t, v in zip(temps, vols)) / vol_sum

        Rx, Ry, Rz = compute_box_directional_resistances(box, layer_lookup)

        results[box.name] = (
            float(peak_temp),
            float(avg_temp),
            float(Rx),
            float(Ry),
            float(Rz),
        )

    return results


def simulator_simulate(
    boxes,
    bonding_box_list,
    TIM_boxes,
    heatsink_obj=None,
    heatsink_list=None,
    heatsink_name=None,
    bonding_list=None,
    bonding_name_type_dict=None,
    is_repeat=False,
    min_TIM_height=0.1,
    power_dict=None,
    anemoi_parameter_ID=None,
    layers=None
):
    """
    Fine-grained thermal simulation using PySpice.

    Inputs:
      boxes            : original input boxes (the ones required by project output)
      bonding_box_list : helper thermal boxes for bonding layers
      TIM_boxes        : helper thermal boxes for TIM layers
      heatsink_obj     : heatsink base geometry + HTC info
      layers           : layer definitions used by stackup parsing

    Output:
      {
        box.name: (peak_temp, avg_temp, Rx, Ry, Rz)
      }
    """

    print("Entering simulator_simulate()")

    # Grid resolution
    XY_PITCH_MM = 4.0
    Z_PITCH_MM = 0.4

    original_boxes = list(boxes)
    original_box_names = set(b.name for b in original_boxes)

    solver_boxes = []
    solver_boxes.extend(original_boxes)
    solver_boxes.extend(bonding_box_list)
    solver_boxes.extend(TIM_boxes)

    heatsink_box = None
    if heatsink_obj is not None:
        heatsink_box = make_heatsink_box(heatsink_obj)
        solver_boxes.append(heatsink_box)
    
    layer_lookup = build_layer_lookup(layers)

    print(f"Original boxes      : {len(original_boxes)}")
    print(f"Bonding boxes       : {len(bonding_box_list)}")
    print(f"TIM boxes           : {len(TIM_boxes)}")
    print(f"Solver boxes total  : {len(solver_boxes)}")
    print(f"XY pitch (mm)       : {XY_PITCH_MM}")
    print(f"Z pitch (mm)        : {Z_PITCH_MM}")

    # Build voxel grid
    t0 = time.time()
    cells, cell_by_ijk, x_lines, y_lines, z_lines = build_voxel_grid(
        solver_boxes=solver_boxes,
        original_box_names=original_box_names,
        layer_lookup=layer_lookup,
        xy_pitch_mm=XY_PITCH_MM,
        z_pitch_mm=Z_PITCH_MM
    )
    t1 = time.time()

    print(f"Voxel cells built   : {len(cells)}")
    print(f"Grid build time (s) : {t1 - t0:.3f}")
    print("Total voxel power =", sum(c.power_w for c in cells))

    if USE_PYSPICE_DEFAULT:
        print("Running PySpice DC operating point analysis...")
        try:
            temperatures = solve_thermal_system_pyspice(
                cells=cells,
                cell_by_ijk=cell_by_ijk,
                heatsink_box=heatsink_box,
                heatsink_obj=heatsink_obj
            )
        except Exception as e:
            print(f"WARNING: PySpice execution failed ({e}); using internal matrix solver")
            temperatures = solve_thermal_system(
                num_cells=len(cells),
                cells=cells,
                cell_by_ijk=cell_by_ijk,
                heatsink_box=heatsink_box,
                heatsink_obj=heatsink_obj
            )
    else:
        print("Running internal sparse thermal solve...")
        temperatures = solve_thermal_system(
            num_cells=len(cells),
            cells=cells,
            cell_by_ijk=cell_by_ijk,
            heatsink_box=heatsink_box,
            heatsink_obj=heatsink_obj
        )

    t2 = time.time()
    print(f"Thermal solve time (s): {t2 - t1:.3f}")

    # Aggregate back to original boxes only
    results = aggregate_results(
        original_boxes=original_boxes,
        cells=cells,
        temperatures=temperatures,
        layer_lookup=layer_lookup
    )

    # helpful summary
    gpu_peak = []
    hbm_peak = []

    for box in original_boxes:
        ctype = ""
        try:
            ctype = box.chiplet_parent.get_chiplet_type()
        except Exception:
            pass

        peak_temp = results[box.name][0]

        if ctype == "GPU":
            gpu_peak.append(peak_temp)
        if ctype == "HBM" or str(ctype).startswith("HBM_l"):
            hbm_peak.append(peak_temp)

    if gpu_peak:
        print(f"Peak GPU temperature (C): {max(gpu_peak):.3f}")
    if hbm_peak:
        print(f"Peak HBM temperature (C): {max(hbm_peak):.3f}")

    print("simulator_simulate() finished")
    return results

# ZUONING END


def print_chiplet_temperatures(chiplet_tree_root, results_dict, original_boxes):
    """
    Traverse the chiplet hierarchy and print mean and maximum temperatures for each chiplet.
    
    Args:
        chiplet_tree_root: Root chiplet of the hierarchy
        results_dict: Dictionary from simulator_simulate with temperatures {box_name: (peak, avg, ...)}
        original_boxes: List of original Box objects
    """
    
    # Build a mapping from box name to chiplet for easier lookup
    box_name_to_chiplet = {}
    
    def map_boxes_to_chiplet(chiplet):
        box = chiplet.get_box_representation()
        if box:
            box_name_to_chiplet[box.name] = chiplet
        for child in chiplet.get_child_chiplets():
            map_boxes_to_chiplet(child)
    
    map_boxes_to_chiplet(chiplet_tree_root)
    
    # Calculate mean temperature of entire system
    total_weighted_temp = 0.0
    total_volume = 0.0
    
    for box in original_boxes:
        if box.name in results_dict:
            peak_temp, avg_temp, _, _, _ = results_dict[box.name]
            volume = box.width * box.length * box.height
            total_weighted_temp += avg_temp * volume
            total_volume += volume
    
    if total_volume > 0:
        system_mean_temp = total_weighted_temp / total_volume
        print(f"Mean temperature of entire system is {system_mean_temp}")
    
    # Recursively print temperatures for each chiplet
    def print_chiplet_stats(chiplet):
        # The chiplet name already contains the full hierarchical path
        full_path = chiplet.get_name()
        
        # Collect all boxes belonging to this chiplet (including descendants)
        chiplet_boxes = []
        
        def collect_boxes(c):
            box = c.get_box_representation()
            if box and box.name in results_dict:
                chiplet_boxes.append(box)
            for child in c.get_child_chiplets():
                collect_boxes(child)
        
        collect_boxes(chiplet)
        
        # Calculate statistics for this chiplet
        if chiplet_boxes:
            temps = []
            weighted_sum = 0.0
            total_vol = 0.0
            
            for box in chiplet_boxes:
                peak_temp, avg_temp, _, _, _ = results_dict[box.name]
                volume = box.width * box.length * box.height
                
                temps.append(peak_temp)
                weighted_sum += avg_temp * volume
                total_vol += volume
            
            if total_vol > 0:
                mean_temp = weighted_sum / total_vol
            else:
                mean_temp = 0.0
            
            max_temp = max(temps) if temps else 0.0
            
            print(f"Mean temperature of {full_path} chiplet is {mean_temp} and its maximum temperature is {max_temp}")
        
        # Recursively process children
        for child in chiplet.get_child_chiplets():
            print_chiplet_stats(child)
    
    # Start traversal from the root
    print_chiplet_stats(chiplet_tree_root)


if __name__ == '__main__':
    therm()

# 
# new_bandwidth = input("Enter the new bandwidth value (GB): ")



# # Update line 33 and line 40 (0-based index: 32 and 39)
# for idx in [32, 39]:
#     if "bandwidth:" in lines[idx]:
#         # Replace the value after 'bandwidth:'
#         parts = lines[idx].split("bandwidth:")
#         lines[idx] = f"{parts[0]}bandwidth: {new_bandwidth} GB\n"

# with open(file_path, "w") as f:
#     f.writelines(lines)

# print("Bandwidth values updated successfully.")
