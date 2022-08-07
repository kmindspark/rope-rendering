from matplotlib import bezier
import bpy, bpy_extras
import sys
import copy
from math import *
import pprint
from mathutils import *
import random
import json
import numpy as np
import os
import time
from sklearn.neighbors import NearestNeighbors
import argparse
import matplotlib.pyplot as plt
import random

class RopeRenderer:
    def __init__(self, rope_radius=None, sphere_radius=None, rope_iterations=None, rope_screw_offset=None, bezier_scale=3.7, bezier_knots=12, save_depth=True, save_rgb=False, coord_offset=20, num_images=10, nonplanar=True):
        """
        Initializes the Blender rope renderer
        :param rope_radius: thickness of rope
        :type rope_radius: float
        :param rope_screw_offset: how tightly wound the "screw" texture is
        :type rope_radius: int
        :param bezier_scale: length of bezier curve
        :type bezier_scale: int
        :param bezier_subdivisions: # nodes in bezier curve - 2
        :type bezier_subdivisions: int
        :param save_rgb: if True, save_rgbs images, else just renders
        :type save_rgb: bool
        :return:
        :rtype:
        """
        self.save_rgb = save_rgb # whether to save_rgb images or not
        self.num_images = num_images
        self.coord_offset = coord_offset
        self.save_depth = save_depth
        self.rope_radius = rope_radius
        self.rope_screw_offset = rope_screw_offset
        self.sphere_radius = sphere_radius
        self.rope_iterations = rope_iterations
        self.nonplanar = nonplanar
        self.bezier_scale = None
        self.bezier_subdivisions = bezier_knots - 2 # the number of splits in the bezier curve (ctrl points - 2)
        self.origin = (0, 0, 0)
        # Make objects
        self.rope = None
        self.rope_asymm = None
        self.bezier = None
        self.bezier_points = None # list of vertices in bezier curve
        self.camera = None
        # Name objects
        self.rope_name = "Rope"
        self.rope_asymm_name = "Rope-Asymmetric"
        self.bezier_name = "Bezier"
        self.camera_name = "Camera"
        # Dictionary to store pixel vals of knots (vertices)
        self.knots_info = {}
        self.i = 0

    def clear(self):
        """
        Deletes any objects or meshes in the scene
        """
        for block in bpy.data.meshes:
            if block.users == 0:
                bpy.data.meshes.remove(block)
        for block in bpy.data.materials:
            if block.users == 0:
                bpy.data.materials.remove(block)
        for block in bpy.data.textures:
            if block.users == 0:
                bpy.data.textures.remove(block)
        for block in bpy.data.images:
            if block.users == 0:
                bpy.data.images.remove(block)
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

    def add_camera(self, fixed=True):
        '''
        Place a camera randomly, fixed means no rotations about z axis (planar camera changes only)
        '''
        if fixed:
            bpy.ops.object.camera_add(location=[0, 0, 1.5])
            self.camera = bpy.context.active_object
            # self.camera.rotation_euler = (0, 0, random.uniform(-pi/8, pi/8)) # fixed z, rotate only about x/y axis slightly
            self.camera.rotation_euler = (random.uniform(-pi/32, pi/32), random.uniform(-pi/32, pi/32), random.uniform(-pi, pi))
            self.camera.name = self.camera_name
        else:
            bpy.ops.object.camera_add(location=[random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)])
            self.camera = bpy.context.active_object
            self.camera.name = self.camera_name
            self.camera.rotation_euler = (random.uniform(-pi/16, pi/16), random.uniform(-pi/16, pi/16), random.uniform(-pi, pi))
        bpy.ops.object.light_add(type='SUN', radius=1, location=(0, 0, 0))

    def make_bezier(self):
        '''
        Create bezier curve
        '''
        bpy.ops.curve.primitive_bezier_curve_add(location=self.origin)
        if self.bezier_scale is None:
            self.bezier_scale = 1 # np.random.uniform(2.85,3.02)
        bpy.ops.transform.resize(value=(self.bezier_scale, self.bezier_scale, self.bezier_scale))
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.curve.subdivide(number_cuts=self.bezier_subdivisions)
        bpy.ops.transform.resize(value=(1, 0, 1))
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.curve.select_all(action='SELECT')
        bpy.ops.curve.handle_type_set(type='AUTOMATIC')
        bpy.ops.object.mode_set(mode='OBJECT')
        self.bezier = bpy.context.active_object
        self.bezier_points = self.bezier.data.splines[0].bezier_points
        self.bezier.name = self.bezier_name
        self.bezier.select_set(False)

        # set geometry bevel with depth of 0.01
        self.bezier.data.bevel_depth = 0.007
        self.bezier.data.bevel_resolution = 10
        # increase the resolution u
        self.bezier.data.resolution_u = 64

    def slightly_randomize(self, point, planar=True, max_offset=0.3):
        # Slightly displaces the position of a point (for randomization in rope configurations)
        offset_x = np.random.uniform(0, max_offset)
        offset_y  = np.random.uniform(0, max_offset)
        offset_z  = np.random.uniform(0, max_offset)
        if np.random.uniform() < 0.5:
            offset_x *= -1
        if np.random.uniform() < 0.5:
            offset_y *= -1
        if np.random.uniform() < 0.5:
            offset_z *= -1
        point.co.x += offset_x
        point.co.y += offset_y
        if not planar:
            point.co.z += offset_z
        return offset_x, offset_y, offset_z

    def add_rope_asymmetry(self):
        '''
        Add sphere, to break symmetry of the rope
        '''
        bpy.ops.mesh.primitive_uv_sphere_add(location=(self.origin[0], self.origin[1], self.origin[2]))
        if self.sphere_radius is not None:
            sphere_radius = self.sphere_radius
        else:
            sphere_radius = np.random.uniform(0.0002, 0.0002)
        bpy.ops.transform.resize(value=(sphere_radius, sphere_radius, sphere_radius))
        bpy.ops.transform.rotate(value= pi / 2, orient_axis='X')
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.join()
        self.rope_asymm = bpy.context.active_object
        self.rope_asymm.name= self.rope_asymm_name
        # add a material to the sphere to set color to white
        mat = bpy.data.materials.new(name="Rope-Asymmetry")
        self.rope_asymm.data.materials.append(mat)
        # mat.base_color = (1, 1, 1)

    def gen_random_knot(self, offset_min, offset_max):
        #TODO: write function to randomly place each point rather than specific places or semi-random locations
        p0 = np.random.choice(range(3, len(self.bezier_points) - 7))
        offset_avg = np.random.uniform(offset_min, offset_max)

        self.bezier_points[p0 + 1].co.y += offset_avg
        # increase z of p0 + 1
        cable_height = 0.025
        self.bezier_points[p0 - 1].co.z += cable_height
        # self.bezier_points[p0 + 1].co.z += cable_height*2
        self.bezier_points[p0].co.z -= cable_height
        self.bezier_points[p0 + 1].co.x += 0.1
        self.bezier_points[p0 + 1].co.z += cable_height
        self.bezier_points[p0 + 2].co.y += offset_avg - 0.1
        self.bezier_points[p0 + 2].co.x = self.bezier_points[p0].co.x - 0.1
        self.bezier_points[p0 + 3].co.x = self.bezier_points[p0 + 2].co.x
        # self.bezier_points[p0 + 4].co.z -= cable_height*2
        avg1 = (self.bezier_points[p0 + 1].co.x + self.bezier_points[p0 + 2].co.x)/2 
        self.bezier_points[p0 + 4].co.x = np.random.uniform(avg1 - 0.01, avg1 + 0.01)
        self.bezier_points[p0 + 4].co.y -= 0.04
        avg2 = (self.bezier_points[p0 + 1].co.x + self.bezier_points[p0 + 2].co.x)/2
        self.bezier_points[p0 + 5].co.x = np.random.uniform(avg2 - 0.01, avg2 + 0.01)
        self.bezier_points[p0 + 5].co.y = 0.25
        self.bezier_points[p0 + 6].co.x = 0.0
        self.bezier_points[p0 + 6].co.y = 0.25
        self.bezier_points[p0 + 6].co.z -= cable_height
        
        return set(range(p0, p0 + 5))

    def make_overhand_knot(self, offset_min, offset_max):
        # Geometrically arrange the bezier points into a loop, and slightly randomize over node positions for variety
        #    2_______1
        #     \  4__/ 
        #      \ | /\
        #       \5/__\____________
        #       / \   | 
        #______0   3__|  
        p0 = 4 # np.random.choice(range(4, len(self.bezier_points) - 5))
        offset_avg = (offset_min + offset_max)/2

        self.bezier_points[p0 + 1].co.y += offset_avg
        # increase z of p0 + 1
        cable_height = 0.025
        self.bezier_points[p0 - 1].co.z += cable_height
        # self.bezier_points[p0 + 1].co.z += cable_height*2
        self.bezier_points[p0].co.z -= cable_height
        self.bezier_points[p0 + 1].co.x += 0.1
        self.bezier_points[p0 + 1].co.z += cable_height
        self.bezier_points[p0 + 2].co.y += offset_avg - 0.1
        self.bezier_points[p0 + 2].co.x = self.bezier_points[p0].co.x - 0.1
        self.bezier_points[p0 + 3].co.x = self.bezier_points[p0 + 2].co.x
        # self.bezier_points[p0 + 4].co.z -= cable_height*2
        self.bezier_points[p0 + 4].co.x = (self.bezier_points[p0 + 1].co.x + self.bezier_points[p0 + 2].co.x)/2
        self.bezier_points[p0 + 4].co.y -= 0.04
        self.bezier_points[p0 + 5].co.x = (self.bezier_points[p0 + 1].co.x + self.bezier_points[p0 + 2].co.x)/2
        self.bezier_points[p0 + 5].co.y = 0.25
        self.bezier_points[p0 + 6].co.x = 0.0
        self.bezier_points[p0 + 6].co.y = 0.25
        self.bezier_points[p0 + 6].co.z -= cable_height

        # self.bezier_points[p0 + 2].co.x = self.bezier_points[p0 + 1].co.x - 0.5
        # self.bezier_points[p0 + 3].co.y = self.bezier_points[p0 + 2].co.y
        # self.bezier_points[p0 + 3].co.x = self.bezier_points[p0 + 1].co.x - 1
        # self.bezier_points[p0 + 4].co.y = self.bezier_points[p0 + 2].co.y
        # self.bezier_points[p0 + 4].co.x = self.bezier_points[p0 + 1].co.x - 1
        
        return set(range(p0, p0 + 5))

    def make_distractor_cables(self, n=2):
        # Create a new bezier curve
        for k in range(n):
            # set the distractor cable points
            center = np.random.uniform(-0.35, 0.35, size=(2,))
            z = np.random.choice([0.075, -0.075])

            bpy.ops.curve.primitive_bezier_curve_add(location=(center[0], center[1], 0))
            bezier_scale = 1 # np.random.uniform(2.85,3.02)
            bpy.ops.transform.resize(value=(bezier_scale, bezier_scale, bezier_scale))
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.curve.subdivide(number_cuts=4)
            bpy.ops.transform.resize(value=(1, 0, 1))
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.curve.select_all(action='SELECT')
            bpy.ops.curve.handle_type_set(type='AUTOMATIC')
            bpy.ops.object.mode_set(mode='OBJECT')
            bezier = bpy.context.active_object
            bezier_points = bezier.data.splines[0].bezier_points
            bezier.name = "Bezier_Distract_{}".format(k)
            bezier.select_set(False)

            # set geometry bevel with depth of 0.01
            bezier.data.bevel_depth = 0.007
            bezier.data.bevel_resolution = 10
            # increase the resolution u
            bezier.data.resolution_u = 64

            theta = np.random.uniform(0, 2*pi)
            cable_length = 1.2
            noise_amount = 0.05
            for i in range(len(bezier_points)):
                bezier_points[i].co.x = center[0] + cable_length*np.cos(theta)*(i - len(bezier_points)/2)/len(bezier_points) + np.random.uniform(-noise_amount, noise_amount)
                bezier_points[i].co.y = center[1] + cable_length*np.sin(theta)*(i - len(bezier_points)/2)/len(bezier_points) + np.random.uniform(-noise_amount, noise_amount)
                bezier_points[i].co.z = z

    def randomize_nodes(self, num, offset_min, offset_max, nonplanar=False, offlimit_indices=set()):
        # Simulating pulling NUM nodes on the rope by (offset_min, offset_max) amount; nonplanar indicates whether upward pulls are allowed, and offlimit_indices specifies Bezier knots that should not be touched (For instance, if you made a loop and wanted to randomize the remaining nodes on the rope)
        knots_idxs = np.random.choice(list(set(range(len(self.bezier_points))) ^ offlimit_indices), min(num, len(self.bezier_points)), replace=False)
        for idx in knots_idxs:
            knot = self.bezier_points[idx]
            offset_y = random.uniform(offset_min, offset_max)
            offset_x = random.uniform(offset_min/2, offset_max/2)
            if random.uniform(0, 1) < 0.5:
                offset_y *= -1
            if random.uniform(0, 1) < 0.5:
                offset_x *= -1
            if nonplanar:
                offset_z = random.uniform(offset_min, offset_max)
                if random.uniform(0, 1) < 0.5:
                    offset_z *= -1
                knot.co.z += offset_z
            res_y = knot.co.y + offset_y
            res_x = knot.co.x + offset_x
            knot.co.y = res_y
            knot.co.x = res_x

    def reposition_camera(self, curve_vertices):
        # Orient camera towards the rope
        bpy.context.scene.camera = self.camera
        bpy.ops.view3d.camera_to_view_selected()
        self.camera.location.x = (curve_vertices[5].co.x + curve_vertices[6].co.x)/2 + np.random.uniform(-0.04, 0.04)
        self.camera.location.y = (curve_vertices[5].co.y + curve_vertices[6].co.y)/2 + np.random.uniform(-0.04, 0.04)
        # self.camera.location.z += np.random.uniform(3.3, 3.6)

    def render_single_scene(self, M_pix=20, M_depth=0.2):
		# Produce a single image of the current scene, save_rgb the mesh vertex pixel coords
        scene = bpy.context.scene
        # Dependency graph used to get mesh vertex coords after deformation (Blender's way of tracking these coords)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        rope_deformed = self.rope_asymm.evaluated_get(depsgraph)
        # Get rope mesh vertices in world space
        # get the bezier points as the coords
        coords = [p.co for p in self.bezier_points]
        print("%d Vertices" % len(coords))
        pixels = {}
        scene.render.resolution_percentage = 100
        scene.render.resolution_x = 200
        scene.render.resolution_y = 200

        for i in range(len(coords)):
            coord = coords[i]
            camera_coord = bpy_extras.object_utils.world_to_camera_view(scene, self.camera, coord)
            render_scale = scene.render.resolution_percentage / 100
            render_size = (
                int(scene.render.resolution_x * render_scale),
                int(scene.render.resolution_y * render_scale),
            )
            p = (round(camera_coord.x * render_size[0]), round(render_size[1] - camera_coord.y * render_size[1]))
            pixels[i] = [p, camera_coord]

        pixels_raw = {i: [pixels[i][0]] for i in pixels}
        
        filename = "{0:06d}_rgb.png".format(self.i)
        if self.save_rgb:
            scene.world.color = (0, 0, 0)
            # scene.render.display_mode
            scene.render.engine = 'BLENDER_WORKBENCH'
            # scene.display_settings.display_device = 'None'
            scene.sequencer_colorspace_settings.name = 'XYZ'
            scene.render.image_settings.file_format='PNG'
            scene.render.filepath = "./images/{}".format(filename)
            bpy.ops.render.render(use_viewport = True, write_still=True)

            saved_img = plt.imread(scene.render.filepath)
            vis = False
            if vis:
                plt.imshow(saved_img)
                for i in range(len(coords)):
                    plt.scatter(pixels_raw[i][0][0], pixels_raw[i][0][1], c='r', s=1)
                plt.show()
            
            self.knots_info[self.i] = pixels_raw
            np.save('./annotated/{}'.format(filename.replace('.png', '.npy')),
                {'img': saved_img, 'pixels': pixels_raw, 'condition': random.choice(([3, 5, 6], [10, 6, 5]))})
        self.i += 1
    
    def generate_random_configuration(self):
        pass

    def get_graph_from_bezier_curve(self):
        """Steps for extracting the graph from the bezier curve.
        
        all_points = `mathutils.geometry.interpolate_bezier` on whole bezier curve
        intersect_grid = 2d grid of intersection info between all pairs of curve segments
        cur_edge_start = (Node())
        cur_edge_height = over
        for point in all_points:
            if point is within a radius of a point from another bezier segment:
                if intersect_grid contig is false:
                    add point as node to graph
                intersect_grid contig = true
            else:
                intersect_grid contig = false
        """

        # all_points_3d_contig = []
        # for i, pt in enumerate(self.bezier_points):
        #     print(pt.)
            # bpy.mathutils.geometry.interpolate_bezier(self.bezier_points[i])


    def run(self):
        # Create new images folder to dump rendered images
        if not os.path.exists("./images"):
            os.makedirs('./images')
        else:
            os.system('rm -rf ./images')
            os.makedirs('./images')
        for i in range(self.num_images):
            x = time.time()
            self.clear()
            self.add_camera(fixed=True)
            self.make_bezier()
            self.add_rope_asymmetry()
            # self.make_simple_loop(0.3, 0.3)
            # self.make_overhand_knot(0.3, 0.3)
            self.gen_random_knot(0.05, 0.4)
            self.randomize_nodes(3, 0.05, 0.05, False)
            self.make_distractor_cables(n=np.random.randint(1, 3))

            self.reposition_camera(self.bezier_points)
            self.render_single_scene(M_pix=10)

            self.get_graph_from_bezier_curve()
            print("Total time for scene {}s.".format(str((time.time() - x) % 60)))
        # if self.save_depth or self.save_rgb:
            # with open("./images/knots_info.json", 'w') as outfile:
            #     json.dump(self.knots_info, outfile, sort_keys=True, indent=2)

if __name__ == '__main__':
    with open("params.json", "r") as f:
        rope_params = json.load(f)
        rope_params['num_images'] = 10
    renderer = RopeRenderer(save_depth=rope_params["save_depth"], save_rgb=(not rope_params["save_depth"]), num_images = rope_params["num_images"], coord_offset=rope_params["coord_offset"], bezier_knots=rope_params["bezier_knots"])
    renderer.run()
