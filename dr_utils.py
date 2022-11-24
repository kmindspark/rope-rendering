import bpy
import numpy as np
import os
from mathutils import Vector
import random
import sys
sys.path.append(os.getcwd())

def create_material(texture_filename):
    img = bpy.data.images.load(texture_filename)
    name = "texture"
    material = bpy.data.materials.new(name= name)
    material.use_nodes = True
    #create a reference to the material output
    material_output = material.node_tree.nodes.get('Material Output')
    Principled_BSDF = material.node_tree.nodes.get('Principled BSDF')                                 

    texImage_node = material.node_tree.nodes.new('ShaderNodeTexImage')
    texImage_node.image = img
    #set location of node
    material_output.location = (400, 20)
    Principled_BSDF.location = (-400, -500)
    texImage_node.location = (0, 0)

    material.node_tree.links.new(texImage_node.outputs[0], Principled_BSDF.inputs[0])
    material.node_tree.nodes["Principled BSDF"].inputs['Specular'].default_value = 0
    material.node_tree.nodes["Principled BSDF"].inputs['Roughness'].default_value = 0.5
    mat = bpy.data.materials.get(name)
    mat.blend_method = 'CLIP'
    return mat

def pattern(obj, texture_filename):
    '''Add image texture to object (don't create new materials, just overwrite the existing one if there is one)'''
    if False:
        if '%sTexture' % obj.name in bpy.data.materials: 
            mat = bpy.data.materials['%sTexture'%obj.name]
        else:
            mat = bpy.data.materials.new(name="%sTexture"%obj.name)
            mat.use_nodes = True
        if "Image Texture" in mat.node_tree.nodes:
            texImage = mat.node_tree.nodes["Image Texture"]
        else:
            texImage = mat.node_tree.nodes.new('ShaderNodeTexImage')
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        texImage.image = bpy.data.images.load(texture_filename)
        mat.node_tree.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])
        mat.specular_intensity = np.random.uniform(0, 0.3)
        mat.roughness = np.random.uniform(0.5, 1)
    else:
        mat = create_material(texture_filename)

    if not obj.data.materials:
        obj.data.materials.append(mat)
    else:
        obj.data.materials[0] = mat

def texture_randomize(obj, textures_folder):
    rand_img_path = random.choice(os.listdir(textures_folder))
    while rand_img_path == '.DS_Store':
        rand_img_path = random.choice(os.listdir(textures_folder))
    img_filepath = os.path.join(textures_folder, rand_img_path)
    pattern(obj, img_filepath)