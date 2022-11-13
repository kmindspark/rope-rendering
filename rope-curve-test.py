from bpy import context, data, ops

# Create a bezier circle and enter edit mode.
ops.curve.primitive_bezier_circle_add(radius=1.0,
                                      location=(0.0, 0.0, 0.0),
                                      enter_editmode=True)

# Subdivide the curve by a number of cuts, giving the
# random vertex function more points to work with.
ops.curve.subdivide(number_cuts=16)

# Randomize the vertices of the bezier circle.
# offset [-inf .. inf], uniform [0.0 .. 1.0],
# normal [0.0 .. 1.0], RNG seed [0 .. 10000].
ops.transform.vertex_random(offset=1.0, uniform=0.1, normal=0.0, seed=0)

# Scale the curve while in edit mode.
ops.transform.resize(value=(2.0, 2.0, 3.0))

# Return to object mode.
ops.object.mode_set(mode='OBJECT')

# Store a shortcut to the curve object's data.
obj_data = context.active_object.data

# Which parts of the curve to extrude ['HALF', 'FRONT', 'BACK', 'FULL'].
obj_data.fill_mode = 'FULL'

# Breadth of extrusion.
obj_data.extrude = 0.125

# Depth of extrusion.
obj_data.bevel_depth = 0.125

# Smoothness of the segments on the curve.
obj_data.resolution_u = 20
obj_data.render_resolution_u = 32

# Create bevel control curve.
ops.curve.primitive_bezier_circle_add(radius=0.25, enter_editmode=True)
ops.curve.subdivide(number_cuts=4)
ops.transform.vertex_random(offset=1.0, uniform=0.1, normal=1.0, seed=0)
bevel_control = context.active_object
bevel_control.data.name = bevel_control.name = 'Bevel Control'

# Set the main curve's bevel control to the bevel control curve.
obj_data.bevel_object = bevel_control
ops.object.mode_set(mode='OBJECT')

# Create taper control curve.
ops.curve.primitive_bezier_curve_add(enter_editmode=True)
ops.curve.subdivide(number_cuts=3)
ops.transform.vertex_random(offset=1.0, uniform=0.1, normal=1.0, seed=0)
taper_control = context.active_object
taper_control.data.name = taper_control.name = 'Taper Control'

# Set the main curve's taper control to the taper control curve.
obj_data.taper_object = taper_control
ops.object.mode_set(mode='OBJECT')

from math import cos, pi, sin, tan
from random import TWOPI, randint, uniform

ops.curve.primitive_bezier_circle_add(enter_editmode=True)
ops.curve.subdivide(number_cuts=18)

# Cache a reference to the curve.
curve = context.active_object

# Locate the array of bezier points.
bez_points = curve.data.splines[0].bezier_points

sz = len(bez_points)
i_to_theta = TWOPI / sz
for i in range(0, sz, 1):
    # Set every sixth coordinate's z to 0.5.
    if i % 6 == 0:
        bez_points[i].co.z = 0.5

    if i % 2 == 0:
        bez_points[i].handle_right *= 2.0
        bez_points[i].handle_left *= 0.5
    elif i % 4 == 0:
        bez_points[i].handle_right.z -= 5.0
        bez_points[i].handle_left.z += 5.0
    else:
        bez_points[i].co *= 0.5

    # Shift cos(t) from -1 .. 1 to 0 .. 4.
    scalar = 2.0 + 2.0 * cos(i * i_to_theta)

    # Multiply coordinate by cos(t).
    bez_points[i].co *= scalar

# Resize within edit mode.
ops.transform.resize(value=(3.0, 3.0, 1.0))

# Return to object mode.
ops.object.mode_set(mode='OBJECT')

# Convert from a curve to a mesh.
ops.object.convert(target='MESH')

# Append modifiers.
skin_mod = curve.modifiers.new(name='Skin', type='SKIN')
subsurf_mod = curve.modifiers.new(name='Subsurf', type='SUBSURF')
stretch_mod = curve.modifiers.new(name='SimpleDeform', type='SIMPLE_DEFORM')

# Adjust modifier options.
skin_mod.use_smooth_shade = True
subsurf_mod.levels = 3
subsurf_mod.render_levels = 3
stretch_mod.deform_method = 'STRETCH'
stretch_mod.factor = 0.5