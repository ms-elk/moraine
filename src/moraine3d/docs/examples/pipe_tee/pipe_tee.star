# Two cylinders unioned into a tee, then smoothed by angle threshold:
# the union is flat-shaded until smooth_normals runs after it.
run = mo.cylinder(id = "cy01", radius = 0.12, height = 0.8, segments = 48)
branch = mo.cylinder(id = "cy02", radius = 0.09, height = 0.5, segments = 48)
branch = mo.transform(branch, id = "tr01", translate = [0.0, 0.0, 0.25], rotate_deg = [90.0, 0.0, 0.0])
tee = mo.boolean(run, branch, id = "bo01", mode = "union")
tee = mo.smooth_normals(tee, id = "sm01", angle_max_deg = 30.0)
mo.output("tee", mo.transform(tee, id = "tr02", translate = [0.0, 0.4, 0.0]))
