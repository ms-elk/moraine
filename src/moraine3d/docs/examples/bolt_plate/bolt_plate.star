# A round plate with eight bolt holes: a rotational array of one bolt,
# baked to a mesh so the boolean can take it as a cutter.
plate = mo.cylinder(id = "pl01", radius = 0.25, height = 0.04, segments = 48)
plate = mo.transform(plate, id = "pl02", translate = [0.0, 0.02, 0.0])
bolt = mo.cylinder(id = "bt01", radius = 0.02, height = 0.08, segments = 16)
bolt = mo.transform(bolt, id = "bt02", translate = [0.18, 0.02, 0.0])
bolts = mo.array(bolt, id = "ar01", count = 8, step_rotate_deg = [0.0, 45.0, 0.0])
cutter = mo.bake_instances(bolts, id = "bk01")
mo.output("plate", mo.boolean(plate, cutter, id = "bl01", mode = "difference"))
