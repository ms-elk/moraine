# A duct mated face to face to a wall port, with a flange riding the
# same solve as a leader: mate constructs the joint, mated declares it
# from the evaluated faces, and contact says the parts touch.
wall = mo.box(id = "wl01", size = [4.0, 3.0, 0.2])
wall = mo.transform(wall, id = "wl02", translate = [0.0, 1.5, 0.0])
port_cut = mo.transform(mo.box(id = "pt01", size = [0.5, 0.7, 0.6]), id = "pt02", translate = [0.8, 1.8, 0.0])
wall = mo.boolean(wall, port_cut, id = "wl03", mode = "difference")
wall = mo.tag(wall, id = "wl04", name = "port", select = {"normal_z": {"gt": 0.9}, "center_z": {"gt": 0.09}})

duct = mo.boolean(
    mo.box(id = "dt01", size = [0.6, 0.8, 1.0]), mo.box(id = "dt02", size = [0.5, 0.7, 1.4]),
    id = "dt03", mode = "difference",
)
duct = mo.tag(duct, id = "dt04", name = "dock", select = {"normal_z": {"lt": -0.9}, "center_z": {"lt": -0.49}})
docked = mo.mate(base = wall, part = duct, id = "dt05", anchor = "port", face = "dock")

flange = mo.boolean(
    mo.box(id = "fl01", size = [0.8, 1.0, 0.04]), mo.box(id = "fl02", size = [0.6, 0.8, 0.2]),
    id = "fl03", mode = "difference",
)
flange = mo.transform(flange, id = "fl04", translate = [0.0, 0.0, -0.48])
flange = mo.mate(base = wall, part = flange, leader = duct, id = "fl05", anchor = "port", face = "dock")

mo.output("wall", wall)
mo.output("duct", docked)
mo.output("flange", flange)
mo.assert_contact(id = "duct_meets_wall", outputs = ["wall", "duct"])
mo.assert_contact(id = "flange_meets_wall", outputs = ["wall", "flange"])
