# A water tower: a tank with a hemispherical dome on four tilted legs
# in a rotational array, two rings of braces, and a ladder of two rails
# and instanced rungs. The tank radius is the one knob.
tank_radius = mo.param("tank_radius", 1.0, min = 0.6, max = 1.6)
LEG_FOOT = 0.707  # the legs stand on a square of side 1.414 m

tank = mo.cylinder(id = "tk01", radius = tank_radius, height = 1.5, segments = 48)
tank = mo.transform(tank, id = "tk02", translate = [0.0, 4.75, 0.0])
dome = mo.transform(mo.sphere(id = "dm01", radius = 1.0, segments = 48, rings = 24), id = "dm02", translate = [0.0, 5.5, 0.0])
upper_half = mo.transform(mo.box(id = "db01", size = [2.2, 1.1, 2.2]), id = "db02", translate = [0.0, 6.05, 0.0])
dome = mo.boolean(dome, upper_half, id = "dm03", mode = "intersection")
tank = mo.boolean(tank, dome, id = "tk03", mode = "union")

leg = mo.box(id = "lg01", size = [0.12, 4.3, 0.12])
leg = mo.transform(leg, id = "lg02", translate = [LEG_FOOT, 2.05, LEG_FOOT], rotate_deg = [-3.8, 0.0, 3.8])
foot_cut = mo.transform(mo.box(id = "lc01", size = [0.8, 0.4, 0.8]), id = "lc02", translate = [LEG_FOOT, -0.2, LEG_FOOT])
leg = mo.boolean(leg, foot_cut, id = "lg04", mode = "difference")
legs = mo.array(leg, id = "lg03", count = 4, step_rotate_deg = [0.0, 90.0, 0.0])

brace_low = mo.transform(mo.box(id = "bl01", size = [1.62, 0.08, 0.08]), id = "bl02", translate = [0.0, 1.5, 0.75])
brace_low = mo.array(brace_low, id = "bl03", count = 4, step_rotate_deg = [0.0, 90.0, 0.0])
brace_high = mo.transform(mo.box(id = "bh01", size = [1.42, 0.08, 0.08]), id = "bh02", translate = [0.0, 3.0, 0.652])
brace_high = mo.array(brace_high, id = "bh03", count = 4, step_rotate_deg = [0.0, 90.0, 0.0])

rail = mo.transform(mo.box(id = "rl01", size = [0.04, 4.5, 0.04]), id = "rl02", translate = [-0.18, 2.25, 0.95])
rails = mo.array(rail, id = "rl03", count = 2, step_translate = [0.36, 0.0, 0.0])
rung = mo.transform(mo.box(id = "rg01", size = [0.4, 0.03, 0.03]), id = "rg02", translate = [0.0, 0.3, 0.95])
rungs = mo.array(rung, id = "rg03", count = 14, step_translate = [0.0, 0.3, 0.0])

mo.output("tank", tank)
mo.output("legs", legs)
mo.output("brace_low", brace_low)
mo.output("brace_high", brace_high)
mo.output("rails", rails)
mo.output("rungs", rungs)
# The braces overlap into one ring and into the legs: the construction, declared.
mo.assert_embedded(id = "low_braces_ring", outputs = ["brace_low"], connected = True, min = 0.01, max = 0.05)
mo.assert_embedded(id = "high_braces_ring", outputs = ["brace_high"], connected = True, min = 0.01, max = 0.05)
mo.assert_embedded(id = "low_braces_tie_legs", outputs = ["brace_low", "legs"], min = 0.02, max = 0.05)
mo.assert_embedded(id = "high_braces_tie_legs", outputs = ["brace_high", "legs"], min = 0.02, max = 0.05)
mo.assert_embedded(id = "legs_root_in_tank", outputs = ["legs", "tank"])
mo.assert_embedded(id = "rungs_thread_rails", outputs = ["rails", "rungs"], min = 0.005, max = 0.02)
mo.assert_embedded(id = "rails_root_in_tank", outputs = ["rails", "tank"])
mo.assert_embedded(id = "top_rungs_reach_tank", outputs = ["rungs", "tank"])
