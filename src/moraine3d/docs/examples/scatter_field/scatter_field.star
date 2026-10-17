# A displaced grid extruded and cut into a ground slab, with forty
# rocks scattered on the displaced sheet: every rock touches the
# ground, the sinks and the rock overlaps are declared free-overlap
# embeds, and the count is pinned.
SIDE = 8.0

relief = mo.grid(id = "gr01", size = [8.1, 8.1], segments = [40, 40])
relief = mo.transform(relief, id = "gr02", translate = [0.0, 0.7, 0.0])
relief = mo.noise_displace(relief, id = "gr03", amplitude = 0.15, frequency = 0.4, octaves = 3, seed = 3)
slab = mo.extrude(relief, id = "gr04", distance = 1.0, direction = [0.0, -1.0, 0.0])
trim = mo.transform(mo.box(id = "gr05", size = [SIDE, 2.0, SIDE], segments = [8, 1, 8]), id = "gr06", translate = [0.0, 1.0, 0.0])
ground = mo.boolean(slab, trim, id = "gr07", mode = "intersection")

rock = mo.sphere(id = "rk01", radius = 0.18, segments = 16, rings = 8)
rocks = mo.scatter(target = relief, prototype = rock, id = "rk02", count = 40, seed = 7, align_normal = True, rand_yaw = True, scale_jitter = [0.5, 1.4])

mo.output("ground", ground)
mo.output("rocks", rocks)
# A rock sinks as its seed puts it, and rocks may overlap each other:
# the free-overlap spellings, with no floor and no ceiling.
mo.assert_embedded(id = "rocks_sunk", outputs = ["rocks", "ground"], min = 0)
mo.assert_embedded(id = "rocks_may_touch", outputs = ["rocks"], self = True)
