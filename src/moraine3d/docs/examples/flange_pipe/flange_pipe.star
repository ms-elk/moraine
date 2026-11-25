# A through-bored pipe with two bolted flanges, one watertight solid:
# tube and flanges unioned, then the bore and eight bolt holes cut.
LENGTH = 1.0
SEGMENTS = 48
FLANGE_R = 0.17
FLANGE_T = 0.024
# Bolt-hole centers on the flange: radius 0.1475, every 45 degrees.
HOLES = [
    [0.1475, 0.0], [0.1043, 0.1043], [0.0, 0.1475], [-0.1043, 0.1043],
    [-0.1475, 0.0], [-0.1043, -0.1043], [0.0, -0.1475], [0.1043, -0.1043],
]

tube = mo.cylinder(id = "tb01", radius = 0.1095, height = LENGTH, segments = SEGMENTS)
tube = mo.transform(tube, id = "tt01", translate = [0.0, LENGTH / 2, 0.0])
flange = mo.cylinder(id = "fl01", radius = FLANGE_R, height = FLANGE_T, segments = SEGMENTS)
bottom = mo.transform(flange, id = "ft01", translate = [0.0, FLANGE_T / 2, 0.0])
top = mo.transform(flange, id = "ft02", translate = [0.0, LENGTH - FLANGE_T / 2, 0.0])
body = mo.boolean(tube, bottom, top, id = "un01", mode = "union")

bore = mo.cylinder(id = "bo01", radius = 0.0995, height = 1.2, segments = SEGMENTS)
bore = mo.transform(bore, id = "bt01", translate = [0.0, LENGTH / 2, 0.0])
hole = mo.cylinder(id = "hl01", radius = 0.0115, height = 1.2, segments = max(3, SEGMENTS // 3))
holes = [
    mo.transform(hole, id = "ht0%d" % (i + 1), translate = [x, LENGTH / 2, z])
    for i, (x, z) in enumerate(HOLES)
]
pipe = mo.boolean(body, bore, id = "cut1", mode = "difference", *holes)

# The walls of the bottom flange's bolt holes: vertical faces inside the
# flange's thickness. The rim is vertical there too and differs only in
# the size of its faces, so the bound is half a rim triangle, and both
# face sizes follow SEGMENTS.
RIM_FACE = math.pi * FLANGE_R / SEGMENTS * FLANGE_T
mo.output("pipe", mo.tag(pipe, id = "tg01", name = "bolt_holes", select = {
    "normal_y": {"gt": -0.1, "lt": 0.1},
    "center_y": {"lt": FLANGE_T},
    "area": {"lt": RIM_FACE / 2},
}))
