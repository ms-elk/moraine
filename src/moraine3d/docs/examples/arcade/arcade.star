# A wall with three semicircular arches and a coping band, one
# watertight mesh: each arch is a box and a half-cylinder unioned,
# cut from the wall, and the coping unioned on top.
WALL = [6.0, 3.2, 0.4]
ARCH_W = 1.2
ARCH_SPAN = 1.5  # the straight part, below the half-round
PITCH = 1.9

wall = mo.box(id = "wl01", size = WALL)
wall = mo.transform(wall, id = "wt01", translate = [0.0, WALL[1] / 2, 0.0])

opening = mo.box(id = "bc01", size = [ARCH_W, ARCH_SPAN, 0.7])
opening = mo.transform(opening, id = "bt01", translate = [0.0, ARCH_SPAN / 2, 0.0])
head = mo.cylinder(id = "cy01", radius = ARCH_W / 2, height = 0.7, segments = 48)
head = mo.transform(head, id = "cy02", translate = [0.0, ARCH_SPAN, 0.0], rotate_deg = [90.0, 0.0, 0.0])
arch = mo.boolean(opening, head, id = "ac01", mode = "union")
arches = [
    mo.transform(arch, id = "at0%d" % (i + 1), translate = [x, 0.0, 0.0])
    for i, x in enumerate([-PITCH, 0.0, PITCH])
]
pierced = mo.boolean(wall, id = "cut1", mode = "difference", *arches)

coping = mo.box(id = "cp01", size = [6.2, 0.15, 0.5])
coping = mo.transform(coping, id = "ct01", translate = [0.0, 3.275, 0.0])
mo.output("arcade", mo.boolean(pierced, coping, id = "un01", mode = "union"))
