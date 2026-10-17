# A two-story brick block: a box with the roof, chimney, string course
# and plinth unioned on, the window and door openings cut, then the
# door lintel and step unioned; sills and lintels are instanced rows
# seated into the wall and declared as embeds.
BODY = [6.0, 6.6, 4.0]
FRONT = 2.0  # the facade plane
WINDOW = [1.1, 1.5, 0.7]
GROUND_SILL_Y = 0.855
UPPER_SILL_Y = 3.955
GROUND_LINTEL_Y = 2.46
UPPER_LINTEL_Y = 5.56

body = mo.box(id = "bd01", size = BODY)
body = mo.transform(body, id = "bd02", translate = [0.0, BODY[1] / 2, 0.0])
roof = mo.transform(mo.box(id = "rf01", size = [6.5, 0.25, 4.5]), id = "rt01", translate = [0.0, 6.725, 0.0])
chimney = mo.transform(mo.box(id = "ch01", size = [0.7, 1.2, 0.7]), id = "ct01", translate = [-1.9, 7.15, -1.2])
cap = mo.transform(mo.box(id = "cc01", size = [0.85, 0.12, 0.85]), id = "cx01", translate = [-1.9, 7.7, -1.2])
course = mo.transform(mo.box(id = "sc01", size = [6.2, 0.15, 4.2]), id = "st01", translate = [0.0, 3.55, 0.0])
plinth = mo.transform(mo.box(id = "pl01", size = [6.16, 0.45, 4.16]), id = "pt01", translate = [0.0, 0.225, 0.0])
shell = mo.boolean(body, roof, chimney, cap, course, plinth, id = "sh01", mode = "union")

window = mo.box(id = "wc01", size = WINDOW)
windows = [
    mo.transform(window, id = "wt0%d" % (i + 1), translate = [x, y, FRONT])
    for i, (x, y) in enumerate([[-1.9, 1.65], [1.9, 1.65], [-1.9, 4.75], [0.0, 4.75], [1.9, 4.75]])
]
door = mo.transform(mo.box(id = "dc01", size = [1.3, 2.5, 0.7]), id = "dt01", translate = [0.0, 1.15, FRONT])
pierced = mo.boolean(shell, id = "cut1", mode = "difference", *(windows + [door]))

lintel = mo.box(id = "ln01", size = [1.3, 0.12, 0.1])
door_lintel = mo.transform(lintel, id = "dl01", translate = [0.0, GROUND_LINTEL_Y, FRONT])
door_step = mo.transform(mo.box(id = "ds01", size = [1.5, 0.12, 0.5]), id = "dx01", translate = [0.0, 0.06, 2.2])
building = mo.boolean(pierced, door_lintel, door_step, id = "un01", mode = "union")

sill = mo.box(id = "sl01", size = [1.3, 0.09, 0.14])
sills_ground = mo.array(
    mo.transform(sill, id = "sg01", translate = [-1.9, GROUND_SILL_Y, FRONT]),
    id = "sa01", count = 2, step_translate = [3.8, 0.0, 0.0],
)
sills_upper = mo.array(
    mo.transform(sill, id = "su01", translate = [-1.9, UPPER_SILL_Y, FRONT]),
    id = "sa02", count = 3, step_translate = [1.9, 0.0, 0.0],
)
lintels_ground = mo.array(
    mo.transform(lintel, id = "lg01", translate = [-1.9, GROUND_LINTEL_Y, FRONT]),
    id = "la01", count = 2, step_translate = [3.8, 0.0, 0.0],
)
lintels_upper = mo.array(
    mo.transform(lintel, id = "lu01", translate = [-1.9, UPPER_LINTEL_Y, FRONT]),
    id = "la02", count = 3, step_translate = [1.9, 0.0, 0.0],
)

mo.output("building", building)
mo.output("sills_ground", sills_ground)
mo.output("sills_upper", sills_upper)
mo.output("lintels_ground", lintels_ground)
mo.output("lintels_upper", lintels_upper)
# Each row seats 2 to 6 cm into the wall: the overlap is the construction.
mo.assert_embedded(id = "ground_sills_seat", outputs = ["building", "sills_ground"], min = 0.02, max = 0.06)
mo.assert_embedded(id = "upper_sills_seat", outputs = ["building", "sills_upper"], min = 0.02, max = 0.06)
mo.assert_embedded(id = "ground_lintels_seat", outputs = ["building", "lintels_ground"], min = 0.02, max = 0.06)
mo.assert_embedded(id = "upper_lintels_seat", outputs = ["building", "lintels_upper"], min = 0.02, max = 0.06)
