# A straight flight: twelve instanced steps and two stringers, each a
# tilted plank cut flush to the ground, bearing the steps on their
# side faces.
RISE = 0.175
RUN = 0.28
WIDTH = 1.0
COUNT = 12
STRINGER_T = 0.06

step = mo.box(id = "st01", size = [WIDTH, RISE, RUN])
step = mo.transform(step, id = "sp01", translate = [0.0, RISE / 2, -RUN / 2])
steps = mo.array(step, id = "sa01", count = COUNT, step_translate = [0.0, RISE, -RUN])

plank = mo.box(id = "sg01", size = [STRINGER_T, 0.28, 3.9])
plank = mo.transform(plank, id = "sr01", translate = [-0.53, 0.9, -1.68], rotate_deg = [32.0, 0.0, 0.0])
ground = mo.box(id = "gb01", size = [0.2, 0.6, 4.6])
ground = mo.transform(ground, id = "gt01", translate = [-0.53, -0.3, -1.68])
stringer = mo.boolean(plank, ground, id = "cs01", mode = "difference")
stringers = mo.array(stringer, id = "sb01", count = 2, step_translate = [1.06, 0.0, 0.0])

mo.output("steps", steps)
mo.output("stringers", stringers)
