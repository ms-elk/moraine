# A revolved column profile with its capital tagged on the prototype,
# a colonnade of four, and an architrave placed on the first capital
# and asserted to rest on every column.
PROFILE = [
    [0.3, 0.0], [0.3, 0.05], [0.24, 0.09], [0.2, 0.13], [0.16, 0.18],
    [0.13, 2.2], [0.18, 2.26], [0.24, 2.3], [0.24, 2.36],
]
PITCH = 1.5

column = mo.revolve(id = "rv01", profile = PROFILE, segments = 48)
column = mo.tag(column, id = "tg01", name = "capital", select = {"normal_y": {"gt": 0.9}, "center_y": {"gt": 2.3}})
colonnade = mo.array(column, id = "ar01", count = 4, step_translate = [PITCH, 0.0, 0.0])
beam = mo.box(id = "bm01", size = [5.1, 0.3, 0.5])
architrave = mo.place(base = colonnade, part = beam, id = "pl01", anchor = "capital", instance = 0, offset = [2.25, 0.0, 0.0])

mo.output("colonnade", colonnade)
mo.output("architrave", architrave)
mo.assert_contact(id = "architrave_rests", outputs = ["colonnade", "architrave"], every = ["colonnade"])
