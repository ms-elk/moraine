# Three instanced pedestals and a crate placed on one of them by its
# copy index: the tag is read on the prototype, the copy is picked at
# the place.
pedestal = mo.box(id = "pd01", size = [0.6, 0.4, 0.6])
pedestal = mo.transform(pedestal, id = "pd02", translate = [0.0, 0.2, 0.0])
pedestal = mo.tag(pedestal, id = "pd03", name = "top", select = {"normal_y": {"gt": 0.9}})
pedestals = mo.array(pedestal, id = "pd04", count = 3, step_translate = [0.8, 0.0, 0.0])
crate = mo.place(base = pedestals, part = mo.box(id = "cr01", size = [0.4, 0.4, 0.4]), id = "cr02", anchor = "top", mode = "on_top", offset = [0.0, 0.0, 0.0], instance = 1)

mo.output("pedestals", pedestals)
mo.output("crate", crate)
mo.assert_contact(id = "crate_on_pedestal", outputs = ["pedestals", "crate"])
