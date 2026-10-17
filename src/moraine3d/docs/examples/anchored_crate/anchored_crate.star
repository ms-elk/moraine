# A crate placed on a pallet's tagged deck and a smaller crate on its
# lid: place constructs each rest, contact declares it.
pallet = mo.box(id = "pl01", size = [1.2, 0.14, 0.9])
pallet = mo.transform(pallet, id = "pl02", translate = [0.0, 0.07, 0.0])
pallet = mo.tag(pallet, id = "pl03", name = "deck", select = {"normal_y": {"gt": 0.9}})
crate = mo.place(base = pallet, part = mo.box(id = "cr01", size = [0.6, 0.5, 0.6]), id = "cr02", anchor = "deck")
crate = mo.tag(crate, id = "cr03", name = "lid", select = {"normal_y": {"gt": 0.9}})
small = mo.place(base = crate, part = mo.box(id = "sm01", size = [0.35, 0.3, 0.35]), id = "sm02", anchor = "lid", offset = [0.1, 0.0, 0.0])

mo.output("pallet", pallet)
mo.output("crate", crate)
mo.output("crate_small", small)
mo.assert_contact(id = "stack_rests", outputs = ["crate", "crate_small"])
