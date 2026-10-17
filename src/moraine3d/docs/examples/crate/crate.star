# A 3 x 2 crate kit from one prototype: bevel, box unwrap, then two
# nested instanced arrays. The prototype is transformed before the
# arrays, so every copy stays an instance through export.
SIZE = [0.8, 0.5, 0.5]

crate = mo.box(id = "bx01", size = SIZE)
crate = mo.bevel(crate, id = "bv01", width = 0.02)
crate = mo.unwrap(crate, id = "uw01", mode = "box")
crate = mo.transform(crate, id = "tr01", translate = [0.0, SIZE[1] / 2, 0.0])
row = mo.array(crate, id = "ar01", count = 3, step_translate = [0.9, 0.0, 0.0])
mo.output("crates", mo.array(row, id = "ar02", count = 2, step_translate = [0.0, 0.55, 0.0]))
