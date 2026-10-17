# One box resting on the ground plane: the smallest asset.
box = mo.box(id = "bx01", size = [1.0, 1.0, 1.0])
mo.output("main", mo.transform(box, id = "tr01", translate = [0.0, 0.5, 0.0]))
