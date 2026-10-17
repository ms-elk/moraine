# A checker on a box-unwrapped box, for reading the UV convention in
# the viewer: v down, (0, 0) at the texture's top-left.
SIZE = [1.0, 0.5, 0.5]

box = mo.box(id = "bx01", size = SIZE)
box = mo.transform(box, id = "tr01", translate = [0.0, SIZE[1] / 2, 0.0])
box = mo.unwrap(box, id = "uw01", mode = "box", uv_scale = 4.0)
checker = mo.checker_texture(id = "ck01", size = 256, tiles = 8)
mo.output("main", mo.material(
    box, base_color_texture = checker, id = "mt01", name = "uv_check", roughness = 0.9,
))
