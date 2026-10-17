# Hollow brick walls placed on a concrete slab by its tagged top face:
# a procedural brick albedo with a normal map derived from the brick
# height, a noise-based concrete, and three knobs, each handed whole
# to one param so the seed rerolls and the factors slide live.
brick_seed = mo.param("brick_seed", 4, min = 0, max = 99, step = 1, label = "Brick variation seed")
brick_relief = mo.param("brick_relief", 0.25, min = 0.0, max = 1.0, step = 0.05, label = "Brick relief")
wall_roughness = mo.param("wall_roughness", 0.85, min = 0.0, max = 1.0, step = 0.05, label = "Wall roughness")

slab = mo.box(id = "sl01", size = [2.4, 0.12, 1.8])
slab = mo.transform(slab, id = "sl02", translate = [0.0, 0.06, 0.0])
slab = mo.tag(slab, id = "sl04", name = "slab_top", select = {"normal_y": {"gt": 0.9}})
concrete = mo.noise_texture(id = "cn01", seed = 11, size = 256, scale = 6, octaves = 5, low = 0.35, high = 0.75)
concrete = mo.colorize(concrete, id = "cn02", color_low = [0.28, 0.28, 0.27], color_high = [0.52, 0.52, 0.5])
concrete_rough = mo.noise_texture(id = "cn03", seed = 12, size = 256, scale = 8, octaves = 4, low = 0.55, high = 0.95)
base = mo.material(
    mo.unwrap(slab, id = "sl03"),
    base_color_texture = concrete, roughness_texture = concrete_rough,
    id = "mt01", name = "concrete",
)

walls = mo.boolean(
    mo.box(id = "wl01", size = [2.0, 1.2, 1.4]),
    mo.transform(mo.box(id = "wl02", size = [1.76, 1.2, 1.16]), id = "wl03", translate = [0.0, 0.12, 0.0]),
    id = "wl04", mode = "difference",
)
walls = mo.place(base = slab, part = walls, id = "pl01", anchor = "slab_top")
walls = mo.unwrap(mo.smooth_normals(walls, id = "sn01"), id = "wl06")
brick = mo.brick_texture(id = "bc01", seed = brick_seed, size = 512, rows = 8, columns = 4, variation = 0.25)
relief = mo.brick_texture(id = "bh01", size = 512, rows = 8, columns = 4, map = "height")
relief = mo.normal_map(relief, id = "bn01", strength = brick_relief)
walls = mo.material(walls, base_color_texture = brick, normal_texture = relief, id = "mt02", name = "brick", roughness = wall_roughness)

mo.output("base", base)
mo.output("walls", walls)
mo.assert_contact(id = "walls_on_slab", outputs = ["base", "walls"])
