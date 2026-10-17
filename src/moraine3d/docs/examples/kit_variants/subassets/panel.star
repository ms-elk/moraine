# A painted panel with two knobs, each handed whole to one param: the
# wear seed rerolls, the roughness slides live.
wear_seed = mo.param("wear_seed", 7, min = 0, max = 99, step = 1, label = "Wear seed")
paint_roughness = mo.param("paint_roughness", 0.6, min = 0.2, max = 1.0, label = "Paint roughness")

panel = mo.unwrap(mo.box(id = "pn01", size = [1.2, 2.4, 0.08]), id = "pn02")
panel = mo.transform(panel, id = "pn03", translate = [0.0, 1.2, 0.0])
wear = mo.noise_texture(id = "nz01", seed = wear_seed, size = 128, scale = 4, octaves = 3)
paint = mo.ramp(wear, id = "rc01", stops = [0.0, 0.5, 1.0], values = [[0.42, 0.44, 0.4], [0.55, 0.56, 0.52], [0.47, 0.49, 0.45]])
mo.output("panel", mo.material(panel, base_color_texture = paint, id = "mt01", name = "panel_paint", roughness = paint_roughness))
