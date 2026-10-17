# A louvred steel door in a frame: two instanced slat rows painted
# from one ramped noise and one streak roughness, the frame from a
# ridged noise. The three knobs reach their one param each, so the
# seeds reroll and the weathering slides live.
SLAT = [2.4, 0.1, 0.02]
ROWS = 6
PITCH = 0.2

streak_seed = mo.param("streak_seed", 22, min = 0, max = 99, step = 1, label = "Streak seed")
weathering = mo.param("weathering", 0.95, min = 0.55, max = 1.0, step = 0.05, label = "Weathering strength")
paint_seed = mo.param("paint_seed", 21, min = 0, max = 99, step = 1, label = "Paint variation seed")

paint = mo.noise_texture(id = "nz01", seed = paint_seed, size = 256, scale = 5, octaves = 5)
paint = mo.ramp(
    paint, id = "rc01", stops = [0.0, 0.45, 0.65, 1.0],
    values = [[0.3, 0.33, 0.36], [0.46, 0.48, 0.5], [0.36, 0.38, 0.41], [0.58, 0.59, 0.6]],
)
streaks = mo.streaks(
    id = "st01", seed = streak_seed, size = 256, count = 14, length = 0.6, width = 0.03,
    wiggle = 0.015, low = 0.35, high = weathering,
)

slats_a = mo.unwrap(mo.box(id = "sa01", size = SLAT), id = "sa02")
slats_a = mo.transform(slats_a, id = "sa03", translate = [0.0, 0.05, 0.0])
slats_a = mo.array(slats_a, id = "sa04", count = ROWS, step_translate = [0.0, PITCH, 0.0])
slats_b = mo.unwrap(mo.box(id = "sb01", size = SLAT), id = "sb02")
slats_b = mo.transform(slats_b, id = "sb03", translate = [0.0, 0.15, -0.002])
slats_b = mo.array(slats_b, id = "sb04", count = ROWS, step_translate = [0.0, PITCH, 0.0])
door = mo.material(slats_a, base_color_texture = paint, roughness_texture = streaks, id = "mt01", name = "door_steel", metallic = 0.15)
door_b = mo.material(slats_b, base_color_texture = paint, roughness_texture = streaks, id = "mt02", name = "door_steel", metallic = 0.15)

post = mo.unwrap(mo.box(id = "fr01", size = [0.08, 1.26, 0.06]), id = "fr02")
post = mo.transform(post, id = "fr03", translate = [-1.24, 0.63, -0.01])
posts = mo.array(post, id = "fr04", count = 2, step_translate = [2.48, 0.0, 0.0])
grain = mo.noise_texture(id = "ng01", seed = 23, size = 256, scale = 6, octaves = 4, mode = "ridged")
grain = mo.ramp(grain, id = "rg01", stops = [0.0, 0.6, 1.0], values = [0.45, 0.6, 0.9])
frame = mo.material(posts, roughness_texture = grain, id = "mt03", name = "frame_steel", base_color = [0.32, 0.33, 0.35], metallic = 0.4)

mo.output("door", door)
mo.output("door_b", door_b)
mo.output("frame", frame)
mo.assert_contact(id = "door_in_frame", outputs = ["door", "frame"], every = ["door", "frame"])
mo.assert_contact(id = "door_b_in_frame", outputs = ["door_b", "frame"], every = ["door_b", "frame"])
