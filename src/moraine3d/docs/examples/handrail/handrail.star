# A handrail swept along an L-shaped path over three posts: the rail
# mitered at the corner, its width a knob inside the profile, and every
# post asserted to carry it.
HEIGHT = 0.9
RUN = 2.0
RETURN = 1.2
POST = 0.04
INSET = 0.1
END = 1.1

half = mo.param("rail_width", 0.06, min = 0.05, max = 0.1, step = 0.01, label = "Rail width") / 2

# Counter-clockwise, seen with the path coming toward you. The origin,
# the middle of the rail's underside, rides the path.
profile = [[-half, 0.0], [half, 0.0], [half, 0.03], [half / 2, 0.04], [-half / 2, 0.04], [-half, 0.03]]
rail = mo.sweep(id = "sw01", profile = profile, path = [[0.0, HEIGHT, 0.0], [RUN, HEIGHT, 0.0], [RUN, HEIGHT, RETURN]])

post = mo.box(id = "bx01", size = [POST, HEIGHT, POST])
first = mo.transform(post, id = "tr01", translate = [INSET, HEIGHT / 2, 0.0])
posts = mo.array(first, id = "ar01", count = 2, step_translate = [RUN - INSET, 0.0, 0.0])
end_post = mo.transform(post, id = "tr02", translate = [RUN, HEIGHT / 2, END])

mo.output("rail", rail)
mo.output("posts", posts)
mo.output("end_post", end_post)
mo.assert_contact(id = "posts_carry_rail", outputs = ["rail", "posts"], every = ["posts"])
mo.assert_contact(id = "end_post_carries_rail", outputs = ["rail", "end_post"])
