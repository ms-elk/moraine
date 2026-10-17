# Two shelves, each a pair of capped posts and two boards, named as
# output groups: unwrap and material on both prototypes before the
# arrays, the cap height read from a tag only the cap can satisfy.
POST_H = 0.8
CAP_T = 0.02
SHELF_B_X = 1.5

post = mo.transform(mo.box(id = "ps01", size = [0.04, POST_H, 0.3]), id = "ps02", translate = [0.0, POST_H / 2, 0.0])
cap = mo.transform(mo.box(id = "cp01", size = [0.06, CAP_T, 0.34]), id = "cp02", translate = [0.0, POST_H + CAP_T / 2, 0.0])
post = mo.boolean(post, cap, id = "ps03", mode = "union")
# The band `center_y > 0.81` is reached by the cap's top alone: built
# without the cap, the tag matches nothing and the entry turns red.
post = mo.tag(post, id = "ps04", name = "cap_top", select = {"normal_y": {"gt": 0.9}, "center_y": {"gt": 0.81}})
post = mo.tag(post, id = "ps05", name = "inner", select = {"normal_x": {"gt": 0.9}, "center_y": {"lt": 0.79}})
post = mo.unwrap(post, id = "ps06", mode = "box", uv_scale = 2.0)
post = mo.material(post, id = "ps07", name = "steel_grey", base_color = [0.35, 0.36, 0.38, 1.0], metallic = 0.9, roughness = 0.45)
post = mo.transform(post, id = "ps08", translate = [-0.48, 0.0, 0.0])
posts = mo.array(post, id = "ps09", count = 2, step_translate = [0.96, 0.0, 0.0])

board = mo.tag(mo.box(id = "bd01", size = [0.92, 0.04, 0.3]), id = "bd02", name = "end_left", select = {"normal_x": {"lt": -0.9}})
board = mo.unwrap(board, id = "bd03", mode = "box", uv_scale = 2.0)
board = mo.material(board, id = "bd04", name = "oak", base_color = [0.62, 0.45, 0.28, 1.0], roughness = 0.7)
board = mo.transform(board, id = "bd05", translate = [0.0, 0.3, 0.0])
boards = mo.array(board, id = "bd06", count = 2, step_translate = [0.0, 0.4, 0.0])

mo.output("shelf_a_posts", posts)
mo.output("shelf_a_boards", boards)
mo.output("shelf_b_posts", mo.transform(posts, id = "sb01", translate = [SHELF_B_X, 0.0, 0.0]))
mo.output("shelf_b_boards", mo.transform(boards, id = "sb02", translate = [SHELF_B_X, 0.0, 0.0]))
mo.assert_contact(id = "boards_held", outputs = ["shelf_a_boards", "shelf_a_posts"], every = ["shelf_a_boards", "shelf_a_posts"])
mo.assert_count(id = "two_boards", outputs = ["shelf_a_boards"], value = 2)
