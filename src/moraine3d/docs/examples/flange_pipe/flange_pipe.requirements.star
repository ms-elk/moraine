mo.assert_extent(id = "length", outputs = ["pipe"], measure = "size_y", value = 1.0)
mo.assert_extent(id = "flange_dia", outputs = ["pipe"], measure = "size_x", value = 0.34)
mo.assert_count(id = "bolt_hole_count", faces = ["pipe", "bolt_holes"], value = 8)
