mo.assert_extent(id = "rise", outputs = ["steps"], measure = "max_y", value = 2.1)
mo.assert_extent(id = "width", outputs = ["steps"], measure = "size_x", value = 1.0)
mo.assert_count(id = "risers", outputs = ["steps"], value = 12)
mo.assert_contact(id = "steps_on_stringers", outputs = ["stringers", "steps"], every = ["steps"], min_area = 0.02)
