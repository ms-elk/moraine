mo.assert_contact(id = "rocks_on_ground", outputs = ["rocks", "ground"], every = ["rocks"])
mo.assert_count(id = "rock_count", outputs = ["rocks"], value = 40)
mo.assert_extent(id = "ground_size", outputs = ["ground"], measure = "size_x", value = 8.0)
