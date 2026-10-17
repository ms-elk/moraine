mo.assert_clearance(id = "rails_clear_legs", outputs = ["rails", "legs"], min = 0.05)
mo.assert_clearance(id = "rungs_clear_legs", outputs = ["rungs", "legs"], min = 0.05)
mo.assert_contact(id = "rungs_span_rails", outputs = ["rungs", "rails"], every = ["rungs"])
