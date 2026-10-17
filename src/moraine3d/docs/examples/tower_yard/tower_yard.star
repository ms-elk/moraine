# A vendored water tower placed twice from one sub-asset file: the
# second tower turned a quarter and built under a wider tank radius
# through an override on the one knob.
TOWER = "subassets/water_tower.star"
PARTS = ["tank", "legs", "brace_low", "brace_high", "rails", "rungs"]

parts = [mo.asset_ref(TOWER, part, id = "rf0%d" % (i + 1)) for i, part in enumerate(PARTS)]
wide_tank = mo.asset_ref(TOWER, "tank", id = "rf07", overrides = {"tank_radius": 1.3})

for i, part in enumerate(PARTS):
    mo.output("a_" + part, mo.transform(parts[i], id = "ta0%d" % (i + 1), translate = [-2.5, 0.0, 0.0]))
b_parts = [wide_tank] + parts[1:]
for i, part in enumerate(PARTS):
    mo.output("b_" + part, mo.transform(
        b_parts[i], id = "tb0%d" % (i + 1), translate = [2.5, 0.0, 0.0], rotate_deg = [0.0, 90.0, 0.0],
    ))

ground = mo.transform(mo.box(id = "gd01", size = [9.0, 0.1, 5.0]), id = "gd02", translate = [0.0, -0.05, 0.0])
mo.output("ground", ground)
mo.waive("ground", "not_sunk")  # the ground slab lies below the plane by design
# The wider tank is a different solid: the roots of the legs, the rails
# and the top rungs in it are declared again for the second tower.
mo.assert_embedded(id = "b_legs_root_in_tank", outputs = ["b_legs", "b_tank"])
mo.assert_embedded(id = "b_rails_root_in_tank", outputs = ["b_rails", "b_tank"])
mo.assert_embedded(id = "b_top_rungs_reach_tank", outputs = ["b_rungs", "b_tank"])
