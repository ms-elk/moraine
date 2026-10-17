# One panel kit placed twice: the second use overrides the kit's two
# knobs and gives its material a name of its own, so the two contents
# ship as two identities under a budget of two.
PANEL = "subassets/panel.star"

plain = mo.asset_ref(PANEL, "panel", id = "rf01")
worn = mo.asset_ref(
    PANEL, "panel", id = "rf02",
    overrides = {"wear_seed": 23, "paint_roughness": 0.85},
    material_rename = {"panel_paint": "panel_paint_worn"},
)
mo.output("panel_a", mo.transform(plain, id = "tr01", translate = [-0.7, 0.0, 0.0]))
mo.output("panel_b", mo.transform(worn, id = "tr02", translate = [0.7, 0.0, 0.0]))
mo.assert_clearance(id = "panels_apart", outputs = ["panel_a", "panel_b"], min = 0.001)
