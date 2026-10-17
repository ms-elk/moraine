# A stringer pallet: seven deck slats on three bearers over three base
# slats, every board an instance of one of two boxes.
SLAT = [0.1, 0.022, 0.8]
BEARER = [1.2, 0.1, 0.1]
LEFT = -0.55  # the first slat's center; the pallet is 1.2 m wide

slat = mo.box(id = "sl01", size = SLAT)
deck_slat = mo.transform(slat, id = "tp01", translate = [LEFT, 0.133, 0.0])
deck = mo.array(deck_slat, id = "ta01", count = 7, step_translate = [0.1833333, 0.0, 0.0])
base_slat = mo.transform(slat, id = "bt01", translate = [LEFT, SLAT[1] / 2, 0.0])
base = mo.array(base_slat, id = "ba01", count = 3, step_translate = [0.55, 0.0, 0.0])

bearer = mo.box(id = "br01", size = BEARER)
bearer = mo.transform(bearer, id = "bp01", translate = [0.0, 0.072, -0.35])
bearers = mo.array(bearer, id = "bb01", count = 3, step_translate = [0.0, 0.0, 0.35])

mo.output("deck", deck)
mo.output("bearers", bearers)
mo.output("base", base)
mo.assert_clearance(id = "deck_gaps", outputs = ["deck"], self = True, min = 0.05)
