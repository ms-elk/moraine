"""Deterministic validation (DESIGN 8.3): the report (``validate``), the
assert kinds it dispatches to (``asserts``, one module per kind), and
the kernels they measure with — exact world bounds (``bounds``), welded
solids (``solids``), penetration depth (``depth``), surface distance
(``distance``), and the instanced copy-pair broad phase the self paths
share (``pairs``). Reads evaluated values only; never imports the
engine.
"""
