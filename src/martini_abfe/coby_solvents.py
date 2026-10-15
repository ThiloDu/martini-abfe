"""COBY definitions of Martini 3 solvents missing from its library, loaded by path."""

solvent_defs = {
    "default": {
        # Octanol, bead positions in nm along the molecular axis
        "OCO": {
            "beads": [
                {"name": "C1", "x": -0.37, "y": 0, "z": 0, "charge": 0},
                {"name": "C2", "x": 0.02, "y": 0, "z": 0, "charge": 0},
                {"name": "PC", "x": 0.37, "y": 0, "z": 0, "charge": 0},
            ],
        },
    },
}
