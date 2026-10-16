import math
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor
from rdkit.Chem import inchi

app = FastAPI()

MAX_SMILES_LENGTH = 1000
MAX_ATOMS = 250

DEPICTIONS_DIR = Path(__file__).parent / "depictions"

def load_depiction_overrides():
    overrides = {}

    for sdf_path in DEPICTIONS_DIR.glob("*.sdf"):
        supplier = Chem.SDMolSupplier(
            str(sdf_path), 
            removeHs=True, 
        )

        for mol in supplier:
            if mol is not None:
                Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
                inchikey = inchi.MolToInchiKey(mol)
                overrides[inchikey] = mol

    return overrides

DEPICTION_OVERRIDES = load_depiction_overrides()

def build_mol_from_smiles(smiles: str):
    if len(smiles) > MAX_SMILES_LENGTH:
        raise HTTPException(status_code=413, detail="SMILES is too long")

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise HTTPException(status_code=400, detail="Invalid SMILES")

    if mol.GetNumAtoms() > MAX_ATOMS:
        raise HTTPException(
            status_code=413,
            detail="Structure has too many atoms",
        )

    inchikey = inchi.MolToInchiKey(mol)
    override = DEPICTION_OVERRIDES.get(inchikey)

    if override is not None:
        return Chem.Mol(override), True

    rdDepictor.Compute2DCoords(mol)
    return mol, False

def apply_custom_palette(opts):
    opts.updateAtomPalette({
        6: (0.10, 0.10, 0.10),   # C - near black
        7: (0.08, 0.38, 0.78),   # N - blue
        8: (0.82, 0.16, 0.16),   # O - red
        9: (0.00, 0.58, 0.68),   # F - teal
        15: (0.76, 0.54, 0.10),  # P - warm golden ochre
        16: (0.84, 0.68, 0.10),  # S - acid-gold / yellow
        17: (0.12, 0.58, 0.30),  # Cl - green
        35: (0.45, 0.24, 0.08),  # Br - deeper brown
        53: (0.50, 0.24, 0.66),  # I - violet
    })

def find_best_rotation(mol, width=500, height=500):
    conf = mol.GetConformer()
    points = [
        (conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y)
        for i in range(mol.GetNumAtoms())
    ]

    best_angle = 0
    best_scale = 0

    # Test orientations in 5-degree increments.
    for angle in range(0, 180, 5):
        radians = math.radians(angle)
        cos_a = math.cos(radians)
        sin_a = math.sin(radians)

        rotated = [
            (
                x * cos_a - y * sin_a,
                x * sin_a + y * cos_a,
            )
            for x, y in points
        ]

        xs = [x for x, _ in rotated]
        ys = [y for _, y in rotated]

        structure_width = max(xs) - min(xs)
        structure_height = max(ys) - min(ys)

        scale = min(
            width / max(structure_width, 0.01),
            height / max(structure_height, 0.01),
        )

        if scale > best_scale:
            best_scale = scale
            best_angle = angle

    return best_angle


@app.get("/render/svg")
def render_svg(smiles: str):
    mol, has_depiction_override = build_mol_from_smiles(smiles)

    drawer = rdMolDraw2D.MolDraw2DSVG(500, 500)
    opts = drawer.drawOptions()
    if not has_depiction_override:
        opts.rotate = find_best_rotation(mol, 500, 500)
    opts.clearBackground = False
    opts.bondLineWidth = 2.5
    opts.padding = 0.06
    opts.additionalAtomLabelPadding = 0.05
    opts.prepareMolsBeforeDrawing = True
    opts.baseFontSize = 0.7
    opts.maxFontSize = 22
    opts.splitBonds = False
    apply_custom_palette(opts)

    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    svg = drawer.GetDrawingText()
    svg = svg.replace("<svg", '<svg style="background-color:white"', 1)

    return Response(content=svg, media_type="image/svg+xml")
