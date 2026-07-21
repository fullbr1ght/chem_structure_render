from io import BytesIO

from fastapi import FastAPI, HTTPException, Response
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor
from PIL import Image

app = FastAPI()


def build_mol_from_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise HTTPException(status_code=400, detail="Invalid SMILES")

    rdDepictor.Compute2DCoords(mol)
    return mol


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


@app.get("/render/svg")
def render_svg(smiles: str):
    mol = build_mol_from_smiles(smiles)

    drawer = rdMolDraw2D.MolDraw2DSVG(500, 300)
    opts = drawer.drawOptions()
    opts.clearBackground = False
    opts.bondLineWidth = 2.3
    opts.padding = 0.10
    opts.additionalAtomLabelPadding = 0.15
    opts.prepareMolsBeforeDrawing = True
    opts.baseFontSize = 0.7
    opts.splitBonds = False
    apply_custom_palette(opts)

    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    svg = drawer.GetDrawingText()
    svg = svg.replace("<svg", '<svg style="background-color:white"', 1)

    return Response(content=svg, media_type="image/svg+xml")


@app.get("/render/png")
def render_png(smiles: str):
    mol = build_mol_from_smiles(smiles)

    drawer = rdMolDraw2D.MolDraw2DCairo(500, 300)
    opts = drawer.drawOptions()
    opts.clearBackground = False
    opts.bondLineWidth = 2.3
    opts.padding = 0.10
    opts.additionalAtomLabelPadding = 0.15
    opts.prepareMolsBeforeDrawing = True
    opts.baseFontSize = 0.7
    opts.splitBonds = False
    apply_custom_palette(opts)

    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    png_bytes = drawer.GetDrawingText()

    image = Image.open(BytesIO(png_bytes)).convert("RGBA")
    white_bg = Image.new("RGBA", image.size, "WHITE")
    white_bg.alpha_composite(image)

    output = BytesIO()
    white_bg.convert("RGB").save(output, format="PNG")

    return Response(content=output.getvalue(), media_type="image/png")