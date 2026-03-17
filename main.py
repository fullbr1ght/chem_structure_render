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


@app.get("/render/svg")
def render_svg(smiles: str):
    mol = build_mol_from_smiles(smiles)

    drawer = rdMolDraw2D.MolDraw2DSVG(500, 300)
    opts = drawer.drawOptions()
    opts.clearBackground = False
    opts.bondLineWidth = 3

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
    opts.bondLineWidth = 3

    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    png_bytes = drawer.GetDrawingText()

    image = Image.open(BytesIO(png_bytes)).convert("RGBA")
    white_bg = Image.new("RGBA", image.size, "WHITE")
    white_bg.alpha_composite(image)

    output = BytesIO()
    white_bg.convert("RGB").save(output, format="PNG")

    return Response(content=output.getvalue(), media_type="image/png")