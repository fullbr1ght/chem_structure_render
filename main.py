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

def prefer_heteroatom_stereo_bonds(mol):
    """Prefer drawing stereobonds to heteroatoms instead of the carbon spine."""
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    chiral_centers = {
        atom_index
        for atom_index, assignment in Chem.FindMolChiralCenters(
            mol,
            includeUnassigned=False,
            includeCIP=True,
            useLegacyImplementation=False,
        )
        if assignment != "?"
    }

    if not chiral_centers or mol.GetNumConformers() == 0:
        return

    conformer = mol.GetConformer()
    Chem.WedgeMolBonds(mol, conformer)

    for atom_index in chiral_centers:
        atom = mol.GetAtomWithIdx(atom_index)
        candidates = []

        for bond in atom.GetBonds():
            other_atom = bond.GetOtherAtom(atom)

            if bond.GetBondType() != Chem.BondType.SINGLE:
                continue

            atomic_number = other_atom.GetAtomicNum()
            is_non_ring_amine = (
                atomic_number == 7
                and not other_atom.IsInRing()
            )
            is_terminal_heteroatom = (
                atomic_number not in (1, 6, 7)
                and other_atom.GetDegree() == 1
            )

            # Prefer NH2/NHR/NR2 groups outside rings, plus terminal OH and
            # halogens. Embedded ring nitrogens and ether oxygens are ignored.
            if not (is_non_ring_amine or is_terminal_heteroatom):
                continue

            candidates.append((
                other_atom.GetDegree() == 1,
                not bond.IsInRing(),
                other_atom.GetIdx() not in chiral_centers,
                -bond.GetIdx(),
                bond,
            ))

        if not candidates:
            continue

        preferred_bond = max(candidates, key=lambda item: item[:-1])[-1]

        # Remove RDKit's existing choice for this center before replacing it.
        for bond in atom.GetBonds():
            if (
                bond.GetBondDir()
                in (Chem.BondDir.BEGINWEDGE, Chem.BondDir.BEGINDASH)
            ):
                bond.SetBondDir(Chem.BondDir.NONE)

        Chem.WedgeBond(preferred_bond, atom_index, conformer)

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
    prefer_heteroatom_stereo_bonds(mol)
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
def render_svg(
    smiles: str,
    highlight_stereo: bool = False,
):
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
    if has_depiction_override:
        opts.useMolBlockWedging = True
    opts.baseFontSize = 0.75
    opts.maxFontSize = 26
    opts.splitBonds = False

    if highlight_stereo:
        opts.useBWAtomPalette()
    else:
        apply_custom_palette(opts)

    if highlight_stereo:
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        chiral_centers = Chem.FindMolChiralCenters(
            mol,
            includeUnassigned=True,
            includeCIP=True,
            useLegacyImplementation=False,
        )

        for atom_index, assignment in chiral_centers:
            if assignment == "?":
                mol.GetAtomWithIdx(atom_index).SetProp("atomNote", "(?)")

        opts.addStereoAnnotation = True
        opts.setAtomNoteColour((0.95, 0.55, 0.10))
        opts.atomHighlightsAreCircles = True
        opts.fillHighlights = True

        highlight_atoms = [
            atom_index
            for atom_index, _ in chiral_centers
        ]
        highlight_colors = {
            atom_index: (0.95, 0.55, 0.10, 0.25)
            for atom_index in highlight_atoms
        }
        highlight_radii = {
            atom_index: 0.15
            for atom_index in highlight_atoms
        }

        drawer.DrawMolecule(
            mol,
            highlightAtoms=highlight_atoms,
            highlightAtomColors=highlight_colors,
            highlightAtomRadii=highlight_radii,
        )
    else:
        drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    svg = drawer.GetDrawingText()
    svg = svg.replace("<svg", '<svg style="background-color:white"', 1)

    return Response(content=svg, media_type="image/svg+xml")
