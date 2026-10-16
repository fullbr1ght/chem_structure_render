#!/usr/bin/env python3

import argparse
from pathlib import Path

from rdkit import Chem


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert MOL files to single-record SDF files while preserving "
            "their molecular coordinates."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="One or more MOL files, or a folder containing MOL files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output SDF path; valid only when converting one MOL file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Place all converted SDF files in this folder",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output SDF files",
    )
    return parser.parse_args()


def collect_mol_files(input_paths):
    mol_files = []

    for entered_path in input_paths:
        path = entered_path.expanduser().resolve()

        if path.is_dir():
            mol_files.extend(sorted(path.glob("*.mol")))
        elif path.is_file() and path.suffix.lower() == ".mol":
            mol_files.append(path)
        else:
            print(f"Skipped unsupported or missing input: {path}")

    return list(dict.fromkeys(mol_files))


def convert_mol_file(input_path, output_path, force=False):
    if output_path.exists() and not force:
        print(f"Skipped existing file: {output_path}")
        return False

    mol = Chem.MolFromMolFile(
        str(input_path),
        sanitize=True,
        removeHs=False,
        strictParsing=True,
    )

    if mol is None:
        print(f"Could not read MOL file: {input_path}")
        return False

    if mol.GetNumConformers() == 0:
        print(f"No molecular coordinates found: {input_path}")
        return False

    if not mol.HasProp("_Name") or not mol.GetProp("_Name").strip():
        mol.SetProp("_Name", input_path.stem)

    writer = Chem.SDWriter(str(output_path))
    try:
        writer.write(mol)
    finally:
        writer.close()

    print(f"Created: {output_path}")
    return True


def main():
    args = parse_args()

    input_paths = args.inputs
    if not input_paths:
        entered_path = input(
            "MOL file or folder path: "
        ).strip().strip('"').strip("'")
        input_paths = [Path(entered_path)]

    mol_files = collect_mol_files(input_paths)
    if not mol_files:
        raise SystemExit("No MOL files were found.")

    if args.output and len(mol_files) != 1:
        raise SystemExit(
            "--output can only be used with one MOL file. "
            "Use --output-dir for batch conversion."
        )

    output_dir = None
    if args.output_dir:
        output_dir = args.output_dir.expanduser().resolve()
        if not output_dir.is_dir():
            raise SystemExit(f"Output directory not found: {output_dir}")

    converted = 0
    for input_path in mol_files:
        if args.output:
            output_path = args.output.expanduser().resolve()
        elif output_dir:
            output_path = output_dir / input_path.with_suffix(".sdf").name
        else:
            output_path = input_path.with_suffix(".sdf")

        if convert_mol_file(input_path, output_path, force=args.force):
            converted += 1

    print(f"Finished: {converted} of {len(mol_files)} file(s) converted.")


if __name__ == "__main__":
    main()
