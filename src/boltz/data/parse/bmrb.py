"""Parse NMR-STAR files from BMRB and convert to PLUMED format.

This module provides utilities for extracting CA and CB chemical shifts
from BMRB NMR-STAR format files and converting them to PLUMED format
for use with chemical shift steering in metadiffusion.

Example usage:
    # As a module
    from boltz.data.parse.bmrb import parse_star_shifts, write_plumed_shifts

    ca_shifts, cb_shifts = parse_star_shifts("bmr26307_3.str")
    write_plumed_shifts(ca_shifts, "ca_shifts.dat", sequence_length=42)
    write_plumed_shifts(cb_shifts, "cb_shifts.dat", sequence_length=42)

    # From command line
    python -m boltz.data.parse.bmrb bmr26307_3.str --seq-length 42
"""

import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple


def parse_star_shifts(star_file: str, verbose: bool = False) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Parse NMR-STAR file and extract CA/CB chemical shifts.

    Args:
        star_file: Path to NMR-STAR format file (e.g., from BMRB)
        verbose: If True, print progress information

    Returns:
        Tuple of (ca_shifts, cb_shifts) where each is a dict mapping
        residue number (1-indexed) to chemical shift value in ppm.
    """
    ca_shifts: Dict[int, float] = {}
    cb_shifts: Dict[int, float] = {}

    with open(star_file, 'r') as f:
        content = f.read()

    # Find the Atom_chem_shift loop
    # Pattern matches: loop_ followed by _Atom_chem_shift columns, then data, then stop_
    pattern = r'loop_\s+(_Atom_chem_shift\.\w+\s+)+(.+?)stop_'
    match = re.search(pattern, content, re.DOTALL)

    if not match:
        raise ValueError("Could not find Atom_chem_shift loop in STAR file")

    # Parse the loop section
    loop_section = match.group(0)
    header_lines = []
    data_lines = []

    in_headers = True
    for line in loop_section.split('\n'):
        line = line.strip()
        if not line or line == 'loop_' or line == 'stop_':
            continue
        if line.startswith('_Atom_chem_shift.'):
            header_lines.append(line)
        elif in_headers and not line.startswith('_'):
            in_headers = False
            data_lines.append(line)
        elif not in_headers:
            data_lines.append(line)

    # Find column indices for required fields
    col_names = [h.replace('_Atom_chem_shift.', '') for h in header_lines]

    try:
        seq_id_idx = col_names.index('Seq_ID')
        atom_id_idx = col_names.index('Atom_ID')
        val_idx = col_names.index('Val')
        comp_id_idx = col_names.index('Comp_ID')
    except ValueError as e:
        raise ValueError(f"Missing required column in STAR file: {e}")

    if verbose:
        print(f"Found {len(data_lines)} data lines")
        print(f"Columns: Seq_ID={seq_id_idx}, Atom_ID={atom_id_idx}, Val={val_idx}, Comp_ID={comp_id_idx}")

    # Parse data lines
    for line in data_lines:
        fields = line.split()
        if len(fields) <= max(seq_id_idx, atom_id_idx, val_idx, comp_id_idx):
            continue

        try:
            seq_id = int(fields[seq_id_idx])
            atom_id = fields[atom_id_idx]
            val = float(fields[val_idx])
            comp_id = fields[comp_id_idx]

            if atom_id == 'CA':
                ca_shifts[seq_id] = val
                if verbose:
                    print(f"  CA: res {seq_id} ({comp_id}) = {val:.3f} ppm")
            elif atom_id == 'CB':
                cb_shifts[seq_id] = val
                if verbose:
                    print(f"  CB: res {seq_id} ({comp_id}) = {val:.3f} ppm")
        except (ValueError, IndexError):
            continue

    return ca_shifts, cb_shifts


def write_plumed_shifts(
    shifts: Dict[int, float],
    output_file: str,
    sequence_length: int,
    missing_value: float = 0.0,
) -> None:
    """Write chemical shifts in PLUMED format.

    Args:
        shifts: Dict mapping residue number to shift value
        output_file: Path to output file
        sequence_length: Total number of residues in sequence
        missing_value: Value to use for residues without data (default 0.0)
    """
    with open(output_file, 'w') as f:
        f.write("# Residue  Shift(ppm)\n")
        for res_id in range(1, sequence_length + 1):
            shift = shifts.get(res_id, missing_value)
            f.write(f"{res_id} {shift:.3f}\n")


def convert_star_to_plumed(
    star_file: str,
    sequence_length: int,
    output_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[str, str]:
    """Convert NMR-STAR file to PLUMED format CA/CB shift files.

    Args:
        star_file: Path to NMR-STAR file
        sequence_length: Number of residues in the sequence
        output_dir: Directory for output files (default: same as input)
        prefix: Prefix for output filenames (default: input filename stem)
        verbose: Print progress information

    Returns:
        Tuple of (ca_file_path, cb_file_path)
    """
    star_path = Path(star_file)

    if output_dir is None:
        output_dir = star_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    if prefix is None:
        prefix = star_path.stem

    # Parse shifts
    if verbose:
        print(f"Parsing {star_file}...")

    ca_shifts, cb_shifts = parse_star_shifts(star_file, verbose=verbose)

    if verbose:
        print(f"\nFound {len(ca_shifts)} CA shifts and {len(cb_shifts)} CB shifts")

    # Write output files
    ca_file = output_dir / f"{prefix}_ca.dat"
    cb_file = output_dir / f"{prefix}_cb.dat"

    write_plumed_shifts(ca_shifts, str(ca_file), sequence_length)
    write_plumed_shifts(cb_shifts, str(cb_file), sequence_length)

    if verbose:
        print(f"\nOutput files:")
        print(f"  CA: {ca_file}")
        print(f"  CB: {cb_file}")

    return str(ca_file), str(cb_file)


def main():
    """Command-line interface for STAR to PLUMED conversion."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert BMRB NMR-STAR chemical shifts to PLUMED format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert with explicit sequence length
  python -m boltz.data.parse.bmrb bmr26307_3.str --seq-length 42

  # Specify output directory and prefix
  python -m boltz.data.parse.bmrb bmr26307_3.str -n 42 -o ./shifts -p abeta42

  # Download from BMRB and convert (requires wget)
  wget https://bmrb.io/ftp/pub/bmrb/entry_directories/bmr26307/bmr26307_3.str
  python -m boltz.data.parse.bmrb bmr26307_3.str -n 42
"""
    )

    parser.add_argument("star_file", help="Path to NMR-STAR format file")
    parser.add_argument(
        "-n", "--seq-length",
        type=int,
        required=True,
        help="Sequence length (number of residues)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        help="Output directory (default: same as input file)"
    )
    parser.add_argument(
        "-p", "--prefix",
        help="Output filename prefix (default: input filename stem)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print progress information"
    )

    args = parser.parse_args()

    try:
        ca_file, cb_file = convert_star_to_plumed(
            args.star_file,
            args.seq_length,
            output_dir=args.output_dir,
            prefix=args.prefix,
            verbose=args.verbose or True,  # Default to verbose for CLI
        )
        print(f"\nConversion complete!")
        print(f"CA shifts: {ca_file}")
        print(f"CB shifts: {cb_file}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
