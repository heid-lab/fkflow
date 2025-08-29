"""
This script reads reaction data from rdb7_full.csv and writes all XYZ coordinates
to a single XYZ file in the same order as the input CSV.
For each reaction, the reactant (r), transition state (ts), and product (p) geometries
are written in that specific order. The SMILES string is included in the comment line.
"""

import ase.io
import pandas as pd
from pathlib import Path


FULL_SMILES_CSV_PATH = 'rdb7/barriers/rdb7_full.csv'
XYZ_SAVE_PATH = 'rdb7'

# Read the full CSV file without splitting
df_all = pd.read_csv(FULL_SMILES_CSV_PATH, sep=',')
print(f"Found {len(df_all)} reactions in the CSV file")

# Create a single output file with all XYZ coordinates
output_file = Path(XYZ_SAVE_PATH) / 'rdb7_full.xyz'
print(f"Writing all XYZ coordinates to {output_file}")

with open(output_file, 'w') as f:
    for _, rxn_row in df_all.iterrows():
        i = rxn_row.rxn
        smiles = rxn_row.smiles  # Get the SMILES string for this reaction
        xyz_folder_path = f'rdb7/geometries/rxn{i:06d}/'
        
        # For each reaction, write r -> ts -> p in that specific order
        for rtsp in ['r', 'ts', 'p']:
            try:
                xyz_file = Path(xyz_folder_path) / f'{rtsp}{i:06d}.xyz'
                print(f"Adding {xyz_file} to output with SMILES: {smiles}")
                
                # Read the structure
                atoms = ase.io.read(xyz_file)
                
                # Add the SMILES string and structure type to the comment
                comment = f"Reaction {i:06d} | {rtsp.upper()} | SMILES: {smiles}"
                atoms.info['comment'] = comment
                
                # Write to output file with the modified comment
                ase.io.write(
                    filename=f, 
                    images=atoms, 
                    format='xyz'
                )
            except FileNotFoundError:
                print(f"Warning: Could not find XYZ file for reaction {i}, structure {rtsp}")

print("Done! All reaction geometries written to a single XYZ file in r-ts-p order for each reaction.")

# Add assertions to verify no data leakage (not relevant for this case, but good practice)
assert len(df_all) == df_all['rxn'].nunique(), "Some reaction IDs appear multiple times in the input CSV"
