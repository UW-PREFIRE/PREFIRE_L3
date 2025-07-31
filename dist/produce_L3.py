"""
Produce gridded, sorted, and/or statistical representations of PREFIRE Level-2B
products.

This program requires python version 3.6 or later, and is importable as a 
python module.
"""

  # From the Python standard library:
from pathlib import Path
import os
import sys
import argparse

  # From other external Python packages:

  # Custom utilities:


#--------------------------------------------------------------------------
def main(anchor_path):
    """Driver routine."""

    sys.path.append(os.path.join(anchor_path, "..", "source"))
    from PREFIRE_L3.create_L3_sorted import create_L3_sorted

#    this_environ = os.environ.copy()

    this_dir = Path(anchor_path)  # typically the 'dist' directory
    ancillary_dir = os.environ["ANCILLARY_DATA_DIR"]
    input_L2B_dir = os.environ["L2B_PRODUCT_DIR"]
    output_dir = os.environ["OUTPUT_DIR"]

    product_moniker = os.environ["PRODUCT_MONIKER"]
    product_full_version = os.environ["PRODUCT_FULLVER"]

    spectral_idx_range_str = os.environ["SPECTRAL_IDX_RANGE_0BI"]
    tokens = spectral_idx_range_str.split(':')
    if tokens[2] == "END":
        spectral_np_idx_range = ("spectral", int(tokens[1]),
                                 None)  # Numpy indexing
    else:
        spectral_np_idx_range = ("spectral", int(tokens[1]),
                                 int(tokens[2])+1)  # Numpy indexing

    # Create the L3 product:
    
    if "SORTED" in product_moniker:
        create_L3_sorted(input_L2B_dir, output_dir, product_full_version,
                         product_moniker, spectral_np_idx_range)
    else:  # General, unsorted
        create_L3_general(input_L2B_dir, output_dir, product_full_version,
                          product_moniker, spectral_np_idx_range)


if __name__ == "__main__":
    # Determine fully-qualified filesystem location of this script:
    anchor_path = os.path.abspath(os.path.dirname(sys.argv[0]))

    # Process arguments:
    arg_description = ("Produce gridded, sorted, and/or statistical "
                       "representations of PREFIRE Level-2B products.")
    arg_parser = argparse.ArgumentParser(description=arg_description)

    args = arg_parser.parse_args()

    # Run driver:
    main(anchor_path)
