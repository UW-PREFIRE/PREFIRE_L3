"""
Create L3 (Level-3) product from PREFIRE Level-2B files.

This program requires python version 3.6 or later, and is importable as a
python module.
"""

from pathlib import Path
import numpy as np
import sys
import os
import glob
import argparse
import netCDF4 as nc4

#from PREFIRE_L3.utils.data_IO import find_files
import PREFIRE_L3.paths


def create_L3_product(input_L2B_dir, L3_output_dir):
    """
    Main function to Create L3 (Level-3) product from PREFIRE Level-2B files.

    Parameters
    ----------
    input_L2B_dir : str
        Directory containing PREFIRE Level-2B granule files.
    L3_output_dir : str
        Directory to which L3 NetCDF-format file(s) will be written.

    Returns
    -------
    None.

    """
    
    # Set ancillary filepaths:
    ancillary_Path = Path(ancillary_dir)
#    CO2_modelfit_fpath = str(ancillary_Path / "CO2_model_spline_tck.h5")

    work_dir = L3_output_dir

    # Calculate stuff:


    # Write L3 product file:
