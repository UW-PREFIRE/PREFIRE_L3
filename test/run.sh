#!/usr/bin/env bash

#set -ve;  # Exit when the first non-zero exit status is encountered, and
           #  print out commands as we execute them
set -e;  # Exit when the first non-zero exit status is encountered

# Make sure to use an absolute path to the script as a basedir so we can call
# the script from any location:
readonly basedir=$(dirname $(realpath $0));

# Set input/output directories to be relative to the current working dir:

ANCILLARY_DATA_DIR=${PWD}/../dist/ancillary;

L2B_PRODUCT_DIR=${PWD}/inputs;

OUTPUT_DIR=${PWD}/outputs;

# For this case, we assume inputs have been already copied, so bail if input
# files/dirs do not exist.
test -d "${ANCILLARY_DATA_DIR}" || { echo "Ancillary input directory does not exist: ${ANCILLARY_DATA_DIR}"; exit 1; }
test -d "${L2B_PRODUCT_DIR}" || { echo "Input L2B directory does not exist: ${L2B_PRODUCT_DIR}"; exit 1; }

# Check if output file directory exists; if not, bail:
tmpdir=$(dirname ${OUTPUT_DIR});
test -d "${tmpdir}" || { echo "Output directory does not exist: ${tmpdir}"; exit 1; }

cd "${basedir}/../dist";

# Set some necessary parameters:

SPECTRAL_IDX_RANGE_0BI="ATRACK_IDXRANGE_0BASED_INCLUSIVE:0:END";
#SPECTRAL_IDX_RANGE_0BI="ATRACK_IDXRANGE_0BASED_INCLUSIVE:0:10";
#SPECTRAL_IDX_RANGE_0BI="ATRACK_IDXRANGE_0BASED_INCLUSIVE:20:21";

PRODUCT_MONIKER="3-SFC-SORTED-ALLSKY";

PRODUCT_FULLVER="S02_R00";

# Make required environment vars available:
export ANCILLARY_DATA_DIR L2B_PRODUCT_DIR OUTPUT_DIR PRODUCT_FULLVER;
export SPECTRAL_IDX_RANGE_0BI PRODUCT_MONIKER;

# If custom conda environment files exist, activate that conda environment:
chk_dir="${basedir}"/../dist/c_env_for_PREFIRE_L3;
if [ -d "$chk_dir" ]; then
   source "${chk_dir}"/bin/activate;
fi

# Execute script that writes a new 'prdgit_version.txt', which contains product
#  moniker(s) and current (latest) git hash(es) that are part of the provenance
#  of this package's product(s).
# *** This step should not be done within the operational system, since that
#     file is created just before delivery to the system.
hn=`hostname -s`;
if [ "x$hn" = "xlongwave" ]; then
   python "${basedir}"/../dist/determine_prdgit.py;
fi

# Run the process/calculation (specifying that numpy, scipy, et cetera should
#  not use more than one thread or process):

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python "${basedir}"/../dist/produce_L3.py;

echo "TEST completed successfully";

# If custom conda environment files exist, DEactivate that conda environment:
chk_dir="${basedir}"/../dist/c_env_for_PREFIRE_L3;
if [ -d "$chk_dir" ]; then
   source "${chk_dir}"/bin/deactivate;
fi

#   if [ $? -ne 0 ]
#   then
#      echo "FAILED";
#   fi
