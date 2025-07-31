"""
Create a Level 3 (sorted by some metric) product from PREFIRE data product
 granules.

This program requires Python version 3.6 or later, and is importable as a
Python module.
"""

  # From the Python standard library:
import glob
import os
import sys
import datetime
import argparse
import copy
from pathlib import Path

  # From other external Python packages:
import numpy as np
import netCDF4 as nc


#------------------------------------------------------------------------------
def SFC_SORTED_ALLSKY(inpd, t_msk=None, diminfo_only=False):
    outd = {}  # Dictionary for field data that is ready for binning/stats

    # Only used as a reference for the number of sort metric elements
    #  (as the output file metadata is defined in the filespecs JSON file):
    outd["ref_sortmetric_d"] = {
                              1: "open water", 2: "sea ice",
                              3: "partial sea ice", 4: "permanent land ice",
                              5: "Antarctic ice shelf", 6: "snow-covered land",
                              7: "partial snow-covered land",
                              8: "snow-free land", 9: "coastal"}
    outd["ref_sortmetric_values"] = np.array(
                                         [x for x in outd["ref_sortmetric_d"]])
    if diminfo_only:
        return len(outd["ref_sortmetric_values"])

    #=== Construct (potentially-filtered) fields to use for binning/stats:

    t_msk2d = np.full(inpd["latitude"].shape, True, dtype="bool")
    if t_msk is not None:
        t_msk2d[:,:] = ( t_msk[:,np.newaxis] == True )

    f_msk2d = ( inpd["sfc_quality_flag"] == 0 )  # use quality flag 0 emiss only
#    f_msk2d = ( inpd["sfc_quality_flag"] <= 1 )  # use quality flag 0 & 1 emiss

    msk2d = ( t_msk2d & f_msk2d )  # Combined elements-to-use mask

    where_res = np.where(msk2d)
    outd["nominal_scenes"] = where_res[1]  # Needed for separation of channels
    outd["nominal_frames"] = where_res[0]  # Needed to distinguish orbit branch

    outd["nominal_lats"] = inpd["latitude"][msk2d]
    outd["nominal_lons"] = inpd["longitude"][msk2d]

    tmp = np.broadcast_to(inpd["satellite_pass_type"][:,np.newaxis],
                          inpd["latitude"].shape)
    outd["sat_pass_type"] = tmp[msk2d]

    outd["nominal_field"] = inpd["sfc_spectral_emis"][msk2d,:]

    #=== Construct the metric for sorting:

       # Only used as a reference for the number of sort metric elements
       #  (as the output file metadata is defined in the filespecs JSON file):
    outd["ref_sortmetric_d"] = {
                              1: "open water", 2: "sea ice",
                              3: "partial sea ice", 4: "permanent land ice",
                              5: "Antarctic ice shelf", 6: "snow-covered land",
                              7: "partial snow-covered land",
                              8: "snow-free land", 9: "coastal"}
    outd["ref_sortmetric_values"] = np.array(
                                         [x for x in outd["ref_sortmetric_d"]])

    if inpd["merged_surface_type_prelim"] is None:  # Need at least AUX-MET
        return None  # Nothing to do for this granule set

    if inpd["merged_surface_type_final"] is not None:  # Use AUX-SAT
        outd["sort_metric"] = inpd["merged_surface_type_final"][msk2d]
    else:  # Use AUX-MET
        outd["sort_metric"] = inpd["merged_surface_type_prelim"][msk2d]

    coastal_lower = 0.1  # lower land fraction threshold defining "coastal"
    coastal_upper = 0.9  # upper land fraction threshold defining "coastal"

    # Reclassify Non-Antarctic coastal FOVs:

    c1A = ( outd["nominal_lats"] > -60. )                   # condition 1A
    c1B = ( inpd["land_fraction"][msk2d] > coastal_lower )  # condition 1B
    c1C = ( inpd["land_fraction"][msk2d] < coastal_upper )  # condition 1C

    outd["sort_metric"][c1A & c1B & c1C] = 9  # Set to coastal surface type

    # Reclassify Antarctic coastal FOVs:

      # Sum of Antarctic land fraction and ice shelf fraction:
    ALF_ISF = (inpd["antarctic_land_fraction"][msk2d]+
               inpd["antarctic_ice_shelf_fraction"][msk2d])

    c2A = ( outd["nominal_lats"] <= -60. )  # condition 2A
    c2B = ( ALF_ISF > coastal_lower )       # condition 2B
    c2C = ( ALF_ISF < coastal_upper )       # condition 2C

    outd["sort_metric"][c2A & c2B & c2C] = 9  # Set to coastal surface type

    return outd


L3_d = {"SFC-SORTED-ALLSKY": {
            "proc_method": SFC_SORTED_ALLSKY,
            "output_data_group": "Sfc-Sorted",
            "tdep_input": [
                           ("2B-SFC",
                             [("Geometry",
                                ("latitude", "longitude", "land_fraction",
                                 "satellite_pass_type")),
                              ("Sfc",
                                ("sfc_spectral_emis", "sfc_quality_flag"))]),
                           ("AUX-MET",
                             [("Aux-Met",
                                ("merged_surface_type_prelim",
                                 "antarctic_land_fraction",
                                 "antarctic_ice_shelf_fraction"))]),
                           ("AUX-SAT",
                             [("Aux-Sat",
                                ("merged_surface_type_final",))])],
            "tind_input": [
                           ("2B-SFC",
                             [("Sfc",
                                ("wavelength", "idealized_wavelength"))]) ],
            "field": ("already_prepared", "sfc_spectral_emis"),
            "sort_metric": ("need_to_prepare", "modified_sfc_type"),
            "output_array_names": [
                    "latitude", "longitude",
                    "wavelength", "idealized_wavelength",
                    "surface_type_for_sorting",
                    "emis_mean", "asc_emis_mean", "desc_emis_mean",
                    "emis_stdev", "asc_emis_stdev", "desc_emis_stdev", 
                    "count", "asc_count", "desc_count",
                    "emis_sum", "asc_emis_sum", "desc_emis_sum",
                    "emis_sumsquares", "asc_emis_sumsquares",
                    "desc_emis_sumsquares"]
         }
         }


#------------------------------------------------------------------------------
def refined_fpath_DTrange_list(all_fpaths_sorted):
    n_tmp0 = len(all_fpaths_sorted)

    # Remove any duplicate fpaths (with respect to start/end time coverage),
    #  retaining those with the latest creation-time filename component:
    tmp0 = [datetime.datetime.strptime(os.path.basename(x).split('_')[5]+
                       " +0000", "%Y%m%d%H%M%S %z") for x in all_fpaths_sorted]
    tmp = [(xb, xe, fp) for xb, xe, fp in zip(tmp0[0:n_tmp0-1], tmp0[1:],
                                                all_fpaths_sorted[0:n_tmp0-1])]
    tmp.append((tmp0[-1], tmp0[-1]+datetime.timedelta(minutes=95.5),
               all_fpaths_sorted[-1]))
    return tmp


#--------------------------------------------------------------------------
def create_L3_sorted(cfg_d):
    """Average, sort, and grid (et cetera) various fields from input PREFIRE
       data product files."""

    # Custom utilities:
    if "append_to_syspath" in cfg_d:
        sys.path.append(cfg_d["append_to_syspath"])

    import PREFIRE_L3.filepaths
    from PREFIRE_tools.utils.numeric import stdev_func
    from PREFIRE_tools.utils.time import (init_leap_s_for_ctimeRefEpoch,
                                          UTC_DT_to_ctime, ctime_to_UTC_DT)
    from PREFIRE_PRD_GEN.file_creation import write_data_fromspec


    leap_s_info = init_leap_s_for_ctimeRefEpoch([2000, 1, 1, 0, 0, 0],
                                            epoch_for_ctime_is_actual_UTC=True)

    UTC_dtrep_range_to_proc = copy.deepcopy(
                                        list(cfg_d["UTC_dtrep_range_to_proc"]))
    if cfg_d["cfg_end_UTC_exclusive"]:
        td_1ms = datetime.timedelta(milliseconds=1)
        UTC_DT_to_proc_t = (datetime.datetime.fromisoformat(
                                UTC_dtrep_range_to_proc[0]+"+00:00"),
                            datetime.datetime.fromisoformat(
                                UTC_dtrep_range_to_proc[1]+"+00:00")-td_1ms)
        UTC_dtrep_range_to_proc[1] = UTC_DT_to_proc_t[1].strftime(
                                                          "%Y-%m-%dT%H:%M:%SZ")
    else:
        UTC_DT_to_proc_t = tuple(
                               [datetime.datetime.fromisoformat(x+"+00:00") for
                                                 x in UTC_dtrep_range_to_proc])
    DT_beg, DT_end = UTC_DT_to_proc_t

    ctime_to_proc_t, _ = UTC_DT_to_ctime([DT_beg, DT_end], 's', leap_s_info)

    fld_d = L3_d[cfg_d["L3_field_moniker"]]

    input_prd_mnk_l = [prd_mnk for prd_mnk, _ in fld_d["tdep_input"]]
    if fld_d["tind_input"] is not None:
        input_prd_mnk_l.extend([prd_mnk for prd_mnk, _ in fld_d["tind_input"]
                                            if prd_mnk not in input_prd_mnk_l])

    input_prd_fpaths = {}

    # Search for all matching granules for the first (primary) time-dependent
    #  PREFIRE data product moniker, then subset the resulting list to include
    #  only those granules within the desired datetime range:

    primary_prd_mnk = input_prd_mnk_l[0]
    input_search_str = str(cfg_d["input_product_Path"] / primary_prd_mnk /
                           "*[0-9].nc")
    all_candidate_fpaths = sorted(glob.glob(input_search_str))
    infp_t = refined_fpath_DTrange_list(all_candidate_fpaths)
    DT_b_files_tmp, DT_e_files_tmp, fpaths_tmp = list(zip(*infp_t))
    tmp_fpaths = np.array(fpaths_tmp)
    DT_b_files = np.array(DT_b_files_tmp)
    DT_e_files = np.array(DT_e_files_tmp)

    input_prd_fpaths[primary_prd_mnk] = tmp_fpaths[np.nonzero(
                            ((DT_b_files >= DT_beg) | (DT_e_files >= DT_beg)) &
                            ((DT_b_files < DT_end) | (DT_e_files < DT_end)))]

    n_input_prd_fpaths = len(input_prd_fpaths[primary_prd_mnk])
    if n_input_prd_fpaths == 0:
        raise RuntimeError("No relevant primary input PREFIRE data product "
                           "files found.")

      # Determine which frames of the first/last granule to use:

    t_msk_2p = [None, None]
    if n_input_prd_fpaths == 1:  # Special case of only one granule
        ign = 0
        with nc.Dataset(input_prd_fpaths[primary_prd_mnk][ign], 'r') as ds:
            ct = ds.groups["Geometry"].variables["ctime"][...]
            ib = np.searchsorted(ct, ctime_to_proc_t[0], side="left")
            tmp = np.array([False if x < i else True for x in range(len(ct))])
            ie = np.searchsorted(ct, ctime_to_proc_t[1], side="right")-1
            tmp[ie:] = False
        tmpct = ct[tmp]
        ct_coverage_start, ct_coverage_end = (np.min(tmpct), np.max(tmpct))
        t_msk_2p[0] = tmp
        t_msk_2p[1] = tmp
    else:
        ignA = 0
        ds = nc.Dataset(input_prd_fpaths[primary_prd_mnk][ignA], 'r')
        ct = ds.groups["Geometry"].variables["ctime"][...]
        i = np.searchsorted(ct, ctime_to_proc_t[0], side="left")
        if i == len(ct):
            ds.close()
            ignA += 1
            ds = nc.Dataset(input_prd_fpaths[primary_prd_mnk][ignA], 'r')
            ct = ds.groups["Geometry"].variables["ctime"][...]
            i = np.searchsorted(ct, ctime_to_proc_t[0], side="left")
        tmp = np.array([False if x < i else True for x in range(len(ct))])
        ct_coverage_start = np.min(ct[tmp])
        t_msk_2p[0] = tmp
        ds.close()

        ignB = n_input_prd_fpaths-1
        ds = nc.Dataset(input_prd_fpaths[primary_prd_mnk][ignB], 'r')
        ct = ds.groups["Geometry"].variables["ctime"][...]
        i = np.searchsorted(ct, ctime_to_proc_t[1], side="right")-1
        if i < 0 or i == len(ct):
            ds.close()
            ignB -= 1
            ds = nc.Dataset(input_prd_fpaths[primary_prd_mnk][ignB], 'r')
            ct = ds.groups["Geometry"].variables["ctime"][...]
            i = np.searchsorted(ct, ctime_to_proc_t[1], side="right")-1
        tmp = np.array([False if x > i else True for x in range(len(ct))])
        ct_coverage_end = np.max(ct[tmp])
        t_msk_2p[1] = tmp
        ds.close()

          # Revise number of input fpaths (and revise any items associated
          #  with that):
        n_input_prd_fpaths = ignB-ignA+1  # Revise number of input fpaths
        input_prd_fpaths[primary_prd_mnk] = np.array([x for x in
                               input_prd_fpaths[primary_prd_mnk][ignA:ignB+1]])

    t_msk = [None] * n_input_prd_fpaths
    t_msk[0] = t_msk_2p[0]
    t_msk[-1] = t_msk_2p[1]

    # Find all secondary product-type granules which are associated with the
    #  granules of the primary PREFIRE data product type for this processing:

    for prd_mnk in input_prd_mnk_l[1:]:
        tmp_fn_l = [os.path.basename(x).replace(primary_prd_mnk, prd_mnk)
                                    for x in input_prd_fpaths[primary_prd_mnk]]
        tmp_Path = cfg_d["input_product_Path"] / prd_mnk
        tmp_fPath_l = [tmp_Path / x for x in tmp_fn_l]
        input_prd_fpaths[prd_mnk] = np.array([str(x) if os.path.isfile(x)
                                               else None for x in tmp_fPath_l])

    # Define the geographic grid that the Level 3 product will be binned to:

    lon_lower = -180.  # [deg_E]
    lon_upper = 180.  # [deg_E]
    dlon = 1.  # [deg_E]
    n_lon = int(np.rint((lon_upper-lon_lower)/dlon))
    lon_midvalues = np.linspace(lon_lower+0.5*dlon, lon_upper-0.5*dlon, n_lon)

    lat_lower = -84.  # [deg_N]
    lat_upper = 84.  # [deg_N]
    dlat = 1.  # [deg_N]
    n_lat = int(np.rint((lat_upper-lat_lower)/dlat))
    lat_midvalues = np.linspace(lat_lower+0.5*dlon, lat_upper-0.5*dlat, n_lat)

    inpd = {}  # Dictionary for input field data

    # Read in any time-<independent> fields:
    for prd_mnk, v_tl in fld_d["tind_input"]:
        if input_prd_fpaths[prd_mnk][0] is not None:
            with nc.Dataset(input_prd_fpaths[prd_mnk][0], 'r') as ds:
                for v_t in v_tl:
                    v_group, v_names = v_t
                    ds_g = ds.groups[v_group]
                    for vn in v_names:
                        inpd[vn] = ds_g[vn][...]
        else:
            for v_t in v_tl:
                _, v_names = v_t
                for vn in v_names:
                    inpd[vn] = None

    # Define output dimensions:

    with nc.Dataset(input_prd_fpaths[primary_prd_mnk][0], 'r') as ds:
        num_scenes = ds.dimensions["xtrack"].size
        num_channels = ds.dimensions["spectral"].size
        product_full_version = ds.full_versionID
        spacecraft_ID = ds.spacecraft_ID
        sensor_ID = ds.sensor_ID

    num_sortm = fld_d["proc_method"](inpd, diminfo_only=True)  # Sort metric dim
    num_lats = len(lat_midvalues)  # latitude dimension
    num_lons = len(lon_midvalues)  # longitude dimension

    # Instantiate cumulative statistics arrays (masked arrays, all data zero):

    counts = np.ma.zeros(
                     (num_scenes, num_sortm, num_lats, num_lons, num_channels),
                         dtype="int32")
    field_sum = np.ma.zeros(
                     (num_scenes, num_sortm, num_lats, num_lons, num_channels))
    field_sumsquares = np.ma.zeros(
                     (num_scenes, num_sortm, num_lats, num_lons, num_channels))

    asc_counts = np.ma.zeros(
                     (num_scenes, num_sortm, num_lats, num_lons, num_channels),
                             dtype="int32")
    asc_field_sum = np.ma.zeros(
                     (num_scenes, num_sortm, num_lats, num_lons, num_channels))
    asc_field_sumsquares = np.ma.zeros(
                     (num_scenes, num_sortm, num_lats, num_lons, num_channels))

    desc_counts = np.ma.zeros(
                     (num_scenes, num_sortm, num_lats, num_lons, num_channels),
                              dtype="int32")
    desc_field_sum = np.ma.zeros(
                     (num_scenes, num_sortm, num_lats, num_lons, num_channels))
    desc_field_sumsquares = np.ma.zeros(
                     (num_scenes, num_sortm, num_lats, num_lons, num_channels))

    tmpgn = [os.path.basename(x).split('.')[0][-5:]
                                    for x in input_prd_fpaths[primary_prd_mnk]]
    tmpgn_num = np.array([int(x) for x in tmpgn], dtype="int32")
    tmpgn_num_missing = [i for x, y in zip(tmpgn_num, tmpgn_num[1:])
                                         for i in range(x + 1, y) if y - x > 1]
    tmpgn_missing = [f"{x:05d}" for x in tmpgn_num_missing]
    input_product_files_str = (
                    "{} (granule_ID {} to {}; missing {}), and any associated "
                    "AUX-MET, AUX-SAT".format(primary_prd_mnk, tmpgn[0],
                    tmpgn[-1], ','.join(tmpgn_missing)))

    # Read in and further process time-<dependent> fields:
    for ign in range(n_input_prd_fpaths):
        print("granule", ign)

        # Read in time-dependent data associated with this granule:
        for prd_mnk, v_tl in fld_d["tdep_input"]:
            if input_prd_fpaths[prd_mnk][ign] is not None:
                with nc.Dataset(input_prd_fpaths[prd_mnk][ign], 'r') as ds:
                    for v_t in v_tl:
                        v_group, v_names = v_t
                        ds_g = ds.groups[v_group]
                        for vn in v_names:
                            inpd[vn] = ds_g[vn][...]
            else:
                for v_t in v_tl:
                    _, v_names = v_t
                    for vn in v_names:
                        inpd[vn] = None

        # Determine fields to process further:
        procd = fld_d["proc_method"](inpd, t_msk[ign])
        if procd is None:
            continue  # Insufficient data -- proceed to any next ign iteration

        # Bin/count/process fields:
        for i in range(len(procd["nominal_lats"])):
            isort = procd["sort_metric"][i]-1  # integer
            iscene = procd["nominal_scenes"][i]  # integer
            iframe = procd["nominal_frames"][i]  # integer

            field = procd["nominal_field"][i,:]
            field_sq = field*field  # Squared

            ilat = int((procd["nominal_lats"][i]-lat_lower)/dlat)
            ilon = max(0,
               min(int((procd["nominal_lons"][i]-lon_lower)/dlon), num_lons-1))

            # Modify full granule arrays

            counts[iscene, isort, ilat, ilon, :] += 1
            field_sum[iscene, isort, ilat, ilon, :] += field
            field_sumsquares[iscene, isort, ilat, ilon, :] += field_sq

            if procd["sat_pass_type"][i] > 0:  # Modify ascending pass arrays
                asc_counts[iscene, isort, ilat, ilon, :] += 1
                asc_field_sum[iscene, isort, ilat, ilon, :] += field
                asc_field_sumsquares[iscene, isort, ilat, ilon, :] += field_sq
            else:  # Modify descending pass arrays
                desc_counts[iscene, isort, ilat, ilon, :] += 1
                desc_field_sum[iscene, isort, ilat, ilon, :] += field
                desc_field_sumsquares[iscene, isort, ilat, ilon, :] += field_sq

    # Mask zero-count array elements:

    msk = ( counts == 0 )
    counts = np.ma.masked_where(msk, counts, copy=False)
    field_sum = np.ma.masked_where(msk, field_sum, copy=False)
    field_sumsquares = np.ma.masked_where(msk, field_sumsquares, copy=False)

    msk = ( asc_counts == 0 )
    asc_counts = np.ma.masked_where(msk, asc_counts, copy=False)
    asc_field_sum = np.ma.masked_where(msk, asc_field_sum, copy=False)
    asc_field_sumsquares = np.ma.masked_where(msk, asc_field_sumsquares,
                                              copy=False)

    msk = ( desc_counts == 0 )
    desc_counts = np.ma.masked_where(msk, desc_counts, copy=False)
    desc_field_sum = np.ma.masked_where(msk, desc_field_sum, copy=False)
    desc_field_sumsquares = np.ma.masked_where(msk, desc_field_sumsquares,
                                               copy=False)

    # Calculate the average field values by dividing the total summed field
    #  values for each bin by the total counts for that bin
    field_avg = field_sum/counts
    asc_field_avg = asc_field_sum/asc_counts
    desc_field_avg = desc_field_sum/desc_counts

    # Calculate the standard deviation of the field values per bin using
    #  the standard deviation function defined above
    field_stdev = stdev_func(counts, field_avg, field_sum, field_sumsquares)
    asc_field_stdev = stdev_func(asc_counts, asc_field_avg, asc_field_sum,
                                 asc_field_sumsquares)
    desc_field_stdev = stdev_func(desc_counts, desc_field_avg,
                                  desc_field_sum, desc_field_sumsquares)

    # Write data to a NetCDF-format file:

    o_data = {}

    latitude = np.broadcast_to(lat_midvalues[:,np.newaxis],
                               (num_lats, num_lons))
    longitude = np.broadcast_to(lon_midvalues[np.newaxis,:],
                                (num_lats, num_lons))
    
    arrays_list = [latitude, longitude, inpd["wavelength"],
                   inpd["idealized_wavelength"],
                   procd["ref_sortmetric_values"],
                   field_avg, asc_field_avg, desc_field_avg,
                   field_stdev, asc_field_stdev, desc_field_stdev,
                   counts, asc_counts, desc_counts,
                   field_sum, asc_field_sum, desc_field_sum,
                 field_sumsquares, asc_field_sumsquares, desc_field_sumsquares]
    
    arrays_names = fld_d["output_array_names"]
    
    for i in range(len(arrays_list)):
        o_data[arrays_names[i]] = arrays_list[i]

    output = {}

    output[fld_d["output_data_group"]] = o_data

    global_atts = {}

    global_atts["granule_ID"] = "not applicable"
    global_atts["spacecraft_ID"] = spacecraft_ID
    global_atts["sensor_ID"] = sensor_ID

    global_atts["ctime_coverage_start_s"] = ct_coverage_start
    global_atts["ctime_coverage_end_s"] = ct_coverage_end
    UTC_DT_t, _ = ctime_to_UTC_DT([ct_coverage_start, ct_coverage_end], 's',
                                  leap_s_info)
    global_atts["UTC_coverage_start"] = (
                                  UTC_DT_t[0].strftime("%Y-%m-%dT%H:%M:%S.%f"))
    global_atts["UTC_coverage_end"] = (
                                  UTC_DT_t[1].strftime("%Y-%m-%dT%H:%M:%S.%f"))

    with open(PREFIRE_L3.filepaths.scipkg_prdgitv_fpath, 'r') as in_f:
        line_parts = in_f.readline().split('(', maxsplit=1)
        global_atts["provenance"] = "{}{} ( {}".format(line_parts[0],
                                                       product_full_version,
                                                       line_parts[1].strip())

    with open(PREFIRE_L3.filepaths.scipkg_version_fpath) as f:
        global_atts["processing_algorithmID"] = f.readline().strip()

    global_atts["input_product_files"] = input_product_files_str

    global_atts["full_versionID"] = product_full_version
    global_atts["archival_versionID"] = (
                           product_full_version.split('_')[0].replace('R', ''))
    global_atts["netCDF_lib_version"] = nc.getlibversion().split()[0]

    # Generate L3 output file name:
    replacements = str.maketrans({'-': '', ':': '', 'T': '', 'Z': ''})
    UTC_rep = [x.translate(replacements) for x in UTC_dtrep_range_to_proc]

    L3_fname = "raw-PREFIRE_SAT{:1d}_3-{}_{}_{}_{}.nc".format(cfg_d["sat_num"],
                   cfg_d["L3_field_moniker"], product_full_version, UTC_rep[0],
                                                          UTC_rep[1])
    global_atts["file_name"] = L3_fname

    now_UTC_DT = datetime.datetime.now(datetime.timezone.utc)
    global_atts["UTC_of_file_creation"] = now_UTC_DT.strftime(
                                                        "%Y-%m-%dT%H:%M:%S.%f")

    # Add global and any group attributes to output dictionary:
    output["Global_Attributes"] = global_atts

    product_specs_fpath = os.path.join(
                               PREFIRE_L3.filepaths.package_ancillary_data_dir,
                                   "L3_product_filespecs.json")

    L3_fpath = str(cfg_d["output_product_Path"] / L3_fname)
    os.makedirs(os.path.abspath(str(cfg_d["output_product_Path"])),
                                exist_ok=True)

    write_data_fromspec(output, L3_fpath, product_specs_fpath,
                        use_shared_geometry_filespecs=False, verbose=True)


if __name__ == "__main__":
    # Process arguments:
    arg_description = ("Create a Level 3 (sorted by some metric) product "
                       "from PREFIRE data product granules.")
    arg_parser = argparse.ArgumentParser(description=arg_description)
    arg_parser.add_argument("L3_field_moniker", type=str,
           help=("The moniker of the specific Level 3 product to be created "
                 "(e.g., 'SFC-SORTED-ALLSKY')."))
    arg_parser.add_argument("input_product_dir", type=str,
          help=("The path (directory name) in which the input PREFIRE data "
                "product subdirectories (e.g., 2B-SFC/, AUX-MET/) can be "
                "found."))
    arg_parser.add_argument("sat_num", help="Which PREFIRE CubeSat {1 or 2}?")
    arg_parser.add_argument("UTC_dtrep_range_to_proc",
           help="The range of UTC datetimes to process; e.g., can be "
                "'2024-09-16T00:00:00Z,END', 'START,END', "
                "'START,2024-09-16T00:00:00Z', 'END-6d,END', "
                "or '2024-09-16T00:00:00Z,2024-10-20T00:00:00Z'")
    arg_parser.add_argument("output_product_dir", type=str,
          help=("The path (directory name) in which the output PREFIRE "
                "Level 3 data product subdirectory (e.g., "
                "3-SFC-SORTED-ALLSKY/) will be or is."))

    args = arg_parser.parse_args()

    # Construct configuration dictionary:

    cfg_d = {}

    cfg_d["append_to_syspath"] = ".."

    cfg_d["L3_field_moniker"] = args.L3_field_moniker
    cfg_d["input_product_Path"] = Path(args.input_product_dir)
    cfg_d["output_product_Path"] = Path(args.output_product_dir)

    cfg_d["sat_num"] = int(args.sat_num)

    now_UTC_DT = datetime.datetime.now(datetime.timezone.utc)
    now_UTC_str = now_UTC_DT.strftime("%Y-%m-%dT%H:%M:%SZ")
    s_str, e_str = [x.strip() for x in args.UTC_dtrep_range_to_proc.split(',')]
    if s_str.lower() == "start":
        if cfg_d["sat_num"] == 1:  # SAT1 (first granule after IOC)
            s_str = "2024-07-24T21:46:00Z"
        else:  # SAT2 (first granule after IOC)
            s_str = "2024-06-29T21:19:00Z"
    elif s_str.lower() == "end-6d":
        tmp_UTC_DT = now_UTC_DT-datetime.timedelta(days=6)
        s_str = tmp_UTC_DT.strftime("%Y-%m-%dT%H:%M:%SZ")
    if e_str.lower() == "end":
        e_str = now_UTC_str
    cfg_d["UTC_dtrep_range_to_proc"] = (s_str, e_str)

    cfg_d["cfg_end_UTC_exclusive"] = True

    # Create the Level 3 sorted data product:
    create_L3_sorted(cfg_d)
