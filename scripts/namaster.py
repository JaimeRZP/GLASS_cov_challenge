import yaml
import argparse
import os.path
import numpy as np
import healpy as hp
import heracles
import pymaster as nmt
import heracles.dices as dices
import time 


# conda activate nmt
def compute_master(f_a, f_b, wsp):
    cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
    cl_decoupled = wsp.decouple_cell(cl_coupled)
    return cl_decoupled

def main():
    # Config from command line
    parser = argparse.ArgumentParser(description="Mask type")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["gaussian", "lognormal", "gatti"],
        help="sim type."
    )
    parser.add_argument(
        "--mask_type",
        type=str,
        required=True,
        choices=["rr2", "dr1", "patch", "tr1"],
        help="mask type."
    )
    parser.add_argument(
        "--recompute",
        default="False",
        help="recompute cls."
    )
    parser.add_argument(
        "--wb",
        default="False",
        help="recompute cls."
    )
    args = parser.parse_args()
    print(f"Using method: {args.mask_type}")
    
    # Config
    config_path = "./sims_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    n = 200
    nside = 2048 #config['nside']
    lmin = config['lmin']
    lmax_full = 2000
    lmax_partial = 4000 #config['lmax_full']
    nlbins = config.get('nlbins', 20)
    extra_nlbins = 4
    mode = args.mode
    mask_type = args.mask_type
    recompute = args.recompute
    wb = args.wb
    path = f"/pscratch/sd/j/jaimerz/{mode}_sims/{mask_type}/"
    
    ls = np.arange(lmax_full + 1)
    ledges = np.logspace(np.log10(lmin), np.log10(lmax_full), nlbins + 1)
    extra_ledges = np.logspace(np.log10(lmax_full), np.log10(lmax_partial), extra_nlbins + 1)
    ledges = np.append(ledges, extra_ledges[1:])
    lgrid = ledges[:-1] + np.diff(ledges)/2
    b = nmt.NmtBin.from_edges(ledges[:-1].astype(int), ledges[1:].astype(int))
    b.lmax = lmax_partial
    
    mask = hp.read_map(f"/pscratch/sd/j/jaimerz/masks/{mask_type}_mask_nside_{nside}.fits")
    mask_apo = nmt.mask_apodization(mask, 0.0, apotype='C1')
    
    # workspaces
    sim_path = f"/pscratch/sd/j/jaimerz/{mode}_sims/{mode}_sim_1_nside_{nside}"
    POS1 = heracles.read_maps(f"{sim_path}/POS_1.fits")['POS', 1]
    if wb == "True":
        SHE1 = heracles.read_maps(f"{sim_path}/SHE_1_wb.fits")['SHE', 1]
    else:
        SHE1 = heracles.read_maps(f"{sim_path}/SHE_1.fits")['SHE', 1]
    map_t = POS1
    if np.iscomplexobj(SHE1):
        map_q = SHE1.real
        map_u = SHE1.imag
    else:
        map_q = SHE1[0]
        map_u = SHE1[1]
    if np.mean(map_t)/np.std(map_t) > 0.1:
        map_t = (map_t - np.mean(map_t))/np.mean(map_t)
    f0 = nmt.NmtField(mask_apo, [map_t], lmax=lmax_partial)
    f2 = nmt.NmtField(mask_apo, [map_q, map_u], lmax=lmax_partial)
    w00 = nmt.NmtWorkspace.from_fields(f0, f0, b)
    w02 = nmt.NmtWorkspace.from_fields(f0, f2, b)
    w22 = nmt.NmtWorkspace.from_fields(f2, f2, b)
    
    cls = {}

    total_time = 0  # <-- added

    for i in range(1, n+1):
        start = time.perf_counter()  # <-- added

        if wb == "True":
            cls_path = f"{path}/cls_nmt/cqs_data_nmt_np_{i}_lmin_{lmin}_lmax_{lmax_partial}_wb.fits"
        else:
            cls_path = f"{path}/cls_nmt/cqs_data_nmt_np_{i}_lmin_{lmin}_lmax_{lmax_partial}.fits"
        if os.path.isfile(cls_path) and recompute == "False":
            print(f"Skipping sim {i}", end='\r')
            _cls_nmt = heracles.read(cls_path)
            _cls_nmt['POS', 'POS', 1, 1] = heracles.Result(_cls_nmt['POS', 'POS', 1, 1].array,
                                                           axis=(0,), spin=(0, 0), ell=lgrid)
            _cls_nmt['POS', 'SHE', 1, 1] = heracles.Result(_cls_nmt['POS', 'SHE', 1, 1].array,
                                                           axis=(1,), spin=(0, 2), ell=lgrid)
            _cls_nmt['SHE', 'SHE', 1, 1] = heracles.Result(_cls_nmt['SHE', 'SHE', 1, 1].array,
                                                           axis=(2,), spin=(2, 2), ell=lgrid)
            cls[i] = _cls_nmt
        else:
            print(f"Unmixing sim {i}", end='\r')
            sim_path = f"/pscratch/sd/j/jaimerz/{mode}_sims/{mode}_sim_{i}_nside_{nside}"
            POS1 = heracles.read_maps(f"{sim_path}/POS_1.fits")
            if wb == "True":
                SHE1 = heracles.read_maps(f"{sim_path}/SHE_1_wb.fits")['SHE', 1]
            else:
                SHE1 = heracles.read_maps(f"{sim_path}/SHE_1.fits")['SHE', 1]
            map_t = POS1['POS', 1]
            if np.iscomplexobj(SHE1):
                map_q = SHE1.real
                map_u = SHE1.imag
            else:
                map_q = SHE1[0]
                map_u = SHE1[1]
            if np.mean(map_t)/np.std(map_t) > 0.1:
                map_t = (map_t - np.mean(map_t))/np.mean(map_t)
            f0 = nmt.NmtField(mask_apo, [map_t])
            f2 = nmt.NmtField(mask_apo, [map_q, map_u])
            cls_00 = compute_master(f0, f0, w00)
            cls_02 = compute_master(f0, f2, w02)
            cls_22 = compute_master(f2, f2, w22)
            _cls_nmt= {}
            _cls_nmt['POS', 'POS', 1, 1] = heracles.Result(cls_00[0], axis=(0,), spin=(0, 0), ell=lgrid)
            _cls_nmt['POS', 'SHE', 1, 1] = heracles.Result(cls_02, axis=(1,), spin=(0, 2), ell=lgrid)
            _cls_nmt['SHE', 'SHE', 1, 1] = heracles.Result(
                np.array([[cls_22[0], cls_22[1]],
                          [cls_22[2], cls_22[3]]]),
                spin=(2, 2), axis=(2,), ell=lgrid)
            cls[i] = _cls_nmt
            heracles.write(cls_path, _cls_nmt)

        total_time += time.perf_counter() - start  # <-- added

    print("Done")

    avg_time = total_time / n  # <-- added
    print(f"Average time per iteration: {avg_time:.3f} seconds")  # <-- added
    
    # Compute covariance
    print("Computing covariance")
    nmt_cqs_cov = dices.jackknife_covariance(cls, nd=0)
    
    # Save
    print("Saving covariance")
    if wb == "True":
        heracles.write(path+f"/covs/cov_nmt_cqs_lmin_{lmin}_l1max_{lmax_partial}_wb.fits", nmt_cqs_cov)
    else:
        heracles.write(path+f"/covs/cov_nmt_cqs_lmin_{lmin}_l1max_{lmax_partial}.fits", nmt_cqs_cov)

if __name__ == "__main__":
    main()