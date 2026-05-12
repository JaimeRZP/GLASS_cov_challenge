import yaml
import heracles
import argparse
import os.path
import numpy as np
import time
from heracles import dices
from dataclasses import replace
from heracles.twopoint import invert_mixing_matrix, apply_mixing_matrix


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
        "--rtol",
        type=float,
        required=False,
        help="recompute cls."
    )
    parser.add_argument(
        "--recompute",
        default="False",
        help="recompute cls."
    )
    args = parser.parse_args()
    print(f"Using method: {args.mask_type}")
    
    # Config
    config_path = "./sims_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    n = 200 #config['nsims']
    nside = 2048 #config['nside']
    lmin = config['lmin']
    lmax_partial = 4000 #config['lmax_partial']
    lmax_full = 2000 #config['lmax_full']
    lmax_mask = 6000 #config['lmax_mask']
    mode = args.mode
    mask_type = args.mask_type  # Default to 'Patch' if not specified
    path = f"/pscratch/sd/j/jaimerz/{mode}_sims/{mask_type}"
    recompute = args.recompute
    print(f"Recompute: {recompute}")

    #options
    options = {}
    if mask_type == "dr1":
        options[('POS', 'POS', 1, 1)] = 0.001
        options[('POS', 'SHE', 1, 1)] = 0.001 
        options[('SHE', 'SHE', 1, 1)] = 0.001
    if mask_type == "tr1":
        options[('POS', 'POS', 1, 1)] = 0.005
        options[('POS', 'SHE', 1, 1)] = 0.005
        options[('SHE', 'SHE', 1, 1)] = 0.005
    if mask_type == "patch":
        options[('POS', 'POS', 1, 1)] = 0.01 #0.0016 # 0.0335
        options[('POS', 'SHE', 1, 1)] = 0.01 #0.0012 # 0.0010
        options[('SHE', 'SHE', 1, 1)] = 0.01 #0.0010 # 0.0536
    
    # Compute mixing matrix
    mms = heracles.read(f"{path}/mixmat_l1max_{lmax_partial}_l2max_{lmax_mask}.fits")
    inv_mms = invert_mixing_matrix(mms, rcond=options)
    heracles.write(f"{path}/inv_mixmat_l1max_{lmax_partial}_l2max_{lmax_mask}.fits", inv_mms)
        
    cls = {}
    total_time = 0

    for i in range(1, n+1):
        start = time.perf_counter()

        cl_path = f"{path}/cls_inv/cls_data_naive_inv_{i}_l1max_{lmax_partial}_l2max_{lmax_mask}.fits"
        if os.path.isfile(cl_path) and recompute=="False":
            inv_cls = heracles.read(cl_path)
        else:
            print(f"Unmixing sim {i}")
            # Load cls
            data_cls = heracles.read(f"{path}/cls/cls_data_{i}_lmax_{lmax_partial}.fits")
            inv_cls = apply_mixing_matrix(data_cls, inv_mms)
            # Save cls
            heracles.write(cl_path, inv_cls)

        cls[i] = inv_cls
        total_time += time.perf_counter() - start

    print("Done")
    avg_time = total_time / n
    print(f"Average time per iteration: {avg_time:.3f} seconds")
    
    # Binning cls
    nlbins = config.get('nlbins', 20)  # Default to 20 if not specified
    ledges = np.logspace(np.log10(lmin), np.log10(lmax_full), nlbins + 1)
    lgrid = (ledges[1:] + ledges[:-1]) / 2
    print(f"Using {len(lgrid)} bins for the covariance matrix.")
    cqs = heracles.binned(cls, ledges)
    
    # compute covariance
    print("Computing covariance")
    #cls_cov = dices.jackknife_covariance(cls, nd=0)
    cqs_cov = dices.jackknife_covariance(cqs, nd=0)
    
    # Save
    print("Saving covariances")
    #heracles.write(path+f"/covs/cov_naive_inv_cls_l1max_{lmax}_l2max_{lmax_mask}.fits", cls_cov)
    heracles.write(path+f"/covs/cov_naive_inv_cqs_lmin_{lmin}_l1max_{lmax_full}_l2max_{lmax_mask}.fits", cqs_cov)

if __name__ == "__main__":
    main()