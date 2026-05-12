import yaml
import argparse
import numpy as np
import healpy as hp
import heracles
import ispice
import os.path
import time
import heracles.dices as dices


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
    args = parser.parse_args()
    print(f"Using method: {args.mask_type}")
    
    # Config
    # export HEALPIX=/home/jaimerzp/Documents/UCL/Healpix_3.83_2024Nov13/Healpix_3.83/
    config_path = "./sims_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    n = 200
    nside = 2048 #config['nside']
    lmax_full = 2000 #config['lmax_full']
    lmax_partial = 4000 #config['lmax_partial']
    lmin = config['lmin']
    mode = args.mode  # "lognormal" or "gaussian"
    recompute = args.recompute
    mask_type = args.mask_type  # Default to 'Patch' if not specified
    path = f"/pscratch/sd/j/jaimerz/{mode}_sims/{mask_type}/"
    l = np.arange(lmax_partial + 1)
    
    cls = {}
    total_time = 0

    for i in range(1, n+1):
        start = time.perf_counter()

        if os.path.isfile(f"{path}/cls_pols/cls_data_pols_{i}_lmax_{lmax_partial}.fits") and recompute=="False":
            print(f"Skipping sim {i}")
            _cls_pols = heracles.read(f"{path}/cls_pols/cls_data_pols_{i}_lmax_{lmax_partial}.fits")
            _cls_pols['POS', 'POS', 1, 1] = heracles.Result(_cls_pols['POS', 'POS', 1, 1].array,
                                                           axis=(0,), spin=(0, 0), ell=l)
            _cls_pols['POS', 'SHE', 1, 1] = heracles.Result(_cls_pols['POS', 'SHE', 1, 1].array,
                                                           axis=(1,), spin=(0, 2), ell=l)
            _cls_pols['SHE', 'SHE', 1, 1] = heracles.Result(_cls_pols['SHE', 'SHE', 1, 1].array,
                                                           axis=(2,), spin=(2, 2), ell=l)
            cls[i] = _cls_pols
        else:
            print(f"Unmixing sim {i}")
            # Load maps
            sim_path = f"/pscratch/sd/j/jaimerz/{mode}_sims/{mode}_sim_{i}_nside_{nside}"
            POS1 = heracles.read_maps(f"{sim_path}/POS_1.fits")['POS', 1]
            SHE1 = heracles.read_maps(f"{sim_path}/SHE_1.fits")['SHE', 1]
            if np.iscomplexobj(SHE1):
                map_q = SHE1.real
                map_u = SHE1.imag
            else:
                map_q = SHE1[0]
                map_u = SHE1[1]
            if np.mean(POS1)/np.std(POS1) > 0.1:
                POS1 = (POS1 - np.mean(POS1))/np.mean(POS1)
            maps_spice = [POS1, map_q, map_u]
            hp.write_map(f"{sim_path}/maps_spice.fits", maps_spice, overwrite=True)
            ispice.ispice(f"{sim_path}/maps_spice.fits",
                        f"{path}/cls_pols/cls_data_pols_l1max_{lmax_partial}.fits",
                        maskfile1=f"/pscratch/sd/j/jaimerz/masks/{mask_type}_mask_nside_{nside}.fits",
                        nlmax=lmax_partial,
                        polarization=True,
                        decouple=True,
                        symmetric_cl=True,
                        covfileout=f"{path}/cls_pols/cov_data_pols_{i}_l1max_{lmax_partial}.fits",
                        kernelsfileout=f"{path}/cls_pols/kernels_pols_{i}_l1max_{lmax_partial}.fits",
                        apodizesigma=10.0,
                        pixelfile=False,
                        #cl_outmask_file=f"{path}/cls_pols/cls_mask_pols_decoupled_{i}.fits",
                        #cl_outmap_file=f"{path}/cls_pols/cls_data_pols_raw_{i}.fits",
                        #corfile=f"{path}/cls_pols/corr_data_pols_{i}.fits",
                        binpath="/global/homes/j/jaimerz/.conda/envs/pols/bin/spice")
        
            # reformat cls
            pols_cl = hp.read_cl(path+f"cls_pols/cls_data_pols_l1max_{lmax_partial}.fits")
            _pols_cl = {}
            _pols_cl['POS', 'POS', 1, 1] = heracles.Result(pols_cl[0], axis=(0,), spin=(0, 0), ell=l)
            _pols_cl['POS', 'SHE', 1, 1] = heracles.Result(
                np.array([pols_cl[3], pols_cl[4]]), axis=(1,), spin=(0, 2), ell=l)
            _pols_cl['SHE', 'SHE', 1, 1] = heracles.Result(
                np.array([[pols_cl[1], pols_cl[5]],
                          [pols_cl[5], pols_cl[2]]]),
                spin=(2, 2), axis=(2,), ell=l)
            # Save cls
            heracles.write(f"{path}/cls_pols/cls_data_pols_{i}_lmax_{lmax_partial}.fits", _pols_cl)
            cls[i] = _pols_cl

        total_time += time.perf_counter() - start

    avg_time = total_time / n
    print(f"Average time per iteration: {avg_time:.3f} seconds")
    
    # Binning cls
    nlbins = config.get('nlbins', 20)  # Default to 20 if not specified
    ledges = np.logspace(np.log10(lmin), np.log10(lmax_full), nlbins + 1)
    lgrid = (ledges[1:] + ledges[:-1]) / 2
    cqs = heracles.binned(cls, ledges)
    
    # Covariance
    print("Computing covariance")
    cqs_cov = dices.jackknife_covariance(cqs, nd=0)
    
    # Save
    print("Saving covariances")
    heracles.write(path+f"/covs/cov_pols_cqs_lmin_{lmin}_l1max_{lmax_full}.fits", cqs_cov)

if __name__ == "__main__":
    main()