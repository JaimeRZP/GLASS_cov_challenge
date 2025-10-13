import yaml
import numpy as np
import glass
import camb
import camb.sources
import heracles

# Config
config_path = "./sims_config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
lmax = config['lmax']
mode = config['mode']  # "lognormal" or "gaussian"

path = f"../{mode}_sims"
nbins = 2
h = 0.7
Oc = 0.25
Ob = 0.05

# make nz's
z = np.arange(0.0, 5.01, 0.01)
dndz = glass.smail_nz(z, 1.0, 1.5, 2.0)
zbins = glass.equal_dens_zbins(z, dndz, nbins)
nz = glass.tomo_nz_gausserr(z, dndz, 0.05, zbins)
nz_1 = nz[:2]
nz_2 = nz[2:]

print(f"saving nz's to {path}/nzs.npy")
np.savez(
    f"{path}/nzs.npz",
    z=z,
    nz_1=nz_1,
    nz_2=nz_2,
    )

# make a cosmology
pars = camb.set_params(
    H0=100 * h,
    omch2=Oc * h**2,
    ombh2=Ob * h**2,
    NonLinear=camb.model.NonLinear_both,
)
pars.set_accuracy(AccuracyBoost=2.0, lAccuracyBoost=2.0, lSampleBoost=2.0)
pars.Want_CMB = False
pars.Want_CMB_lensing = False
pars.min_l = 1
pars.set_for_lmax(2 * lmax)

pars.SourceWindows = [
    camb.sources.SplinedSourceWindow(z=z, W=nz_i, source_type="counts") for nz_i in nz_1
] + [
    camb.sources.SplinedSourceWindow(z=z, W=nz_i, source_type="lensing") for nz_i in nz_2
]

# Make theory cls
cls_dict = camb.get_results(pars).get_source_cls_dict(lmax=lmax, raw_cl=True)
cls = [cls_dict[f"W{i+1}xW{j+1}"] for i, j in glass.spectra_indices(nbins)]

# Turn into heracles results
results = {}
for key in cls_dict.keys():
    results[key] = heracles.Result(cls_dict[key])
heracles.write(f"{path}/cls_theory_lmax_{lmax}.fits", results)
