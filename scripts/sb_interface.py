import numpy as np
import heracles
import yaml
import subprocess
import os


def write_cl_tab(ascii_folder, ascii_filename, cl_3d, ells, zbins):

    with open(f'{ascii_folder}/{ascii_filename}', 'w') as file:
        file.write(f'#ell\t\tzi\tzj\t{ascii_filename}\n')
        for ell_idx, ell_val in enumerate(ells):
            for zi in range(zbins):
                for zj in range(zbins):
                    value = cl_3d[ell_idx, zi, zj]
                    file.write(f"{ell_val:.3f}\t\t{zi}\t{zj}\t{value:.10e}\n")


# ===== SETTINGS ========
path_to_sb_main = '/home/davide/Documenti/Lavoro/Programmi/Spaceborne/main.py'
path_to_sb_cfg = '/home/davide/Documenti/Lavoro/Programmi/GLASS_cov_challenge/config_spaceborne_out.yaml'
data_sb_path = '../data/data_sb'
Nbins = 2
ls = np.arange(256 + 1)

with open('../config_spaceborne_in.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

# TODO Jaime: there are actually important, check values used for the challenge
cfg['nz']['ngal_sources'] = [10., 10.]  # gal/arcmin^2/zbin
cfg['nz']['ngal_lenses'] = [10., 10.]  # gal/arcmin^2/zbin
cfg['covariance']['sigma_eps_i'] = 0.
# ===== END SETTINGS ========

cfg['nz']['dzGC'] = [0.] * Nbins
cfg['nz']['dzWL'] = [0.] * Nbins
cfg['misc']['output_path'] = data_sb_path

# ell binning
cfg['ell_binning']['ell_min'] = int(ls.min())
cfg['ell_binning']['ell_max_3x2pt'] = int(ls.max())
cfg['ell_binning']['ell_max_GC'] = int(ls.max())
cfg['ell_binning']['ell_max_WL'] = int(ls.max())
cfg['ell_binning']['ell_max_WL_opt'] = int(ls.max())
cfg['ell_binning']['nbl_WL_opt'] = int(len(ls))
cfg['ell_binning']['binning_type'] = 'unbinned'

# cells
cfg['C_ell']['cl_LL_path'] = f'{data_sb_path}/cl_ll_tab.txt'
cfg['C_ell']['cl_GL_path'] = f'{data_sb_path}/cl_gl_tab.txt'
cfg['C_ell']['cl_GG_path'] = f'{data_sb_path}/cl_gg_tab.txt'
cfg['C_ell']['mult_shear_bias'] = [0] * Nbins
cfg['C_ell']['mult_shear_bias'] = [0] * Nbins
cfg['C_ell']['mult_shear_bias'] = [0] * Nbins

# n(z) - THIS IS USED ONLY FOR THE NON-GAUSSIAN COVARIANCE
z_nz = np.load('/home/davide/Documenti/Lavoro/Programmi/GLASS_cov_challenge/data/nzs.npz')['z']
nz_1 = np.load('/home/davide/Documenti/Lavoro/Programmi/GLASS_cov_challenge/data/nzs.npz')['nz_1']
nz_2 = np.load('/home/davide/Documenti/Lavoro/Programmi/GLASS_cov_challenge/data/nzs.npz')['nz_2']
# plt.plot(z_nz, nz_1[0, :], label='nz_1[0]')
# plt.plot(z_nz, nz_1[1, :], label='nz_1[1]')
# plt.plot(z_nz, nz_2[0, :], label='nz_2[0]')
# plt.plot(z_nz, nz_2[1, :], label='nz_2[1]')
nz_sb_fmt = np.concatenate((z_nz[:, None], nz_1.T), axis=1)
np.savetxt(f'{data_sb_path}/nz_sb_fmt.txt', nz_sb_fmt)
cfg['nz']['nz_lenses_filename'] = f'{data_sb_path}/nz_sb_fmt.txt'
cfg['nz']['nz_sources_filename'] = f'{data_sb_path}/nz_sb_fmt.txt'

cfg['covariance']['split_gaussian_cov'] = False

# ================== HERACLES Cls ==================
theory_cls = heracles.read("../data/theory_cls.fits")
_theory_cls = {}
_theory_cls[("POS", "POS", 1, 1)] = theory_cls["W1xW1"]
_theory_cls[("POS", "POS", 2, 2)] = theory_cls["W2xW2"]
_theory_cls[("G_E", "G_E", 1, 1)] = theory_cls["W3xW3"]
_theory_cls[("G_E", "G_E", 2, 2)] = theory_cls["W4xW4"]
_theory_cls[("POS", "G_E", 1, 1)] = -theory_cls["W1xW3"].__array__()
_theory_cls[("POS", "G_E", 2, 2)] = -theory_cls["W2xW4"].__array__()
_theory_cls[("POS", "POS", 1, 2)] = theory_cls["W1xW2"]
_theory_cls[("POS", "G_E", 1, 2)] = -theory_cls["W1xW4"].__array__()
_theory_cls[("POS", "G_E", 2, 1)] = -theory_cls["W2xW3"].__array__()
_theory_cls[("G_E", "G_E", 1, 2)] = theory_cls["W3xW4"]


__theory_gg = np.zeros((len(ls), Nbins, Nbins))
__theory_gs = np.zeros((len(ls), Nbins, Nbins))
__theory_ss = np.zeros((len(ls), Nbins, Nbins))
for i in range(1, Nbins + 1):
    for j in range(i, Nbins + 1):
        __theory_gg[:, i - 1, j - 1] = _theory_cls[("POS", "POS", i, j)]
        __theory_gs[:, j - 1, i - 1] = _theory_cls[("POS", "POS", i, j)]

        __theory_gs[:, i - 1, j - 1] = _theory_cls[("POS", "G_E", i, j)]
        __theory_gs[:, j - 1, i - 1] = _theory_cls[("POS", "G_E", i, j)]

        __theory_ss[:, i - 1, j - 1] = _theory_cls[("G_E", "G_E", i, j)]
        __theory_ss[:, j - 1, i - 1] = _theory_cls[("G_E", "G_E", i, j)]

write_cl_tab(data_sb_path, 'cl_ll_tab.txt', __theory_ss, ls, Nbins)
write_cl_tab(data_sb_path, 'cl_gl_tab.txt', __theory_gs, ls, Nbins)
write_cl_tab(data_sb_path, 'cl_gg_tab.txt', __theory_gg, ls, Nbins)

with open(path_to_sb_cfg, 'w') as f:
    yaml.dump(cfg, f)

subprocess.run(['python', path_to_sb_main, '--config', path_to_sb_cfg], check=True)

# TODO should we add G_B?
probe_dict = {
    0: 'G_E',
    1: 'POS',
}

cov_10d = np.zeros((2, 2, 2, 2, len(ls), len(ls), Nbins, Nbins, Nbins, Nbins))

cov_10d[0, 0, 0, 0] = np.load(f'{data_sb_path}/cov_LLLL_G_6D.npz')['arr_0']
cov_10d[0, 0, 1, 0] = np.load(f'{data_sb_path}/cov_LLGL_G_6D.npz')['arr_0']
cov_10d[0, 0, 1, 1] = np.load(f'{data_sb_path}/cov_LLGG_G_6D.npz')['arr_0']
cov_10d[1, 0, 1, 0] = np.load(f'{data_sb_path}/cov_GLGL_G_6D.npz')['arr_0']
cov_10d[1, 0, 1, 1] = np.load(f'{data_sb_path}/cov_GLGG_G_6D.npz')['arr_0']
cov_10d[1, 1, 1, 1] = np.load(f'{data_sb_path}/cov_GGGG_G_6D.npz')['arr_0']

# reshape output to heracles format
cov_dict = {}
for probe_a_ix in range(2):
    for probe_b_ix in range(2):
        for probe_c_ix in range(2):
            for probe_d_ix in range(2):
                for zi in range(Nbins):
                    for zj in range(Nbins):
                        for zk in range(Nbins):
                            for zl in range(Nbins):
                                probe_a_str = probe_dict[probe_a_ix]
                                probe_b_str = probe_dict[probe_b_ix]
                                probe_c_str = probe_dict[probe_c_ix]
                                probe_d_str = probe_dict[probe_d_ix]
                                cov_dict[(probe_a_str, probe_b_str, probe_c_str, probe_d_str, zi, zj, zk, zl)] = \
                                    cov_10d[probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix, :, :, zi, zj, zk, zl]

