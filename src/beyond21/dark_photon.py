import sys
sys.path.append("../")

from matplotlib import pylab
import numpy as np
from scipy.integrate import cumulative_trapezoid, quad
from tqdm import *
from scipy.interpolate import interp1d
import scipy.special as sp

# Dark photon perturbation theory modules
grf_path = "/home/bakerem/dark-photons-perturbations"
sys.path.append(grf_path)

from grf.grf import PerturbedProbability, FIRAS
from grf.pk_interp import PowerSpectrumGridInterpolator
from grf.units import *
from astropy.cosmology import FlatLambdaCDM

# Built once: loading the power spectrum dominates the cost of each call
# ponytail: P(k) grid is precomputed with grf's Planck18 (gen_pk.py); regenerate it for other cosmologies
pspec_lin_baryon = PowerSpectrumGridInterpolator("lin_baryon")
_probs = {}

def _get_prob(cosmo):
    """FIRAS object whose n_H, H(z), and CLASS x_e follow the Beyond21 cosmology `cosmo`. Cached per parameter set."""
    key = (cosmo.h, cosmo.Omega_m, cosmo.Omega_b, cosmo.Omega_r, cosmo.cosmo.Neff, cosmo.Y_He, cosmo.Tcmb0)
    if key not in _probs:
        astro = FlatLambdaCDM(H0=100 * cosmo.h, Om0=cosmo.Omega_m, Ob0=cosmo.Omega_b,
                              Tcmb0=cosmo.Tcmb0, Neff=cosmo.cosmo.Neff)
        prob = FIRAS(pspec_lin_baryon, cosmo=astro)
        prob.Y_p = cosmo.Y_He
        prob.initialize_class_inst()  # rerun CLASS x_e with the Beyond21 helium fraction
        # Same n_H [cm^-3] and H(z) [1/s] as evolution.py, converted to grf natural units
        prob.n_H = lambda z: cosmo.nH * (1 + z)**3 * Centimeter**-3
        prob.dz_dt = lambda z: -cosmo.hubble(1 + z) / Sec * (1 + z)
        _probs[key] = prob
    return _probs[key]

def get_Pconv(mA, epsilon, omega_ary, cosmo):
    omega_ary = np.atleast_1d(omega_ary) * eV
    m_Ap = mA * eV

    # Density fluctuation cutoff for lognormal model validity
    one_plus_delta_bound = 1e2  # δ < 100 (conservative bound)

    # Redshift array (exclude EoR to avoid 21cmFAST complications)
    rs_array = np.geomspace(0.001, 17, 1000)
    rs_array = rs_array[(rs_array < 5) | (rs_array > 35)]  # Skip 5 < z < 35

    # dP/dz on the (omega, z) grid in one vectorized call, then integrate over z
    dPdrs_array = _get_prob(cosmo)._dP_dz(
        z_ary=rs_array,
        m_Ap=m_Ap,
        k_min=1e-3, k_max=1e3,  # Integration bounds
        omega=omega_ary,
        pdf='lognormal',
        one_plus_delta_bound=one_plus_delta_bound
    )[0]
    return np.trapz(dPdrs_array, rs_array, axis=1) * epsilon**2
