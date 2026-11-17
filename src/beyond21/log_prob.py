import copy

import numpy as np
from numpy.linalg import slogdet, solve
from scipy.interpolate import interp1d
from scipy.stats import norm, truncnorm

import beyond21.global_wrapper as ob
import beyond21.constants as consts

# - - - - - - - - - - - - - - - -
# Set priors and prior transform
# - - - - - - - - - - - - - - - -

# In the 21cmFAST star-formation parametrization the pivot mass is fixed to
# 1e10 Msun and the normalization carries an extra factor of 2, i.e.
# F_starII = 2 * 10**log10FstarII with log10Mpivot = 10. log10Mpivot is then not
# a free parameter and drops out of the fit.
MPIVOT_21CMFAST = 10.0       # log10(Mpivot / Msun)
FSTARII_FACTOR_21CMFAST = 2.0


SARAS_PARS = ['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6']

# The PopIII (molecular-cooling) parameters. A popII_only run drops them from the
# fit: beyond21 is told model='PopII', so nothing reads them.
POPIII_PARS = ['log10FstarIII', 'alpha_starIII', 'log10McutIII', 'log10FescIII',
               'alpha_escIII']


def _all_par_priors(xray_mode, saras = False, popII_only = False):
    """Every parameter the run mode defines, before anything is held fixed."""
    par_priors = {
       't_star' : [0,1],
       'log10FstarIII' : [-3,0],
       #'alpha_starIII': [-0.5,2.5],
       'alpha_starIII': [-2,2],
       'log10McutIII' : [6,10],
       'log10FstarII' : [-3,2],
       #'log10FstarII' : [-3,0],
       'log10Mpivot' : [10,15],
       #'log10Mpivot' : [8,11],
       'alpha_1' : [0,1.5],
       'alpha_2' : [-1.5,0],
        #    'alpha_1' : [0,2],
        #    'alpha_2' : [0,5],
       'log10McutII' : [6,10],
       'log10FescIII' : [-3,0],
       'alpha_escIII' : [-2,3],
       'log10FescII' : [-3,0],
       'alpha_escII' : [-2,1],
       'log10Fx' : [-2.5,3.5],
       'Emin': [0.1,1.5],
       'log_epsilon': [-9,-5],
    }
    if xray_mode in ['single_xPL','full']:
        par_priors['alpha_s'] = [0.9,0.44] # Gaussian prior with mean [mean,sigma]
    
    if xray_mode == 'full':
        par_priors['alpha_h'] = [1.62,0.22]# Gaussian prior with mean [mean,sigma]
        par_priors['E_break'] = [4.38,1.3]  # Truncated Gaussian prior with mean [mean,sigma], truncated to [Emin, 30]

    if saras:
        for a in SARAS_PARS:
            par_priors[a] = [-5,5]

    if popII_only:
        for key in POPIII_PARS:
            par_priors.pop(key, None)

    return par_priors


# Every parameter that can be held fixed: the union over all run modes
ALL_PARS = list(_all_par_priors('full', saras = True).keys())


# The minimum-mass parameters, and the one non-numeric value they accept. Fixing
# one of them to 'Matom' hands beyond21 the string rather than a number, so the
# cutoff follows its redshift-dependent atomic-cooling mass instead of sitting at
# a constant floor. No value of log10Mcut* can reproduce that, hence the sentinel.
MCUT_PARS = ['log10McutII', 'log10McutIII']
MATOM = 'Matom'


def _fixed_value(key, val):
    """A fixed value: a float, or 'Matom' for one of the MCUT_PARS."""
    if isinstance(val, str) and val.strip().lower() == MATOM.lower():
        if key not in MCUT_PARS:
            raise ValueError(f"{MATOM!r} can only be fixed for {MCUT_PARS}, not {key!r}")
        return MATOM
    return float(val)


def resolve_fixed(sfe_21cmFAST = False, fixed = None):
    """
    The parameters held fixed in this run, as a name -> value dict.

    `fixed` is whatever the caller asked to hold fixed; the 21cmFAST SFE fixes
    log10Mpivot on top of it. Any parameter can be fixed, including the ones a
    run mode pins down on its own (constant_xPL pins alpha_s to 1, say), in
    which case the fixed value wins over the mode's.

    Values are floats, except that log10McutII / log10McutIII also take 'Matom'.
    """
    resolved = {key: _fixed_value(key, val) for key, val in dict(fixed or {}).items()}

    if sfe_21cmFAST:
        pivot = resolved.setdefault('log10Mpivot', MPIVOT_21CMFAST)
        if pivot != MPIVOT_21CMFAST:
            raise ValueError(
                f"sfe_21cmFAST fixes log10Mpivot to {MPIVOT_21CMFAST}, but `fixed` "
                f"asks for {pivot}"
            )

    unknown = [key for key in resolved if key not in ALL_PARS]
    if unknown:
        raise ValueError(f"cannot fix {unknown}: no such parameter. "
                         f"The parameters are {ALL_PARS}")

    return resolved


def format_fixed(fixed):
    """
    Write a fixed-parameter dict as a single command-line token, e.g.
    {'log10Mpivot': 10, 'Emin': 0.5} -> 'log10Mpivot=10,Emin=0.5'.

    An empty dict becomes 'none', so the argument is always there to be parsed.
    'Matom' passes through as itself, e.g. 'log10McutII=Matom'.
    """
    if not fixed:
        return 'none'
    return ','.join(f"{key}={val}" for key, val in fixed.items())


def parse_fixed(text):
    """Read back what format_fixed wrote."""
    if not text or text == 'none':
        return {}

    fixed = {}
    for item in text.split(','):
        key, sep, val = item.partition('=')
        if not sep:
            raise ValueError(f"cannot parse fixed parameter {item!r}, expected name=value")
        key = key.strip()
        fixed[key] = _fixed_value(key, val.strip())
    return fixed


def get_par_priors(xray_mode, saras = False, sfe_21cmFAST = False, fixed = None,
                   popII_only = False):
    """The sampled parameters: what the run mode defines, minus what is held fixed."""
    par_priors = _all_par_priors(xray_mode, saras, popII_only)

    for key in resolve_fixed(sfe_21cmFAST, fixed):
        # a fixed parameter this mode does not sample anyway (alpha_s under
        # constant_xPL, say) simply has nothing to remove here
        par_priors.pop(key, None)

    return par_priors


# Parameter range for plots
def get_par_range(xray_mode, saras = False, sfe_21cmFAST = False, fixed = None,
                  popII_only = False):
    par_priors = get_par_priors(xray_mode, saras, sfe_21cmFAST, fixed, popII_only)
    fixed_pars = resolve_fixed(sfe_21cmFAST, fixed)
    par_range = copy.deepcopy(par_priors)

    if 'alpha_s' in par_range:
        par_range['alpha_s'] = [0.9 - 3*0.44, 0.9 + 3*0.44]
    if 'E_break' in par_range:
        # Emin is the lower truncation of E_break, whether it is sampled or fixed
        emin_low = par_range['Emin'][0] if 'Emin' in par_range else fixed_pars['Emin']
        par_range['E_break'] = [np.minimum(emin_low, 4.38-3*1.3),4.38+3*1.3] # truncate lower limit
    if 'alpha_h' in par_range:
        par_range['alpha_h'] = [1.62 - 3*0.22, 1.62 + 3*0.22]
    return par_range

# prior transform for nested sampling
def prior_transform(u, xray_mode, saras = False, sfe_21cmFAST = False, fixed = None,
                    popII_only = False):
    """
    Maps unit cube u ~ [0, 1]^d to physical parameter space.
    All priors are uniform except:
      - alpha_h: Gaussian
      - alpha_s: Gaussian
      - E_break: truncated Gaussian with bounds [Emin, 30]
    """
    theta = []
    par_priors = get_par_priors(xray_mode, saras, sfe_21cmFAST, fixed, popII_only)
    fixed_pars = resolve_fixed(sfe_21cmFAST, fixed)
    keys = list(par_priors.keys())
    idx_Emin = keys.index('Emin') if 'Emin' in keys else None
                                    # 'Emin' precedes 'E_break', so theta[idx_Emin]
                                    # is already filled when E_break is reached

    for i, key in enumerate(keys):
        a, b = par_priors[key]

        if key in ['alpha_h','alpha_s']:
            mu, sigma = a, b
            theta.append(norm.ppf(u[i], loc=mu, scale=sigma))
        elif key == 'E_break':
            mu, sigma = a, b
            Emin = theta[idx_Emin] if idx_Emin is not None else fixed_pars['Emin']
            low, high = Emin, 30
            low_n = (low - mu) / sigma #stdevs from mu to low limit
            high_n = (high - mu) / sigma #stdevs from mu to upper limit
            theta.append(truncnorm.ppf(u[i], low_n, high_n, loc=mu, scale=sigma))
        else:
            low, high = a, b
            theta.append(low + u[i] * (high - low))
    return np.array(theta)


# log prior for MCMC
def log_prior(theta, xray_mode, saras = False, sfe_21cmFAST = False, fixed = None,
              popII_only = False):
    """
    Log-prior matching the prior_transform.

    Priors:
      - uniform for all parameters except:
      - alpha_h: Gaussian
      - alpha_s: Gaussian
      - E_break: truncated Gaussian with bounds [Emin, 30]
    """
    logp = 0.0
    par_priors = get_par_priors(xray_mode, saras, sfe_21cmFAST, fixed, popII_only)
    fixed_pars = resolve_fixed(sfe_21cmFAST, fixed)
    params = dict(zip(par_priors.keys(), theta))

    for key, val in params.items():
        a, b = par_priors[key]

        if key in ['alpha_h','alpha_s']:
            mu, sigma = a, b
            logp += norm.logpdf(val, loc=mu, scale=sigma)
        elif key == 'E_break':
            mu, sigma = a, b
            low = params['Emin'] if 'Emin' in params else fixed_pars['Emin']
            high = 30.0

            if not (low <= val <= high):
                return -np.inf
            low_n = (low - mu) / sigma
            high_n = (high - mu) / sigma
            logp += truncnorm.logpdf(val, low_n, high_n, loc=mu, scale=sigma)
        else:
            low, high = a, b
            if not (low <= val <= high):
                return -np.inf
            logp += -np.log(high - low)

    return logp


# - - - - - - - - - - - - - - 
# Global object initialization
# - - - - - - - - - - - - - -

def initialize_global_object(theta, xray_mode, sfe_21cmFAST = False, fixed = None,
                             popII_only = False, mA = None):
    if xray_mode not in ['full', 'single_xPL', 'constant_xPL']:
        raise ValueError(f"Unknown xray_mode: {xray_mode!r}")

    # theta follows the order of the sampled parameters, which is exactly the
    # order of get_par_priors (without the SARAS foregrounds, stripped by the caller)
    keys = list(get_par_priors(xray_mode, saras = False,
                               sfe_21cmFAST = sfe_21cmFAST, fixed = fixed,
                               popII_only = popII_only).keys())
    if len(theta) != len(keys):
        raise ValueError(
            f"xray_mode {xray_mode!r} (sfe_21cmFAST={sfe_21cmFAST}, fixed={fixed}, "
            f"popII_only={popII_only}) expects {len(keys)} parameters, got {len(theta)}"
        )
    # the fixed values sit next to the sampled ones, and win over whatever the
    # run mode would have pinned them to
    par = dict(zip(keys, theta))
    par.update(resolve_fixed(sfe_21cmFAST, fixed))

    def pop3(key, unused):
        """A PopIII parameter.

        In a two-population run it must be there, and a missing one is a bug worth
        raising on. Under popII_only it is not sampled: beyond21 gets model='PopII'
        and never reads it, but the parameter dicts below keep the same shape, so a
        placeholder goes in rather than a special case at every use.
        """
        return par.get(key, unused) if popII_only else par[key]

    t_star = par['t_star']
    log10FstarIII = pop3('log10FstarIII', -2.5)
    alpha_starIII = pop3('alpha_starIII', -0.5)
    log10FstarII = par['log10FstarII']
    alpha_star1II = par['alpha_1']
    alpha_star2II = par['alpha_2']
    log10FescIII = pop3('log10FescIII', -2.0)
    alpha_escIII = pop3('alpha_escIII', 0.0)
    log10FescII = par['log10FescII']
    alpha_escII = par['alpha_escII']
    log10Fx = par['log10Fx']
    Emin = par['Emin']

    # not sampled in every mode: constant_xPL pins the soft X-ray slope, and
    # single_xPL ties the hard slope to the soft one
    alpha_s = par.get('alpha_s', 1)
    alpha_h = par.get('alpha_h', alpha_s)
    E_break = par.get('E_break', 2)

    log10MpivotII = par['log10Mpivot']   # sampled, or fixed (10 under sfe_21cmFAST)


    def mcut(value):
        return MATOM if value == MATOM else 10**value

    McutII = mcut(par['log10McutII'])
    McutIII = mcut(pop3('log10McutIII', 5.0))
    FstarII = (FSTARII_FACTOR_21CMFAST if sfe_21cmFAST else 1.0) * 10**log10FstarII
    FstarIII = 10**log10FstarIII
    MpivotII = 10**log10MpivotII
    FescII = 10**log10FescII
    FescIII = 10**log10FescIII
    Fx = 10**log10Fx

    cosmo_params = {
        'Om0' : 0.3111,
        'Ob0' : 0.0490,
        'H0' : 67.66,
        'sigma8' : 0.8102,
        'ns' : 0.9665,
        'Neff' : 3.046,
        'Y_He' : 0.24, 
        'hmf_kwargs' : {"mdef": "fof", "model": "sheth99"}
        }

    star_formation_params = {
                    # 'PopII' drops the molecular-cooling population entirely:
                    # beyond21's has_popIII goes False and the PopIII entries below
                    # are never read
                    'model' : 'PopII' if popII_only else 'PopII+PopIII',
                    'M_cutII' : McutII,
                    'M_cutIII' : McutIII,
                    'eps_t' : t_star,
                    'F_starII' : FstarII,
                    'F_starIII' : FstarIII,
                    'Mpivot' : MpivotII,
                    'alphaII' : alpha_star1II,
                    'betaII' : alpha_star2II,
                    'alphaIII': alpha_starIII,
                    'A_LW' : 2,
                    'B_LW' : 0.6,
                    # 'A_LW' : 22.87,
                    # 'B_LW' : 0.47,
                    # 'A_vrel' : 1,
                    # 'B_vrel' : 1.8,
                    }

    
    Xray_params = {
        'LSFRII' : 10**(40.5)* Fx,
        'LSFRIII' : 10**(40.5)* Fx,
        'E_min' : Emin,
        'E_max' : 30,
        'alpha_s' : alpha_s,
        'alpha_h' : alpha_h,
        'E_break' : E_break,
        }
    
    reion_params = {
        'F_escII' : FescII,
        'F_escIII' : FescIII,
        'alpha_escII' : alpha_escII,
        'alpha_escIII' : alpha_escIII,
        'N_ionII' : 5000,          # Number of ionizing photons per baryon in Pop-II stars
        'N_ionIII': 50000,
        #'ion_prescription' : 'reprod_21cmFAST',  # Use finite difference of collapsed stellar mass for ionizing photon budget
        # 'ion_prescription' : 'reprod_21cmFAST',  # Use finite difference of collapsed stellar mass for ionizing photon budget
        # 'photoheat_prescription' : 'sobacchi_mesinger',  # Use mass-dependent photoheating suppression
        #'photoheat_prescription' : 'popIII_quenched',  
        #'photoheat_prescription' : 'fixed_threshold',  

    }


    T21_obj = ob.GlobalWrapper(star_formation_params = star_formation_params, xray_params = Xray_params, reion_params = reion_params, cosmo_params = cosmo_params, Lya_Heat = False, CMB_Heat = False, photoheat = True,
                             dp_kwargs = None if mA is None else {'mA': mA, 'log_epsilon': par['log_epsilon']})
    T21_obj.EvolveIGM(z_min = 5, z_max = 2000, Nz = 250, ivp_kwargs = {"rtol":1e-4,"max_step":0.01})
    
    return T21_obj

# - - - - - - - - - - - - 
# Compute UVLF likelihood
# - - - - - - - - - - - - 
def compute_uvlf_likelihood(T21_obj, redshift, mags, mean, up, low):
    if T21_obj.star_formation_params['model'] == 'PopII':
        modeled_UVLF_combined = T21_obj.UVLF(redshift, Muv = mags, sigma_MUV = 0.01, kUVII=1.15e-28, kUVIII=1.15e-28)
    elif T21_obj.star_formation_params['model'] == 'PopII+PopIII':
        modeled_UVLFII, modeled_UVLFIII = T21_obj.UVLF(redshift, Muv = mags, sigma_MUV = 0.01, kUVII=1.15e-28, kUVIII=1.15e-28)
        modeled_UVLF_combined = modeled_UVLFII + modeled_UVLFIII

    diff = modeled_UVLF_combined - mean
    var = np.where(diff >= 0, up**2, low**2)
    return np.sum(-0.5 * diff**2 / var+ 0.5 * np.log(2 / np.pi)- np.log(up + low))

    # diff = modeled_UVLF_combined - mean
    # var = np.where(modeled_UVLF_combined > mean, up**2, low**2)
    # return -0.5*np.sum(diff**2 / var + np.log(2*np.pi*var))



# - - - - - - - - - - - - - - - -
# Compute reionization likelihood
# - - - - - - - - - - - - - - - -
def reionization_likelihood(T21_obj):
    # log-likelihood contribution from reionization constraints (CMB optical depth and quasar spectra)
    log_likelihood = 0

     # constraints from CMB optical depth
    tau_mean = 0.054
    tau_stdev = 0.007

    
    tau = T21_obj.tau          # CMB optical depth to reionization
    log_likelihood += -0.5*(tau-tau_mean)**2/tau_stdev**2  

    # constraints from quasar spectra
    xHI_out = T21_obj.xHI
    rs_out = T21_obj.rs
    xHI_interp = interp1d(rs_out-1,xHI_out)
    
    xHI5p9 = xHI_interp(5.9)
    xHI_5p9_mean = 0.06
    xHI_5p9_stdev = 0.05
    if xHI5p9>xHI_5p9_mean:
       log_likelihood += -0.5*(xHI5p9-xHI_5p9_mean)**2/xHI_5p9_stdev**2  #else adds 0 to likelihood

    # xHI upper limits, z : [mean, stdev]
    # Qspectra = {
    #     '5.481' : [0.030, 0.048],
    #     '5.654' : [0.095, 0.037],
    #     '5.831' : [0.191, 0.056],
    #     '6.043' : [0.199, 0.087]
    # }

    # xHI_model = {
    #     '5.481' : xHI_interp(5.481),
    #     '5.654' : xHI_interp(5.654),
    #     '5.831' : xHI_interp(5.831),
    #     '6.043' : xHI_interp(6.043)
    # }

    # for key in Qspectra:
    #     mean, sigma = Qspectra[key]
    #     if xHI_model[key] > mean:
    #         log_likelihood += -0.5*(xHI_model[key] - mean)**2 / sigma**2
    
    return log_likelihood

# - - - - - - - - - - - 
# Compute CXB likelihood
# - - - - - - - - - - - 
def cxb_likelihood(global_obj):
    #log-likelihood contribution from X-ray constraints
    z_unresolved = 6

    CXB_model = { # CXB contribution in different energy bands
        'soft' : global_obj.CXB(z_unresolved,1,2), 
        'med' : global_obj.CXB(z_unresolved,2,8), 
        'hard' : global_obj.CXB(z_unresolved,8,24)
    }

    rescale = 1
    CXB_Chandra = { # Chandra measurements [mean, stdev] 
        'soft' : np.array([3.41, 0.46]) * 1e-9 / rescale, 
        'med' : np.array([1.12, 0.56]) * 1e-8 / rescale, 
        'hard' : np.array([6.01, 0.14]) * 1e-8 / rescale  
    }

    log_likelihood = 0
    for key in CXB_Chandra:
        mean, sigma = CXB_Chandra[key]
        if CXB_model[key] > mean:
            log_likelihood += -0.5*(CXB_model[key] - mean)**2 / sigma**2
    
    return log_likelihood, np.array(list(CXB_model.values()))

# - - - - - - - - - - - -
# Compute 21cm likelihood 
# - - - - - - - - - - - -

def LogPoly_SARAS3(a_params, nu_data):
    '''
        Input
            - a_params: Coefficients of log polynomial
            - nu_data [MHz]: Measured frequencies
        return
            - FG model
    '''
    log10_nu = np.log10(nu_data)
    log10_nu_max = np.max(log10_nu)
    log10_nu_min = np.min(log10_nu)
    fact = (log10_nu_max - log10_nu_min) / 2
    Log10_nu_rescaled = (log10_nu - log10_nu_min) / fact - 1
    
    Log10_FG = 0
    for i,a in enumerate(a_params):
        Log10_FG += a * Log10_nu_rescaled**i 
        
    return Log10_FG

def LoadDataSARAS(Tsky = False, Residuals = False, NoiseRMS = False):
    data = np.loadtxt("/home/bakerem/Beyond21/src/beyond21/DataSARAS3.txt", usecols=(0, 1, 2, 3, 4))
    freq = data[:, 0]

    if Tsky:
        return freq, data[:, 1]
    elif NoiseRMS:
        return freq, data[:, 2]

def MHz_to_z(frequency_MHz):
    #Instert frequency in MHz and get redshift
    frequency_Hz=1e6*frequency_MHz
    lambda21=21.106114054160
    z=consts.c/frequency_Hz/lambda21-1
    return(z)

nu_data, Tsky_data = LoadDataSARAS(Tsky = True)
nu_data, NoiseRMS = LoadDataSARAS(NoiseRMS = True)
z_data_SARAS3 = MHz_to_z(nu_data)

def compute_21cm_likelihood(a_params, nu_data, Tsky_data, NoiseRMS, Tsignal, PolyOrder=6):
    Model = Tsignal + 10**LogPoly_SARAS3(a_params, nu_data)
    chi2 = np.sum((Tsky_data - Model)**2 / NoiseRMS**2)
    logconst = -0.5 * np.sum(np.log(2 * np.pi * NoiseRMS**2)) #Normalization 1/sqrt(2*pi*sigma)
    return -0.5 * chi2 + logconst

# - - - - - - - - - - - -
# Compute full likelihood
# - - - - - - - - - - - -
def log_likelihood_calc(theta, xray_mode, saras = False, reion = True, cxb = True, uvlf = True,
                        sfe_21cmFAST = False, fixed = None, popII_only = False, mA = None):
    """
    Total log-likelihood.

    mA is the dark-photon mass in eV. None switches the dark photon off (and
    log_epsilon, though sampled, then has no effect).

    Data switches (each term is always computed and stored in the blob, the
    switch only controls whether it is added to the total):
      - saras: SARAS3 21cm term (also adds the foreground parameters to theta)
      - reion: reionization term (CMB tau + quasar xHI)
      - cxb:   unresolved CXB term
      - uvlf:  UV luminosity function term

    sfe_21cmFAST switches the star-formation efficiency to the 21cmFAST
    parametrization: log10Mpivot fixed to 10 (and dropped from theta) with
    F_starII = 2 * 10**log10FstarII.

    popII_only drops the molecular-cooling population: the four PopIII parameters
    leave theta and beyond21 runs with model='PopII'. The UVLF term then compares
    the data against the PopII luminosity function alone, since there is no second
    population to add to it.

    fixed is {parameter: value} for parameters held fixed rather than sampled;
    they are dropped from theta and never enter the prior.
    """
    fixed_pars = resolve_fixed(sfe_21cmFAST, fixed)

    if saras:
        # the sampled foreground coefficients sit at the end of theta, in the
        # order of SARAS_PARS, with the fixed ones missing
        a_sampled = [a for a in SARAS_PARS if a not in fixed_pars]
        theta_model = theta[:len(theta) - len(a_sampled)]
        a_values = dict(zip(a_sampled, theta[len(theta) - len(a_sampled):]))
        a_values.update(fixed_pars)
        a_params = np.array([a_values[a] for a in SARAS_PARS])
        T21_obj = initialize_global_object(theta_model, xray_mode,
                                           sfe_21cmFAST = sfe_21cmFAST, fixed = fixed,
                                           popII_only = popII_only, mA = mA)
    else:
        T21_obj = initialize_global_object(theta, xray_mode,
                                           sfe_21cmFAST = sfe_21cmFAST, fixed = fixed,
                                           popII_only = popII_only, mA = mA)

    log_likelihood = 0
    log_likelihood_reion = reionization_likelihood(T21_obj)
    if reion:
        log_likelihood += log_likelihood_reion
    log_likelihood_CXB, CXB = cxb_likelihood(T21_obj)
    if cxb:
        log_likelihood += log_likelihood_CXB
    #log_likelihood += cxb_likelihood(T21_obj)

    if saras:
        T21_out = T21_obj.T21
        rs_out = T21_obj.rs
        T21_of_z_interp = interp1d(rs_out-1,T21_out)
        T21_at_SARAS3_freq = T21_of_z_interp(z_data_SARAS3)
        log_likelihood += compute_21cm_likelihood(a_params, nu_data, Tsky_data, NoiseRMS, T21_at_SARAS3_freq, PolyOrder=6)
        
    # UVLFs
    # - - - 
    uvlf_likelihood = 0

    for z, data in Bouwens21_UVLF.items(): # Log likelihood Bouwens21
        uvlf_likelihood += compute_uvlf_likelihood(T21_obj, z, data['mag'], data['mean'], data['up'], data['low'])
    
    for z, data in Bouwens16_UVLF_short.items(): # Log likelihood Bouwens16 addition
        uvlf_likelihood += compute_uvlf_likelihood(T21_obj, z, data['mag'], data['mean'], data['up'], data['low'])
    
    # for z, data in Donnan24_UVLF.items(): # Log likelihood new-JWST
    #     uvlf_likelihood += compute_uvlf_likelihood(T21_obj, z, data['mag'], data['mean'], data['up'], data['low'])

    if uvlf:
        log_likelihood += uvlf_likelihood

    max_idx = 82 # corresponds to z=40
    blob = np.concatenate([CXB, T21_obj.T21[-max_idx:], T21_obj.xHI[-max_idx:], T21_obj.Q_ion_interp(T21_obj.rs[-max_idx:]-1), np.array([log_likelihood_reion, log_likelihood_CXB, uvlf_likelihood])]) # 3 first entries CXB then 82 entries T21
    #blob = (T21_obj.T21[T21_obj.rs<40])
    return log_likelihood, blob

def log_prob(theta, xray_mode, saras = False, reion = True, cxb = True, uvlf = True,
             sfe_21cmFAST = False, fixed = None, popII_only = False):
    lp = log_prior(theta, xray_mode, saras, sfe_21cmFAST, fixed, popII_only)
    if not np.isfinite(lp):
        EMPTY_BLOB = np.full(252, np.nan, dtype=float)
        return -np.inf, EMPTY_BLOB

    ll, blob = log_likelihood_calc(theta, xray_mode, saras, reion = reion, cxb = cxb, uvlf = uvlf,
                                   sfe_21cmFAST = sfe_21cmFAST, fixed = fixed,
                                   popII_only = popII_only)

    if not np.isfinite(ll):
        EMPTY_BLOB = np.full(252, np.nan, dtype=float)
        return -np.inf, EMPTY_BLOB
    
    return lp + ll, blob


# - - - - - - - -
# UVLF data dicts
# - - - - - - - -
# Donnan24_UVLF = {
#     9: {
#         'mag': np.array([-19.75, -19.25, -18.55, -18.05, -17.55]),
#         'mean': 1e-6 * np.array([144, 235, 486, 1110, 1776]),
#         'up':   1e-6 * np.array([30, 60, 157, 310, 578]),
#         'low':  1e-6 * np.array([28, 49, 139, 310, 510])
#     },
#     10: {
#         'mag': np.array([-19.75, -19.25, -18.55, -18.05, -17.5]),
#         'mean': 1e-6 * np.array([92, 177, 321, 686, 1278]),
#         'up':   1e-6 * np.array([25, 53, 127, 245, 486]),
#         'low':  1e-6 * np.array([20, 45, 111, 223, 432])
#     },
#     11: {
#         'mag': np.array([-19.75, -19.25, -18.75, -18.25]),
#         'mean': 1e-6 * np.array([100, 144, 234, 641]),
#         'up':   1e-6 * np.array([37, 81, 118, 361]),
#         'low':  1e-6 * np.array([30, 63, 96, 281])
#     },
#     12.5: {
#         'mag': np.array([-19.75, -19.25, -18.75, -18.25]),
#         'mean': 1e-6 * np.array([34, 43, 80, 217]),
#         'up':   1e-6 * np.array([23, 35, 51, 153]),
#         'low':  1e-6 * np.array([15, 22, 36, 104])
#     }
# }


Bouwens21_UVLF = {
    6: {
        'mag': np.array([-19.52, -18.77, -17.77, -16.77]),
        'mean': np.array([0.001147, 0.002820, 0.008360, 0.017100]),
        'up':   np.array([0.000157, 0.000440, 0.001660, 0.005260]),
        'low':  np.array([0.000157, 0.000440, 0.001660, 0.005260])
    },
    7: {
        'mag': np.array([-19.69, -19.19, -18.69, -17.94, -16.94]),
        'mean': np.array([0.000589, 0.001172, 0.001433, 0.005760, 0.008320]),
        'up':   np.array([0.000126, 0.000336, 0.000419, 0.001440, 0.002900]),
        'low':  np.array([0.000126, 0.000336, 0.000419, 0.001440, 0.002900])
    },
    8: {
        'mag': np.array([-19.35, -18.6, -17.6]),
        'mean': np.array([0.000657, 0.001100, 0.003020]),
        'up':   np.array([0.000233, 0.000340, 0.001140]),
        'low':  np.array([0.000233, 0.000340, 0.001140])
    },
    9: {
        'mag': np.array([-19.12, -17.92]),
        'mean': np.array([0.000187, 0.000923]),
        'up':   np.array([0.000150, 0.000501]),
        'low':  np.array([0.000150, 0.000501])
    },
    10: {
        'mag': np.array([-19.25, -18.25, -17.25]),
        'mean': np.array([0.000034, 0.000190, 0.000630]),
        'up':   np.array([0.000022, 0.000120, 0.000520]),
        'low':  np.array([0.000022, 0.000120, 0.000520])
    }
}

Bouwens16_UVLF_short = {
    6: {
        'mag': np.array([-15.75, -15.25, -14.75, -14.25, -13.75, -13.25, -12.75]),
        'mean': np.array([0.0415, 0.0599, 0.0817, 0.1052, 0.1275, 0.1464, 0.1584]),
        'up':   np.array([0.0354, 0.0757, 0.1902, 0.5414, 1.6479, 5.4369, 19.8047]),
        'low':  np.array([0.0069, 0.0106, 0.0210, 0.0434, 0.0747, 0.1077, 0.1343])  # Converted 69e-4 to 0.0069 for consistency
    }}


