import numpy as np
import log_prob as NF
import run_table
import os
import sys
import multiprocessing
from nautilus import Sampler

# Load run arguments from bash
if len(sys.argv) < 5:
    raise ValueError(
        "Usage: python script.py <chain_num> <run_mode> <saras> <fresh_run> "
        "[<reion> <cxb> <uvlf> <save_blobs> <sfe_21cmFAST> <fixed> <popII_only> <mA>]"
    )
chain_num = sys.argv[1]
run_mode = sys.argv[2]
saras = sys.argv[3] == "True"
fresh_run = sys.argv[4] == "True"   # True: start a new run, False: resume an existing one

# Likelihood terms to include, all on unless switched off from the command line
reion = sys.argv[5] == "True" if len(sys.argv) > 5 else True
cxb = sys.argv[6] == "True" if len(sys.argv) > 6 else True
uvlf = sys.argv[7] == "True" if len(sys.argv) > 7 else True

# Store the 252-entry blob (CXB, T21, xHI, Q_ion, per-term log-likelihoods) with
# every sample. False keeps the .h5 file small, but the diagnostics are lost.
save_blobs = sys.argv[8] == "True" if len(sys.argv) > 8 else True

# 21cmFAST star-formation efficiency: log10Mpivot fixed to 10 (and dropped from
# the fit) with F_starII = 2 * 10**log10FstarII
sfe_21cmFAST = sys.argv[9] == "True" if len(sys.argv) > 9 else False

# Parameters held fixed instead of sampled, as 'name=value,name=value' (or 'none').
# They drop out of theta, so the run has one dimension fewer per fixed parameter.
fixed = NF.parse_fixed(sys.argv[10]) if len(sys.argv) > 10 else {}

# Drop the molecular-cooling population: the four PopIII parameters leave theta and
# beyond21 runs with model='PopII', so the run has four dimensions fewer.
popII_only = sys.argv[11] == "True" if len(sys.argv) > 11 else False

# Dark-photon mass in eV; 'none' (the default) switches the dark photon off
mA = float(sys.argv[12]) if len(sys.argv) > 12 and sys.argv[12] != "none" else None

print(f"Likelihood terms: saras={saras}, reion={reion}, cxb={cxb}, uvlf={uvlf} | "
      f"save_blobs={save_blobs} | sfe_21cmFAST={sfe_21cmFAST} | popII_only={popII_only} | "
      f"fixed={fixed} | mA={mA}", flush=True)

par_priors = NF.get_par_priors(run_mode, saras = saras, sfe_21cmFAST = sfe_21cmFAST,
                               fixed = fixed, popII_only = popII_only)
ndim = len(par_priors)

n_live = 1000           # number of live points
n_eff = 5000            # target effective sample size of the posterior
f_live = 0.01           # fraction of the evidence left in the live set at the end
EMPTY_BLOB = np.full(252, np.nan, dtype=float)


# Prior transform: maps the unit hypercube to physical parameters.
# Same priors as NF.log_prior (uniform, except Gaussian alpha_s/alpha_h and
# E_break truncated to [Emin, 30]), including the SARAS foreground parameters.
# nautilus calls this with the unit cube alone, so bind the run settings here.
def prior_transform(u):
    return NF.prior_transform(u, run_mode, saras = saras,
                              sfe_21cmFAST = sfe_21cmFAST, fixed = fixed,
                              popII_only = popII_only)


# Nested sampling needs the likelihood alone, the prior is in the transform above
def log_likelihood_wrapper(theta):
    try:
        ll, blob = NF.log_likelihood_calc(theta, run_mode, saras = saras,
                                          reion = reion, cxb = cxb, uvlf = uvlf,
                                          sfe_21cmFAST = sfe_21cmFAST, fixed = fixed,
                                          popII_only = popII_only, mA = mA)
    except Exception as err:
        print(f"Likelihood failed at {theta}: {err}", flush=True)
        ll, blob = -np.inf, EMPTY_BLOB

    if not np.isfinite(ll):
        ll, blob = -np.inf, EMPTY_BLOB

    return (ll, blob) if save_blobs else ll


if __name__ == "__main__":

    filename = f"/home/bakerem/Beyond21/tutorial/Beyond21Naut/chains/nautilus{chain_num}.h5"

    if fresh_run:
        # start from scratch: resume=False wipes anything already in the file
        print(f"Starting a fresh nautilus run: ndim={ndim}, n_live={n_live}.", flush=True)
    else:
        if not os.path.exists(filename):
            raise ValueError(
                f"Cannot resume: {filename} does not exist. Run with fresh_run=True."
            )
        print(f"Resuming the nautilus run stored in {filename}.", flush=True)

    # record what this run was configured with, before it starts
    table = run_table.record_run(run_table.table_path(filename), 'nautilus',
                                 chain_num, run_mode, saras, reion, cxb, uvlf,
                                 save_blobs, sfe_21cmFAST, fixed, popII_only)
    print(f"Settings recorded in {table}", flush=True)

    ncpus = 4
    sampler = Sampler(
        prior_transform,
        log_likelihood_wrapper,
        n_dim=ndim,
        n_live=n_live,
        pool=ncpus,
        filepath=filename,
        resume=not fresh_run,
    )
    #sampler.discard_exploration = True

    sampler.run(f_live=f_live, n_eff=n_eff, verbose=True)

    print(f"log Z = {sampler.log_z:.3f}", flush=True)
