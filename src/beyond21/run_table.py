"""
A table of run settings, one row per chain.

Each run records what it was configured with as it starts, so the table always
says what is in every chain{n}.h5 / nautilus{n}.h5 sitting next to it -- including
runs that are still going and runs that died. Re-running a chain number replaces
its row, matching the fact that a fresh run overwrites the chain file itself.

Read it with:

    import pandas as pd
    pd.read_csv('/path/to/mcmc_results/run_settings.csv')

or, without pandas,

    import run_table
    print(run_table.show('/path/to/mcmc_results/run_settings.csv'))
"""

import csv
import os
from datetime import datetime

COLUMNS = ['chain_num', 'sampler', 'run_mode', 'saras', 'reion', 'cxb', 'uvlf',
           'sfe_21cmFAST', 'popII_only', 'fixed', 'save_blobs', 'date']

FILENAME = 'run_settings.csv'


def table_path(results_file):
    """Path of the settings table living next to a chain file."""
    return os.path.join(os.path.dirname(results_file), FILENAME)


def _format_fixed(fixed):
    """
    The fixed parameters as 'name=value,name=value', the same form the run
    scripts take on the command line. Kept here rather than imported from
    log_prob, so this module stays free of the model's dependencies.
    """
    if not fixed:
        return 'none'
    if isinstance(fixed, str):
        return fixed
    return ','.join(f"{key}={val}" for key, val in fixed.items())


def _sort_key(row):
    try:
        return (row['sampler'], int(row['chain_num']))
    except (KeyError, TypeError, ValueError):
        return (row.get('sampler', ''), 0)


def record_run(path, sampler, chain_num, run_mode, saras, reion, cxb, uvlf,
               save_blobs, sfe_21cmFAST=False, fixed=None, popII_only=False):
    """
    Add a row for this run to the table at `path`, creating the file if needed.

    Any existing row with the same sampler and chain_num is replaced, so the
    table holds exactly one row per chain. Rows are kept sorted by sampler and
    chain number. Returns the path written.
    """
    row = {
        'chain_num': str(chain_num),
        'sampler': str(sampler),
        'run_mode': str(run_mode),
        'saras': str(saras),
        'reion': str(reion),
        'cxb': str(cxb),
        'uvlf': str(uvlf),
        'sfe_21cmFAST': str(sfe_21cmFAST),
        'popII_only': str(popII_only),
        'fixed': _format_fixed(fixed),
        'save_blobs': str(save_blobs),
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    rows = []
    if os.path.exists(path):
        with open(path, newline='') as fh:
            for old in csv.DictReader(fh):
                # keep only the known columns, so an older table with a different
                # set of columns is carried over rather than crashing the run
                old = {col: (old.get(col) or '') for col in COLUMNS}
                if (old['sampler'], old['chain_num']) == (row['sampler'], row['chain_num']):
                    continue        # superseded by this run
                rows.append(old)

    rows.append(row)
    rows.sort(key=_sort_key)

    # write to a temporary file first, so an interrupted write cannot destroy the
    # table that is already there
    tmp = path + '.tmp'
    with open(tmp, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)

    return path


def read(path):
    """Return the table as a list of dicts, or an empty list if it does not exist."""
    if not os.path.exists(path):
        return []
    with open(path, newline='') as fh:
        return list(csv.DictReader(fh))


def show(path):
    """Return the table as an aligned plain-text string, for printing."""
    rows = read(path)
    if not rows:
        return f"no runs recorded in {path}"

    widths = {col: max(len(col), *(len(row.get(col, '')) for row in rows))
              for col in COLUMNS}
    header = '  '.join(col.ljust(widths[col]) for col in COLUMNS)
    lines = [header, '-' * len(header)]
    for row in rows:
        lines.append('  '.join(row.get(col, '').ljust(widths[col]) for col in COLUMNS))
    return '\n'.join(lines)
