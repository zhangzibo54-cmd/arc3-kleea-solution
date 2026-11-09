# ============================== ISTA CODEBOOK EXPERTS (not carried in this notebook) ==============================
# The full cell (ISTA kernels and loader hook) lives in the ista-r25 notebooks. The hard10 notebooks
# run with ISTA off and leave it out to stay under Kaggle's 1 MB notebook limit.
ISTA = bool(V.get('ISTA', False))
assert not ISTA, 'ISTA needs the full ISTA CODEBOOK EXPERTS cell'


def _install_ista(venv, python, run_env):
    raise RuntimeError('ISTA is not carried in this notebook')


def ista_check_load(log_path):
    """Nothing to check: ISTA is off."""
