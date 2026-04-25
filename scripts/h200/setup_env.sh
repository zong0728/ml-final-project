#!/usr/bin/env bash
# One-time setup on DeltaAI for the ml-final-outage project.
# Run this from the LOGIN node. It does NOT submit any GPU job.
#
# Convention: project lives at /projects/bgyq/sguan/ml-final-outage/, completely
# isolated from /projects/bgyq/sguan/11685-diffusion-project/.
set -euo pipefail

PROJ_DIR="/projects/bgyq/sguan/ml-final-outage"
ENV_NAME="outage"

echo ">>> Cloning repo to $PROJ_DIR"
mkdir -p "$(dirname "$PROJ_DIR")"
if [[ ! -d "$PROJ_DIR/.git" ]]; then
  git clone https://github.com/zong0728/ml-final-project.git "$PROJ_DIR"
else
  cd "$PROJ_DIR" && git pull
fi
cd "$PROJ_DIR"

echo ">>> Loading DeltaAI Python module (provides conda + torch on aarch64)"
module load python/miniforge3_pytorch/2.10.0

echo ">>> Creating conda env '$ENV_NAME' (Python 3.11) — sandboxed under \$HOME"
# Use --prefix so we don't collide with the diffusion project's envs path.
ENV_PATH="$HOME/.conda/envs/$ENV_NAME"
if [[ ! -d "$ENV_PATH" ]]; then
  conda create -p "$ENV_PATH" python=3.11 -y
fi

echo ">>> Activating $ENV_PATH and installing deps"
source activate "$ENV_PATH"
pip install --quiet --upgrade pip
# Note: on aarch64 we install torch too (the module-provided torch is tied to
# its own python; once we activate our env we lose that). Use cu12 wheel.
pip install --quiet \
  xarray netCDF4 numpy pandas scikit-learn \
  lightgbm catboost xgboost statsmodels \
  matplotlib seaborn tqdm pyarrow PyYAML

# Torch on aarch64 — the cluster's preinstalled torch can't be imported from
# our env directly. We install our own. NVIDIA hosts aarch64 wheels via
# https://download.pytorch.org/whl/cu124.
pip install --quiet --index-url https://download.pytorch.org/whl/cu124 torch || \
  pip install --quiet torch    # fallback: PyPI wheel (may be CPU-only on aarch64)

python -c "import torch; print('torch:', torch.__version__, 'cuda(login=False expected):', torch.cuda.is_available())"

echo
echo ">>> Verifying"
python -c "
import xarray, numpy, pandas, sklearn, lightgbm, catboost, xgboost, statsmodels
print('all good:',
      'lgb', lightgbm.__version__,
      'cat', catboost.__version__,
      'xgb', xgboost.__version__,
      'xr', xarray.__version__)
"

echo
echo ">>> NEXT: copy dataset/data/*.nc from local. From your laptop run:"
echo "    rsync -avz --progress dataset/data/*.nc \\"
echo "        sguan@dtai-login.delta.ncsa.illinois.edu:$PROJ_DIR/dataset/data/"
echo
echo ">>> Done. Project root: $PROJ_DIR"
