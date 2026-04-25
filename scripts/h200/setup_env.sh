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

echo ">>> Creating conda env '$ENV_NAME' (Python 3.11)"
# DeltaAI uses miniforge; module-load python first.
module load python/miniforge3-pytorch || module load python/miniforge3 || true
if ! conda env list | grep -q "^$ENV_NAME "; then
  conda create -n "$ENV_NAME" python=3.11 -y
fi

echo ">>> Installing deps"
# Use pip rather than conda — faster, fewer surprises.
source activate "$ENV_NAME"
pip install --quiet --upgrade pip
pip install --quiet \
  xarray netCDF4 numpy pandas scikit-learn \
  lightgbm catboost xgboost statsmodels \
  matplotlib seaborn tqdm pyarrow

# torch is already available via the cluster module — don't double-install on aarch64.
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"

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
