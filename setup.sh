CONDA_ENV_NAME="madmjx"

eval "$(~/miniconda3/bin/conda shell.bash hook)"

conda update -n base -c conda-forge conda -y

conda install -n base conda-libmamba-solver -y
conda config --set solver libmamba

conda create -n ${CONDA_ENV_NAME} python=3.11 -y

conda activate ${CONDA_ENV_NAME}

conda install pip cmake -y
pip install uv

conda env config vars set CUDA_HOME=""
conda activate ${CONDA_ENV_NAME}
conda install -c "nvidia/label/cuda-12.5.1" cuda-toolkit -y
conda activate ${CONDA_ENV_NAME}

conda install -c conda-forge cudnn -y
conda install xorg-xorgproto -y
conda install -c conda-forge vulkan-tools -y

uv pip install jax[cuda12_local]==0.5.3 jaxlib mujoco-mjx mujoco mujoco_warp brax matplotlib

git clone https://github.com/shacklettbp/madrona_mjx.git
cd madrona_mjx
git submodule update --init --recursive

rm -rf build
mkdir build
cd build

SYSROOT="$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot"

cmake -S .. -B . \
  -DCMAKE_EXE_LINKER_FLAGS_INIT="--sysroot=$SYSROOT" \
  -DCMAKE_SHARED_LINKER_FLAGS_INIT="--sysroot=$SYSROOT" \
  -DCMAKE_MODULE_LINKER_FLAGS_INIT="--sysroot=$SYSROOT" \
  -DCMAKE_C_FLAGS_INIT="" \
  -DCMAKE_CXX_FLAGS_INIT=""

make -j

cd ..
uv pip install -e .

