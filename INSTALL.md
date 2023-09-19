
# Install

```bash
conda create python=3.9 pip -n ralle
conda activate ralle

git clone --recursive https://github.com/yhoshi3/RaLLe
cd ralle

# install torch and transformers (required)
conda install -y pytorch=2.0.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install transformers accelerate bitsandbytes

# install ralle (required)
pip install .
# or `pip install gradio mlflow; export PYTHONPATH=$PYTHONPATH:$(pwd)`

# install KILT (required)
cd KILT; pip install .; cd ..

# install faiss (required)
conda install -y -c pytorch faiss-gpu

# install BM25 (optional)
conda install -y openjdk=11 maven
pip install pyserini

# install DiskANN (optional)
conda install -y boost cmake mkl libaio llvm-openmp gperftools ninja
conda install -y mkl-static -c intel
cd DiskANN
git checkout 0.5.0.rc3
git apply ../diskann_bs_16k.patch
export CMAKE_ARGS="-DOMP_PATH=$CONDA_PREFIX/lib -DMKL_PATH=$CONDA_PREFIX/lib -DMKL_INCLUDE_PATH=$CONDA_PREFIX/include"
pip install .
cd ..
```

**Notes:**

Due to [PydanticUserError](https://github.com/mlflow/mlflow/issues/9331) when importing MLflow (as of Aug. 2023), downgrade MLflow as work around.

```bash
pip install mlflow==2.4.1
```
