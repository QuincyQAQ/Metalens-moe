# Install PyTorch
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install dependencies
pip install -r requirements.txt

echo "Miniconda and virtual environment '$ENV_NAME' installed successfully!"

conda search mkl

conda remove mkl=2025.0.0 -y