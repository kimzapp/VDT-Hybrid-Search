# miniconda
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh
~/.bashrc
conda list
conda create -n hybrid-search python=3.10
conda activate hybrid-search

# install dependencies
pip install -r requirements.txt