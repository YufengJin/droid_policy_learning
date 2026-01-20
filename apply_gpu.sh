salloc -p a100 --gres=gpu:1 --constraint=a100_80 --time=04:00:00

source /apps/python/3.12-conda/etc/profile.d/conda.sh
conda activate xil

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
export HTTP_PROXY=http://proxy.nhr.fau.de:80
export HTTPS_PROXY=http://proxy.nhr.fau.de:80
