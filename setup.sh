#!/bin/bash

CONDAENV=ontoemma

# Install conda
if ! (which conda); then
  echo "No conda installation found.  Installing..."
  if [[ $(uname) == "Darwin" ]]; then
    wget -nc --continue https://repo.continuum.io/miniconda/Miniconda-3.5.5-MacOSX-x86_64.sh
    bash Miniconda-3.5.5-MacOSX-x86_64.sh -b || true
  else
    wget -nc --continue https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b || true
  fi
  export PATH=$HOME/miniconda/bin:$HOME/miniconda3/bin:$PATH
fi

source ~/miniconda3/bin/deactivate ${CONDAENV}

conda remove -y --name ${CONDAENV} --all

conda create -n ${CONDAENV} -y python==3.6 pip pytest || true

echo "Activating Conda Environment ----->"
source ~/anaconda3/bin/activate ${CONDAENV}

pip install -r requirements.txt

if [[ $(uname) == "Darwin" ]]; then
  conda install -y pytorch torchvision -c soumith
else
  conda install -y pytorch torchvision cuda80 -c soumith
fi

python -m spacy download en

python setup.py develop

