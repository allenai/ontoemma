
# OntoEMMA ontology matcher

This ontology matcher can be used to generate alignments between knowledgebases.

## Installation

Go to the base git directory and run: `./setup.sh`

This will create an `ontoemma` conda environment and install all required libraries.

## Run OntoEmma (align two KBs)
To run OntoEmma, use `run_ontoemma.py`. The wrapper implements the following arguments:

- -p \<model_type> (lr = logistic regression, nn = neural network)
- -m \<model_path>
- -s \<source_ont> 
- -t \<target_ont>
- -i \<input_alignment>
- -o \<output_file>
- -g \<cuda_device>

Example usage: 

`python run_ontoemma.py -p nn -m model_path -s source_ont.owl -t target_ont.owl -i input_alignment.tsv -o output_alignment.tsv -g 0`

This script assumes that the model has been pre-trained, and uses *align* functions in `OntoEmma.py` accordingly.

## Train OntoEmma
To train an alignment model, use `train_ontoemma.py`. The wrapper takes the following arguments:

- -p \<model_type> (lr = logistic regression, nn = neural network)
- -m \<pretrained_model>
- -c \<configuration_file>
- -g \<cuda_device>

Example usage:

`python train_ontoemma.py -p nn -m model_path -c configuration_file.json -g 0`

This script will then use the *train* function in `OntoEmma.py` to train the model. If no GPU is specified, the program defaults to CPU.