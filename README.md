# OntoEMMA ontology matcher

[![codecov](https://codecov.io/gh/allenai/ontoemma/branch/master/graph/badge.svg)](https://codecov.io/gh/allenai/ontoemma)

This ontology matcher can be used to generate alignments between knowledgebases.

## Installation

Go to the base git directory and run: `./setup.sh`

This will create an `ontoemma` conda environment and install all required libraries.

## Train OntoEmma
To train an alignment model, use `train_ontoemma.py`. The wrapper takes the following arguments:

- -p \<model_type> (lr = logistic regression, nn = neural network)
- -m \<model_path>
- -c \<configuration_file>

Example usage:

`python train_ontoemma.py -p nn -m model_path -c configuration_file.json`

This script will then use the *train* function in `OntoEmma.py` to train the model.

### OntoEmma module
The module `OntoEmma` is used for accessing the training and alignment capabilities of OntoEmma.

#### Train mode
In training mode, the `OntoEmma` module can use the `OntoEmmaLRModel` logistic regression module or AllenNLP to train the model:

NN with AllenNLP:

- Training data is formatted according to [Data format: OntoEmma training data](https://docs.google.com/a/allenai.org/document/d/1t8cwpTRqcscFEZOQJrtTMAhjAYzlA_demc9GCY0xYaU/edit?usp=sharing)
- Train model using AllenNLP; example configuration file given in: `config/example_ontoemma_config.json`
- Save model to specified serialization directory

Configuration file:
- Training and validation data are outputs of `extract_training_data_from_umls.py`
- Enriched training/validation data have been enriched with DBPedia and MeSH terms, and Wikipedia summary sentences, see `scripts/enrich_match_data.py`

When training other models with `OntoEmmaModel`, the module performs the following:

- Load training data from file (training data is extracted from UMLS, see UMLS training data for more information)
- Iterate through KB pairs in training data, loading KBs, calculating features for each pair
- Train OntoEmmaModel using features and labels calculated from training data
- Save OntoEmmaModel to disk

#### Align mode
In alignment mode, the `OntoEmma` module performs the following:

- Load source and target ontologies specified by `source_kb_path` and `target_kb_path` (`OntoEmma` is able to handle KnowledgeBase json and pickle files, as well as KBs in OBO, OWL, TTL, and RDF formats to the best of its ability. It can also load an ontology from a web URI.)
- Initialize `CandidateSelection` module for source and target KBs

If using NN model with AllenNLP:

- Write candidates to file
- Call AllenNLP OntoEmma predictor on data
- Read predictor output from file

If using logistic regression model:

- Load LR model
- Initialize `FeatureGenerator` module
- For each candidate pair, generate features using `FeatureGenerator` and make alignment predictions using `OntoEmmaModel`

For all models following predictions:

- If `gold_file_path` is specified, evaluate alignment against gold standard
- If `output_file_path` is specified, save alignment to file

### Candidate selection module
The module `CandidateSelection` is used to select candidate matched pairs from the source and target KBs.

`CandidateSelection` is initialized with the following inputs:

- `source_kb` source KB as KnowledgeBase object
- `target_kb` target KB as KnowledgeBase object

The module builds the following token maps:

- `s_ent_to_tokens` mapping entities in source KB to word and n-gram character tokens
- `t_ent_to_tokens` mapping entities in target KB to word and n-gram character tokens
- `s_token_to_ents` mapping tokens found in source KB to entities in source KB containing those tokens
- `t_token_to_ents` mapping tokens found in target KB to entities in target KB containing those tokens
- `s_token_to_idf` mapping tokens found in source KB to their IDF in source KB
- `t_token_to_idf` mapping tokens found in target KB to their IDF in target KB

Candidates are accessed through the `select_candidates` method, which takes an input research_entity_id from the source KB and returns an ordered list of candidates from the target KB. The candidates are ordered by the sum of their token IDF scores.

The output of the `CandidateSelection` module is evaluated using the `eval` method, which takes as input:

- `gold_mappings` a list of tuples defining the gold standard mappings
- `top_ks` a list of k's (top k from candidate list to return)

The `eval` method compares the candidates generated against the gold standard mappings, returning the following:

- `cand_count` total candidate yield
- `precisions` precision value associated with each k in `top_ks`
- `recalls` recall value associated with each k in `top_ks`

### Feature generation module
The module `FeatureGenerator` is used to generate features from a candidate pair.

`FeatureGenerator` is initialized with the following inputs:

- `s_kb` source KB as KnowledgeBase object
- `t_kb` target KB as KnowledgeBase object

The module generates word and character-based n-gram tokens of entity aliases and the canonical names of entity parents and children. The module also uses a nltk stemmer and lemmatizer to produce stemmed and lemmatized version of canonical name tokens.

The `calculate_features` method is the core of this module. It generates a set of pairwise features between two input entities given by their respective entity ids from the source and target KBs. This is returned as the feature vector used in the `OntoEmmaModel`.

### UMLS training data
The module `extract_training_data_from_umls` is used to extract KB and concept mapping data from UMLS for use in ontology matching training and evaluation.

`extract_training_data_from_umls` takes as inputs:

- `UMLS_DIR` directory to UMLS META RRF files
- `OUTPUT_DIR` directory to write KnowledgeBases and mappings

UMLS data subsets are currently located at `/net/nfs.corp/s2-research/scigraph/data/ontoemma/2017AA_OntoEmma/`. `OUTPUT_DIR` defaults to `/net/nfs.corp/s2-research/scigraph/data/ontoemma/umls_output/`.

`extract_training_data_from_umls ` produces as output:

- KnowledgeBase objects written to json files in `OUTPUT_DIR/kbs/`. One json object is generated for each KnowledgeBase. Currently, KBs specified in kb_name_list are read from the UMLS subset and converted into KnowledgeBase objects.
- Pairwise KB mapping files are written to tsv files in `OUTPUT_DIR/mappings/`. Mappings are derived as pairs of raw_ids that are mapped under the same concept ID in UMLS.
- Negative mappings are sampled from UMLS as negative training data. The full training data including negatives are written to tsv files in `OUTPUT_DIR/training/`.

#### Data downloads
You will need to download a copy of the UMLS Metathesaurus by following these instructions: [https://www.nlm.nih.gov/research/umls/knowledge_sources/metathesaurus/index.html](https://www.nlm.nih.gov/research/umls/knowledge_sources/metathesaurus/index.html)

Contexts extracted from the Semantic Scholar API for some UMLS KBs are available in [this](s3://ai2-s2-ontoemma/contexts/) S3 bucket.

```bash
aws s3 cp s3://ai2-s2-ontoemma/contexts/ data/kb_contexts/ --recursive
```

Once you have downloaded both datasets, update the corresponding path variables in `emma/paths.py` to point to the appropriate directories.

#### Sampling negative data

Hard negatives are sampled using the CandidateSelection module, selecting from candidate pairs that are negative matches. Easy negatives are sampled randomly from the rest of the KB. Currently, 5 hard negatives and 5 easy negatives are sampled for each positive match.

#### Mapping file format

Mapping files are of the format described in [Data format: KB alignment file](https://docs.google.com/a/allenai.org/document/d/1VSeMrpnKlQLrJuh9ffkq7u7aWyQuIcUj4E8dUclReXM). For UMLS positive mappings, the provenance is given as \<UMLS\_header\>:\<CUI\>; for UMLS negative mappings, as \<UMLS\_header\>. Example data consisting of two positive and two negative mappings:

`CPT:90281		DRUGBANK:DB00028	1	UMLS2017AA:C0358321`
`CPT:90283		DRUGBANK:DB00028	1	UMLS2017AA:C0358321`
`CPT:83937		DRUGBANK:DB00426	0	UMLS2017AA`
`CPT:1014233		DRUGBANK:DB05907	0	UMLS2017AA`

## Run OntoEmma (align two KBs using trained model)
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


## Human annotations for evaluation
Available [here](https://docs.google.com/spreadsheets/u/1/d/e/2PACX-1vTsmTqHl6AusMLLXLIUGEYPNvCx_2_57dOEHSlaR9idrKDunlM0GwVHBlLUeQ0Tbq_15cthQ4GLHl4r/pubhtml#)


### Other

String processing utilities are found in `string_utils.py`.

Constants used by OntoEmma are found in `constants.py`. Input training data and training model parameters are specified in the training configuration files in `config/`.

This script will then use the *train* function in `OntoEmma.py` to train the model. If no GPU is specified, the program defaults to CPU.
