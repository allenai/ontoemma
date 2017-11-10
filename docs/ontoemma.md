
## OntoEmma module
The module `OntoEmma` is used for accessing the training and alignment capabilities of OntoEmma.

### Train mode
In training mode, the `OntoEmma` module can use the `OntoEmmaLRModel` logistic regression module or AllenNLP to train the model:

NN with AllenNLP:

- Training data is formatted according to [Data format: OntoEmma training data](https://docs.google.com/a/allenai.org/document/d/1t8cwpTRqcscFEZOQJrtTMAhjAYzlA_demc9GCY0xYaU/edit?usp=sharing)
- Train model using AllenNLP; example configuration file given in `/config` directory
- Save model to specified serialization directory
- GPU flag gives user the option of specifying a CUDA device for training

When training other models with `OntoEmmaModel`, the module performs the following:

- Load training data from file 
- Calculate features for each entity in the training data
- Train OntoEmmaModel using features and labels
- Save OntoEmmaModel to disk

### Align mode
In alignment mode, the `OntoEmma` module performs the following:

- Load source and target ontologies specified by `source_kb_path` and `target_kb_path` (`OntoEmma` is able to handle KnowledgeBase json and pickle files, as well as KBs in OBO, OWL, TTL, and RDF formats to the best of its ability. It can also load an ontology from a web URI.)
- Initialize `CandidateSelection` module for source and target KBs

If using NN model with AllenNLP:

- Call AllenNLP OntoEmma predictor on data in batches
- Return predictor outputs as output alignment
- GPU flag allows user to specify a CUDA device for prediction

If using logistic regression model:

- Load LR model
- Initialize `FeatureGenerator` module
- For each candidate pair, generate features using `FeatureGenerator` and make alignment predictions using `OntoEmmaModel`

For all models following predictions:

- If `gold_file_path` is specified, evaluate alignment against gold standard
- If `output_file_path` is specified, save alignment to file

## Candidate selection module
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

## Feature generation module
The module `FeatureGenerator` is used to generate features from a candidate pair.

The module generates word and character-based n-gram tokens of entity aliases and the canonical names of entity parents and children. The module also uses a nltk stemmer and lemmatizer to produce stemmed and lemmatized version of canonical name tokens.

The `calculate_features` method is the core of this module. It generates a set of pairwise features between two input entities identified by their research entity ids. Features are returned as a feature vector for use in the `OntoEmmaModel`.

## UMLS training data
The module `extract_training_data_from_umls` is used to extract KB and concept mapping data from UMLS for use in ontology matching training and evaluation.

`extract_training_data_from_umls` takes as inputs:

- `UMLS_DIR` directory to UMLS META RRF files
- `OUTPUT_DIR` directory to write KnowledgeBases and mappings

UMLS data subsets are currently located at `/net/nfs.corp/s2-research/scigraph/data/ontoemma/2017AA_OntoEmma/`. `OUTPUT_DIR` defaults to `/net/nfs.corp/s2-research/scigraph/data/ontoemma/umls_output/`.

`extract_training_data_from_umls ` produces as output:

- KnowledgeBase objects written to json files in `OUTPUT_DIR/kbs/`. One json object is generated for each KnowledgeBase. Currently, KBs specified in kb_name_list are read from the UMLS subset and converted into KnowledgeBase objects.
- Pairwise KB mapping files are written to tsv files in `OUTPUT_DIR/mappings/`. Mappings are derived as pairs of raw_ids that are mapped under the same concept ID in UMLS.
- Negative mappings are sampled from UMLS as negative training data. The full training data including negatives are written to tsv files in `OUTPUT_DIR/training/`.

### Sampling negative data

Hard negatives are sampled using the CandidateSelection module, selecting from candidate pairs that are negative matches. Easy negatives are sampled randomly from the rest of the KB. The number of easy and hard negatives sampled per positive is given in `constants.py`.

### Mapping file format

Mapping files are of the format described in [Data format: KB alignment file](https://docs.google.com/a/allenai.org/document/d/1VSeMrpnKlQLrJuh9ffkq7u7aWyQuIcUj4E8dUclReXM). For UMLS positive mappings, the provenance is given as \<UMLS\_header\>:\<CUI\>; for UMLS negative mappings, as \<UMLS\_header\>. Example data consisting of two positive and two negative mappings:

`CPT:90281		DRUGBANK:DB00028	1	UMLS2017AA:C0358321`
`CPT:90283		DRUGBANK:DB00028	1	UMLS2017AA:C0358321`
`CPT:83937		DRUGBANK:DB00426	0	UMLS2017AA`
`CPT:1014233		DRUGBANK:DB05907	0	UMLS2017AA`

## Other

String processing utilities are found in `string_utils.py`.

Constants are found in `constants.py`.

Reused file paths are found in `paths.py`.