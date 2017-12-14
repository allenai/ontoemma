
# OntoEMMA ontology matcher

This ontology matcher can be used to generate entity alignments between knowledgebases.

## Installation

Go to the base git directory and run: `./setup.sh`

This will create an `ontoemma` conda environment and install all required libraries.

## Run OntoEMMA (align two KBs)
To run OntoEMMA, use `run_ontoemma.py`. 

For a full description of arguments and usages, run: `python run_ontoemma.py -h`

The wrapper implements the following arguments:

- -p \<model_type> (lr = logistic regression, nn = neural network)
- -m \<model_path>
- -s \<source_ont> 
- -t \<target_ont>
- -i \<input_alignment>
- -o \<output_file>
- -a \<alignment_strategy>
- -g \<cuda_device>

Example usage: 

`python run_ontoemma.py -p nn -m model_path -s source_ont.owl -t target_ont.owl -i input_alignment.tsv -o output_alignment.tsv -g 0`

The above aligns the source and target ontology files using the neural network model given at model_path, using GPU 0. This model specified must be pre-trained.

## Train and evaluate OntoEMMA
To train or evaluate a trained alignment model, use `train_ontoemma.py`. 

For a full description of arguments and usages, run: `python train_ontoemma.py -h`

The wrapper takes the following arguments:

- -p \<model_type> (lr = logistic regression, nn = neural network)
- -m \<model_path>
- -c \<configuration_file>
- -e 
- -d \<evaluation\_data\_file>
- -g \<cuda_device>

Example usage for training:

`python train_ontoemma.py -p lr -m model_path -c configuration_file.json`

The above trains a logistic regression model over the training data specified in the configuration file.

The -e flag specifies evaluation mode. This evaluates the specified model on an input evaluation data set with the same format as the training data. An example usage for evaluation:

`python train_ontoemma.py -e -p lr -m model_path -d evaluation_data_path`

The above evaluates the specified logistic regression model over the data given in evaluation\_data\_path.

## Modules

### OntoEMMA module
The module `OntoEmma` is used for accessing the training and alignment capabilities of OntoEMMA.

#### Train mode
In training mode, the `OntoEmma` module can use the `OntoEmmaLRModel` logistic regression module or AllenNLP to train the model

Training data should be formatted as a jsonlines file where each line specifies the following information:

```json
{
	"label": 1,
	"source_ent": {
		"research_entity_id": ENT_ID,
		"canonical_name": ENT_NAME,
		"aliases": [ENT_ALIAS1, ENT_ALIAS2, ...],
		"definition": ENT_DEFINITION,
		"other_context": [ENT_CONTEXT1, ENT_CONTEXT2, ...]
	},
	"target_ent": {
		...
	}
```

A configuration file should be used to specify the training data and configurations of the model. See examples below:

- LR model: `config/ontoemma_lr_config.json`
- NN model: `config/ontoemma_nn_all.json`

#### Align mode
In alignment mode, the `OntoEmma` module performs the following:

- Load source and target KB files (accepted file formats: json/pickle outputs of kb_utils, OWL, RDF)
- Initialize `CandidateSelection` module for source and target KBs
- Compute similarity scores of candidate pairs using specified model
- Compute neighborhood similarity scores if required
- Compute global alignment using specified alignment strategy
- If `gold_file_path` is specified, evaluate alignment against gold standard
- If `output_file_path` is specified, save alignment to file

### Candidate selection module
The module `CandidateSelection` is used to select candidate matched pairs from the source and target KBs.

`CandidateSelection` is initialized with the following inputs:

- `source_kb` source KB as KnowledgeBase object
- `target_kb` target KB as KnowledgeBase object

Candidates are accessed through the `select_candidates` method, which takes an input research_entity_id from the source KB and returns an ranked list of candidates from the target KB. The candidates are ranked by the sum of their token IDF scores.

### Feature generation module
The module `EngineeredFeatureGenerator` is used to generate features from a candidate pair.

The module generates engineered features using the word and character n-gram tokens of entity aliases, definitions, synonyms, and dependancy parse roots. 

### Extracting training data from UMLS
The module `extract_training_data_from_umls` is used to extract KB and concept mapping data from UMLS for use in ontology matching training and evaluation.

`extract_training_data_from_umls` takes as inputs:

- `UMLS_DIR` directory to UMLS META RRF files
- `OUTPUT_DIR` directory to write KnowledgeBases and mappings

`extract_training_data_from_umls ` produces as output:

- KnowledgeBase objects written to json files in `OUTPUT_DIR/kbs/`. One json object is generated for each KnowledgeBase. Currently, KBs specified in kb_name_list are read from the UMLS subset and converted into KnowledgeBase objects.
- Pairwise KB mapping files are written to tsv files in `OUTPUT_DIR/mappings/`. Mappings are derived as pairs of raw_ids that are mapped under the same concept ID in UMLS.
- Negative mappings are sampled from UMLS as negative training data. The full training data including negatives are written to tsv files in `OUTPUT_DIR/training/`.

NOTE: training data does not need to be extracted from UMLS. The user can provide training alignments from whatever source they would like, but it should be formatted as specified above, and both positive and negative alignments should be provided.

#### Sampling negative data

Hard negatives are sampled using the CandidateSelection module, selecting from candidate pairs that are negative matches. Easy negatives are sampled randomly from the rest of the KB. Currently, 5 hard negatives and 5 easy negatives are sampled for each positive match.

#### Mapping file format

Mapping files take the format given below. For UMLS positive mappings, the provenance is given as \<UMLS\_header\>:\<CUI\>; for UMLS negative mappings, as \<UMLS\_header\>. Example data consisting of two positive and two negative mappings:

```
CPT:90281		DRUGBANK:DB00028	1	UMLS2017AA:C0358321
CPT:90283		DRUGBANK:DB00028	1	UMLS2017AA:C0358321
CPT:83937		DRUGBANK:DB00426	0	UMLS2017AA
CPT:1014233		DRUGBANK:DB05907	0	UMLS2017AA
```