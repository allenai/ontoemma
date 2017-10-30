#!/usr/bin/env python
import os
import sys
import getopt
import nltk
import ssl

from emma.OntoEmma import OntoEmma
import emma.constants


def main(argv):
    model_path = None
    model_type = "nn"
    source_ont_file = None
    target_ont_file = None
    input_alignment_file = None
    output_alignment_file = None

    sys.stdout.write('\n')
    sys.stdout.write('-------------------------\n')
    sys.stdout.write('OntoEMMA version 0.1     \n')
    sys.stdout.write('-------------------------\n')
    sys.stdout.write('https://github.com/allenai/ontoemma\n')
    sys.stdout.write('An ML-based ontology matcher to produce entity alignments between knowledgebases\n')
    sys.stdout.write('\n')

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        nltk.download("stopwords")

    try:
        # TODO(waleeda): use argparse instead of getopt to parse command line arguments.
        opts, args = getopt.getopt(
            argv, "hs:t:i:o:m:p:", ["source=", "target=", "input=", "output=", "model_path=", "model_type="]
        )
    except getopt.GetoptError:
        sys.stdout.write('Unknown option... -h or --help for help.\n')
        sys.exit(1)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            sys.stdout.write('Options: \n')
            sys.stdout.write('-s <source_ontology_file>\n')
            sys.stdout.write('-t <target_ontology_file>\n')
            sys.stdout.write('-i <input_alignment_file>\n')
            sys.stdout.write('-o <output_alignment_file>\n')
            sys.stdout.write('-m <model_location>\n')
            sys.stdout.write('-p <model_type>')
            sys.stdout.write('Example usage: \n')
            sys.stdout.write(
                '  ./OntoEmmaWrapper.py -s source_ont.json -t target_ont.json -i gold_alignment.tsv -o generated_alignment.tsv -m model_serialization_dir -p nn\n'
            )
            sys.stdout.write('-------------------------\n')
            sys.stdout.write('Accepted KB file formats: json, pickle, owl\n')
            sys.stdout.write('Accepted alignment file formats: rdf, tsv\n')
            sys.stdout.write('Accepted model types: nn (neural network), lr (logistic regression)\n')
            sys.stdout.write('Pretrained models can be found at:\n')
            sys.stdout.write('  /net/nfs.corp/s2-research/scigraph/ontoemma/')
            sys.stdout.write('-------------------------\n')
            sys.stdout.write('\n')
            sys.exit(0)
        elif opt in ("-s", "--source"):
            source_ont_file = os.path.abspath(arg)
            sys.stdout.write('Source ontology file is %s\n' % source_ont_file)
        elif opt in ("-t", "--target"):
            target_ont_file = os.path.abspath(arg)
            sys.stdout.write('Target ontology file is %s\n' % target_ont_file)
        elif opt in ("-i", "--input"):
            input_alignment_file = os.path.abspath(arg)
            sys.stdout.write(
                'Input alignment file is %s\n' % input_alignment_file
            )
        elif opt in ("-o", "--output"):
            output_alignment_file = os.path.abspath(arg)
            sys.stdout.write(
                'Output alignment file is %s\n' % output_alignment_file
            )
        elif opt in ("-m", "--model"):
            model_path = os.path.abspath(arg)
        elif opt in ("-p", "--model-type"):
            if arg in emma.constants.IMPLEMENTED_MODEL_TYPES:
                model_type = arg
                sys.stdout.write(
                    'Model type is %s\n' % emma.constants.IMPLEMENTED_MODEL_TYPES[model_type]
                )
            else:
                sys.stdout.write('Error: Unknown model type...\n')
                sys.exit(1)

    sys.stdout.write('\n')

    if source_ont_file is not None and target_ont_file is not None:
        matcher = OntoEmma()
        matcher.align(
            model_type, model_path,
            source_ont_file, target_ont_file,
            input_alignment_file, output_alignment_file
        )


if __name__ == "__main__":
    main(sys.argv[1:])
