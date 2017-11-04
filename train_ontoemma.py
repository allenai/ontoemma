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
    config_file = None

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
            argv, "hc:m:p:", ["config=", "model_path=", "model_type="]
        )
    except getopt.GetoptError:
        sys.stdout.write('Unknown option... -h or --help for help.\n')
        sys.exit(1)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            sys.stdout.write('Options: \n')
            sys.stdout.write('-c <configuration_file>\n')
            sys.stdout.write('-m <model_location>\n')
            sys.stdout.write('-p <model_type>')
            sys.stdout.write('Example usage: \n')
            sys.stdout.write(
                '  ./train_ontoemma.py -c configuration_file.json -m model_file_path -p nn\n'
            )
            sys.stdout.write('-------------------------\n')
            sys.stdout.write('Accepted model types: nn (neural network), lr (logistic regression)\n')
            sys.stdout.write('-------------------------\n')
            sys.stdout.write('\n')
            sys.exit(0)
        elif opt in ("-c", "--config"):
            config_file = os.path.abspath(arg)
            sys.stdout.write('Configuration file is %s\n' % config_file)
        elif opt in ("-m", "--model"):
            model_path = os.path.abspath(arg)
            sys.stdout.write('Model output path is %s\n' % model_path)
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

    if config_file is not None:
        matcher = OntoEmma()
        matcher.train(
            model_type, model_path, config_file
        )


if __name__ == "__main__":
    main(sys.argv[1:])
