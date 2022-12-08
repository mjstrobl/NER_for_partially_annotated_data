# NER for Partially Annotated Data

Code and datasets used in our paper "Named Entity Recognition for Partially Annotated Datasets" (https://arxiv.org/abs/2204.09081).

## Setup

1. Install requirements in requirements.txt
2. Create WEXEA dataset: https://github.com/mjstrobl/WEXEA
3. Download CoreNLP: Download CoreNLP (https://stanfordnlp.github.io/CoreNLP/download.html).
4. Adjust upper case variables in python files accordingly. We are using the ``article_2`` directory of the WEXEA output.
5. Download hierarchy.json.tar.gz from https://drive.google.com/file/d/1MIWoUikaRVxrZrR_WlKEYc8TyNXqri0q/view?usp=sharing
6. Extract to data/.

## Create type hierarchy

1. Run ``src/typed_hierarchy_creator.py`` and set appropriate category from Wikipedia.
2. Type 'y' for keep, 'n' for ignore and 's' for keep subcategories only.
3. Once all categories are seen, the algorithm stops.

## Create data in CoNLL format

1. Start server (https://stanfordnlp.github.io/CoreNLP/corenlp-server.html): ``java -mx4g -cp "<path to CoreNLP>/*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000 -ner.model 4class -ner.applyFineGrained false -ner.statisticalOnly true``
2. Run ``process_hierarchy.sh``

## Datasets

Wikipedia category hierarchies, related article names, sentences from Wikipedia and gold annotated datasets for Food and Drugs can be found in the data/ directory.

## Train models

Please find code for all three models used in the paper in src/training. Paths to the datasets need to be adjusted in src/traininig/config/config.json.