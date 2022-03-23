#!/bin/sh

python3 src/wexea_searcher.py

shuf -o data/tmp/sentences_shuffled.txt data/tmp/sentences.txt

python3 src/data_filterer.py

python3 src/sentence_pretagger.py

python3 src/data_to_conll.py