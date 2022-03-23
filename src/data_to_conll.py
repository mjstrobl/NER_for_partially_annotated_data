import json
import re
from transformers import AutoTokenizer

ENTITY_LENGTH = 6
RE_LINKS = re.compile(r'\[{2}(.*?)\]{2}', re.DOTALL | re.UNICODE)

ENTITY_TYPE = 'FOOD'

def get_actual_words(word_tokens):
    actual_words = []
    for i in range(len(word_tokens)):
        word_token = word_tokens[i]
        if word_token[0] == '#' and len(actual_words) > 0:
            actual_words[-1] += word_token.replace('#', '')
        else:
            actual_words.append(word_token)

    return actual_words

def process_tokens(actual_words,important_articles_lower):
    indices_to_exclude = set()
    found_indices = set()

    excludes = 0

    for j in reversed(range(1, ENTITY_LENGTH)):
        for i in range(len(actual_words)):
            if i in found_indices or i + j > len(actual_words):
                continue

            potential_entity = ' '.join(actual_words[i:i+j])
            if potential_entity.lower() in important_articles_lower:
                found_indices.update(set(range(i, i + j)))
                indices_to_exclude.update(set(range(i, i + j)))
                excludes += 1

    labels = []
    for i in range(len(actual_words)):
        if i in indices_to_exclude:
            labels.append('EXCLUDE')
        else:
            labels.append("O")

    return labels, excludes

def process(important_articles_lower,tokenizer):
    all_excludes = 0
    with open('../data/wiki_' + ENTITY_TYPE.lower() + '_conll.txt','w') as f_out:
        with open('../data/tmp/sentences_tagged.txt') as f:
            for line in f:
                line = line.strip()

                words = []
                labels = []

                while True:
                    match = re.search(RE_LINKS, line)
                    if match:
                        entity = match.group(1)
                        start = match.span()[0]
                        end = match.span()[1]
                        parts = entity.split('|')
                        if len(parts) != 3:
                            line = line[:start] + entity + line[end:]
                            continue
                        entity = parts[0]
                        mention = parts[1]
                        answer = parts[2]

                        before = line[:start]
                        line = line[end:]

                        mention_tokens = tokenizer.tokenize(mention)

                        actual_mention_words = get_actual_words(mention_tokens)
                        before_tokens = tokenizer.tokenize(before)
                        actual_before_words = get_actual_words(before_tokens)

                        before_labels, current_excludes = process_tokens(actual_before_words,important_articles_lower)
                        all_excludes += current_excludes
                        words.extend(actual_before_words)
                        labels.extend(before_labels)
                        words.extend(actual_mention_words)
                        if answer == 'YES':
                            label = 'B-' + ENTITY_TYPE
                        else:
                            label = 'NO'

                        while len(labels) < len(words):
                            labels.append(label)
                            if answer == 'YES':
                                label = 'I-' + ENTITY_TYPE
                            else:
                                label = 'NO'


                    else:
                        break

                before_tokens = tokenizer.tokenize(line)
                actual_before_words = get_actual_words(before_tokens)
                before_labels, current_excludes = process_tokens(actual_before_words, important_articles_lower)
                all_excludes += current_excludes
                words.extend(actual_before_words)
                labels.extend(before_labels)

                assert len(words) == len(labels)

                for i in range(len(words)):
                    f_out.write(words[i] + ' ' + labels[i] + '\n')

                f_out.write('\n')

    print("all excludes")
    print(all_excludes)

if ENTITY_TYPE == "FOOD":
    important_articles_lower = json.load(open("../data/important_articles_food_lower.json"))
else:
    important_articles_lower = json.load(open("../data/important_articles_drugs_lower.json"))


print('read important articles')
model_name_or_path = "bert-base-cased"

tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=None,
        use_fast=True
    )

process(important_articles_lower,tokenizer)