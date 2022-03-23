import json
import re
import os
from nltk.tokenize import sent_tokenize

RE_LINKS = re.compile(r'\[{2}(.*?)\]{2}', re.DOTALL | re.UNICODE)


ENTITY_TYPE = "FOOD"
WEXEA_DICT_PATH = '<path to WEXEA dictionaries>'
WEXEA_ARTICLE_PATH = "<path to WEXEA article_2 directory>"

redirects = json.load(open(WEXEA_DICT_PATH + 'redirects.json'))
id2title = json.load(open(WEXEA_DICT_PATH + 'id2title.json'))
title2id = json.load(open(WEXEA_DICT_PATH + 'title2id.json'))
aliases_reverse = json.load(open(WEXEA_DICT_PATH + "aliases_reverse.json"))

hierarchy = json.load(open('../data/hierarchy.json'))

keep = json.load(open('../data/typed_hierarchy.txt'))

new_id2title = {}
for id in id2title:
    new_id2title[int(id)] = id2title[id]

id2title = new_id2title

new_hierarchy = {}
for key in hierarchy:
    new_hierarchy[int(key)] = hierarchy[key]

hierarchy = new_hierarchy

important_articles = set()
important_articles_lower = set()

for title in keep:
    response = keep[title]
    if "number" in title.lower():
        adsfds = 0
    if response != 's' and response != 'y':
        continue

    important_articles.add(title)

    category = title.replace("category:",'').replace("Category:",'')
    bracket_index = category.find(' (')
    if bracket_index > -1:
        category = category[:bracket_index]
    important_articles_lower.add(category.lower())


    if category[-1] == 's':
        important_articles_lower.add(category[:-1].lower())

    if title in title2id:
        id = title2id[title]
        if id in hierarchy:
            children = hierarchy[id]

            for i in range(len(children)):
                child = children[i]
                if child in id2title and not id2title[child].startswith("Category:") and not id2title[
                    child].startswith("List of") and not id2title[child].startswith("Template:") and not id2title[
                    child].startswith("File:"):

                    if response == 'y' or (response == 's' and i == 0):
                        article = id2title[child]

                        important_articles.add(article)

                        if article in redirects:
                            article = redirects[article]
                            important_articles.add(article)

                        if article in aliases_reverse:
                            aliases = aliases_reverse[article]
                            for alias in aliases:
                                if aliases[alias] > 1:
                                    important_articles_lower.add(alias.lower())

                        bracket_index = article.find(' (')
                        if bracket_index > -1:
                            article = article[:bracket_index]

                        important_articles_lower.add(article.lower())

                        if article[-1] == 's':
                            important_articles_lower.add(article.lower()[:-1])

if ENTITY_TYPE == "FOOD":
    filename = "../data/important_articles_food"
else:
    filename = "../data/important_articles_drugs"

with open(filename + '.json','w') as f:
    json.dump(list(important_articles),f)

with open(filename + '_lower.json','w') as f:
    json.dump(list(important_articles_lower),f)

with open('../data/tmp/sentences.txt','w') as f_out:
    for subdir, dirs, files in os.walk(WEXEA_ARTICLE_PATH):
        for file in files:
            filename = os.path.join(subdir, file)

            try:
                with open(filename) as f:
                    for line in f:
                        line = line.strip()

                        if line.startswith(';') or line.startswith('='):
                            continue

                        if not line.startswith('*'):
                            previous_end_index = 0
                            found_matches = False
                            while True:
                                match = re.search(RE_LINKS, line[previous_end_index:])
                                if match:
                                    start = match.start() + previous_end_index
                                    end = match.end() + previous_end_index
                                    entity = match.group(1)
                                    alias = entity
                                    pos_bar = entity.find('|')
                                    if pos_bar > -1:
                                        alias = entity[pos_bar + 1:]
                                        entity = entity[:pos_bar]

                                    if entity in redirects:
                                        entity = redirects[entity]
                                    elif entity[0].upper() + entity[1:] in redirects:
                                        entity = redirects[entity[0].upper() + entity[1:]]

                                    alternative_entity = entity[0].upper() + entity[1:]

                                    if alias.lower() in important_articles_lower and (entity in important_articles or alternative_entity in important_articles):
                                        found_matches = True

                                        if end < len(line) and line[end] == "s":
                                            alias = alias + "s"
                                            end += 1
                                        elif end < len(line)-1 and line[end:end+2] == "es":
                                            alias = alias + "es"
                                            end += 2

                                        before = line[:start] + "[[" + entity + "|" + alias + "|YES]]"
                                        previous_end_index = len(before)
                                        line = before + line[end:]
                                    elif alias.lower() in important_articles_lower:
                                        found_matches = True

                                        if end < len(line) and line[end] == "s":
                                            alias = alias + "s"
                                            end += 1
                                        elif end < len(line)-1 and line[end:end+2] == "es":
                                            alias = alias + "es"
                                            end += 2

                                        before = line[:start] + "[[" + entity + "|" + alias + "|NO]]"
                                        previous_end_index = len(before)
                                        line = before + line[end:]
                                    else:
                                        previous_end_index = start + len(alias)
                                        line = line[:start] + alias + line[end:]
                                else:
                                    break

                            if found_matches:
                                sentences = sent_tokenize(line)
                                for sentence in sentences:
                                    if ('[[') in sentence:
                                        f_out.write(sentence + '\n')

                # Do something with the file
            except IOError:
                n = 0
                print("File not accessible: " + filename)




