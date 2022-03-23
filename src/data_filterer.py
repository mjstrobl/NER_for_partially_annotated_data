import re
import json

MAX_OCCURRENCE = 100
FILTER_POPULAR_ENTITIES = True
WEXEA_DICT_PATH = '<path to WEXEA dictionaries>'

RE_LINKS = re.compile(r'\[{2}(.*?)\]{2}', re.DOTALL | re.UNICODE)

aliases = json.load(open(WEXEA_DICT_PATH + "aliases.json"))

alias_counter = []
for alias in aliases:
    entities = aliases[alias]
    counter = 0
    for entity in entities:
        counter += entities[entity]

    alias_counter.append((alias,counter))

alias_counter.sort(key=lambda x:x[1],reverse=True)

popular_aliases = set()
if FILTER_POPULAR_ENTITIES:
    for i in range(10000):
        popular_aliases.add(alias_counter[i][0])

c = {}
answers = {'YES':0,'NO':1}
sentences = 0

output_filename = '../data/tmp/sentences_shuffled_filtered.txt'

with open(output_filename,'w') as f_out:
    with open('../data/tmp/sentences_shuffled.txt') as f:
        for line in f:
            line = line.strip()

            if "thumb|" in line or "thumbnail|" in line or line.startswith('#') or line.startswith('*') or line.count(' ') > 128:
                continue

            original_line = "" + line
            answers_good = True
            mentions = []
            idx = 0
            while True:
                match = re.search(RE_LINKS, line[idx:])
                if match:
                    entity = match.group(1)
                    start = match.span()[0]+idx
                    end = match.span()[1]+idx
                    parts = entity.split('|')
                    if len(parts) != 3:
                        line = line[:start] + entity + line[end:]
                        continue
                    entity = parts[0]
                    mention = parts[1]
                    answer = parts[2]
                    if answer not in answers:
                        print(original_line)
                        answers_good = False
                        break

                    idx = end

                    if mention not in popular_aliases:
                        mentions.append((mention,answer))
                    else:
                        before = line[:start] + mention
                        idx = len(before)
                        line = before + line[end:]
                else:
                    break

            #if len(c) > 100000:
            #    break
            if answers_good:
                good = False
                for tuple in mentions:
                    mention = tuple[0]
                    if mention not in c:
                        good = True
                        break
                    elif c[mention] <= MAX_OCCURRENCE:
                        good = True
                        break

                if good:
                    f_out.write(line + '\n')
                    sentences += 1
                    for tuple in mentions:
                        mention = tuple[0]
                        answer = tuple[1]
                        if mention not in c:
                            c[mention] = 0

                        c[mention] += 1

                        answers[answer] += 1

    l = []
    for mention in c:
        l.append((mention,c[mention]))

    l.sort(key=lambda x:x[1],reverse=True)

    for i in range(100):
        print(l[i][0] + '\t' + str(l[i][1]))

    print('answers:')
    print(answers)

    print('sentences:')
    print(sentences)