from stanza.server import CoreNLPClient, StartServer
import re

RE_LINKS = re.compile(r'\[{2}(.*?)\]{2}', re.DOTALL | re.UNICODE)

props = {"ssplit.isOneSentence": True, "ner.applyNumericClassifiers": False,
             "ner.model": "edu/stanford/nlp/models/ner/english.conll.4class.distsim.crf.ser.gz",
             "ner.applyFineGrained": False, "ner.statisticalOnly": True, "ner.useSUTime": False}

annotators = ['tokenize', 'ssplit', 'pos', 'lemma', 'ner']

client = CoreNLPClient(
    annotators=annotators,
    properties=props,
    timeout=60000, endpoint="http://localhost:9000", start_server=StartServer.DONT_START, memory='16g')

with open('../data/tmp/sentences_tagged.txt','w') as f_out:
    filepath = '../data/tmp/sentences_shuffled_filtered.txt'
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            entities = []
            annotated_indices = set()
            no_ranges = []
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

                    if answer == 'YES':
                        entities.append((start,len(mention),entity,mention,'YES'))
                        annotated_indices.update(range(start, start+len(mention)))
                    else:
                        no_ranges.append(((start,len(mention),entity,mention,'NO'),range(start, start + len(mention))))

                    line = line[:start] + mention + line[end:]

                else:
                    break

            ignored_no_ranges_indices = set()
            annotation = client.annotate(line, properties=props, annotators=annotators)
            for i, sent in enumerate(annotation.sentence):
                for mention in sent.mentions:
                    ner = mention.ner
                    tokens = sent.token[mention.tokenStartInSentenceInclusive:mention.tokenEndInSentenceExclusive]
                    if len(tokens) == 1 and tokens[0].pos.startswith("PRP"):
                        continue

                    start = tokens[0].beginChar
                    end = tokens[-1].endChar

                    alias = line[start:end]
                    if len(annotated_indices.intersection(set(range(start, start + len(alias))))) == 0:
                        prefix = ''
                        for j in range(len(no_ranges)):
                            no_range = set(no_ranges[j][1])
                            if len(no_range.intersection(set(range(start, start + len(alias))))) > 0:
                                ignored_no_ranges_indices.add(j)
                                prefix = 'NO_'
                                break

                        if ner == "LOCATION":
                            ner = prefix + "LOC"
                        elif ner == 'ORGANIZATION':
                            ner = prefix + 'ORG'
                        elif ner == "PERSON":
                            ner = prefix + 'PER'
                        else:
                            ner = prefix + ner
                        entities.append((start, len(alias), alias,alias, ner))

            for j in range(len(no_ranges)):
                if j not in ignored_no_ranges_indices:
                    entities.append(no_ranges[j][0])

            entities.sort(key = lambda x:x[0],reverse=True)
            for tuple in entities:
                line = line[:tuple[0]] + '[[' + tuple[2] + '|' + tuple[3] + "|" + tuple[4] + ']]' + line[tuple[0] + tuple[1]:]

            if re.match('.*[?.!:]$', line) is not None:
                f_out.write(line + '\n')
