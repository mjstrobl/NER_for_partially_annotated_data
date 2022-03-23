import json
import sys

exclude = {'','',''}

visited = []
queue = []


WEXEA_DICT_PATH = '<path to WEXEA dictionaries>'
ENTITY_TYPE = "FOOD"
CATEGORY = "Category:Food and drink"
#CATEGORY = "Category:Drugs"

def bfs(visited, hierarchy,root):

    keep = {}
    rules = []

    visited.append(root)
    queue.append(root)

    while queue:


        parent = queue.pop(0)
        if parent in hierarchy and parent in id2title and id2title[parent].startswith('Category:'):
            title = id2title[parent]
            print("current title: " + title)

            for tuple in rules:
                rule = tuple[0]
                answer = tuple[1]
                start_end = tuple[2]
                if start_end == 'c':
                    if rule in title.lower():
                        keep[title] = answer
                elif start_end == 's':
                    if title.lower().startswith(rule):
                        keep[title] = answer
                elif start_end == 'e':
                    if title.lower().endswith(rule):
                        keep[title] = answer

            if title in keep:
                if keep[title] != 'n':
                    for child in hierarchy[parent]:
                        if child not in visited and child in id2title and id2title[child].startswith('Category:'):
                            visited.append(child)
                            queue.append(child)
                continue

            # ask user here:
            print()
            print(title)
            print("parent: " + id2title[parent])
            sub_categories = []
            pages = []
            for child in hierarchy[parent]:
                if child in id2title:
                    if id2title[child].startswith('Category:'):
                        sub_categories.append(id2title[child])
                    else:
                        pages.append(id2title[child])

            print("Subcategories:")
            print(sub_categories[:min(10, len(sub_categories))])
            print("Pages:")
            print(pages[:min(10, len(pages))])

            key = input("Press yes (y) or sub only (s) or no (n) or print (p) or write (w <path>) or read (r <path>)")
            if len(key) == 0:
                print('accident?')
                queue.insert(0, parent)
                continue
            if key == 'n':
                print('ignore')
                keep[title] = key
                continue
            elif key == 'p':
                print("print")
                queue.insert(0, parent)
                print(keep)
                continue
            elif key[0] == 'd':
                print('create startswith rule')
                queue.insert(0, parent)
                if len(key) > 5:
                    answer = key[2]
                    start_end = key[4]
                    if answer == 'n' or answer == 'y' or answer == 's':
                        if start_end == 's':
                            rules.append((key[6:], answer,start_end))
                        elif start_end == 'e':
                            rules.append((key[6:], answer, start_end))
                        elif start_end == 'c':
                            rules.append((key[6:], answer, start_end))
                        else:
                            print('start end incorrect')
                    else:
                        print('wrong answer')
                print(rules)
            elif key[0] == 'r':
                print("read")
                tokens = key.strip().split(' ')
                if len(tokens) == 2:
                    keep = json.load(open(tokens[1]))
                else:
                    print('Please provide filename!')
                queue.insert(0, parent)
                print(keep)
                continue
            elif key[0] == 'w':
                print("write")
                queue.insert(0,parent)
                tokens = key.strip().split(' ')
                if len(tokens) == 2:
                    with open(tokens[1],'w') as f:
                        json.dump(keep,f)
                else:
                    print('Please provide filename!')
                continue
            elif key == "y" or key == 's':
                print('yes: ' + key)
                keep[title] = key
                for child in hierarchy[parent]:
                    if child not in visited and child in id2title and id2title[child].startswith('Category:'):
                        visited.append(child)
                        queue.append(child)
            else:
                queue.insert(0, parent)
            print("Queue: " + str(len(queue)))

    return keep

sys.setrecursionlimit(5000)

id2title = json.load(open(WEXEA_DICT_PATH + 'id2title.json'))
title2id = json.load(open(WEXEA_DICT_PATH + 'title2id.json'))
category_redirects = json.load(open(WEXEA_DICT_PATH + 'category_redirects.json'))

hierarchy = json.load(open('../data/hierarchy.json'))

new_id2title = {}
for id in id2title:
    new_id2title[int(id)] = id2title[id]

id2title = new_id2title

print('read dictionaries')

new_hierarchy = {}
for key in hierarchy:
    new_hierarchy[int(key)] = hierarchy[key]

hierarchy = new_hierarchy

print('created new hierarchy')

if CATEGORY in title2id and title2id[CATEGORY] in hierarchy:
    keep = bfs(visited, hierarchy, title2id[CATEGORY])
    with open('typed_hierarchy.txt', 'w') as f:
        json.dump(keep, f)

