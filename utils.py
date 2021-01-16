import numpy as np
import torch
import re

alphabet = re.compile(r'^[a-zA-Z]+$')

from copy import copy
from collections import defaultdict

def build_graph(matrix):
    graph = defaultdict(list) 

    for idx in range(0, len(matrix)):
        for col in range(idx+1, len(matrix)):
            graph[idx].append((col, matrix[idx][col] ))
    return graph

def BFS(s, end, graph, max_size=-1, black_list_relation=[]):
    visited = [False] * (max(graph.keys())+100) 
  
    # Create a queue for BFS 
    queue = [] 

    # Mark the source node as  
    # visited and enqueue it 
    queue.append((s, [(s, 0)]))
    
    found_paths = []

    visited[s] = True
    
    while queue: 

        s, path = queue.pop(0)

        # Get all adjacent vertices of the 
        # dequeued vertex s. If a adjacent 
        # has not been visited, then mark it 
        # visited and enqueue it 
        for i, conf in graph[s]:
            if i == end:
                found_paths.append(path+[(i, conf)])
                break
            if visited[i] == False:
                queue.append((i, copy(path)+[(i, conf)]))
                visited[i] = True
    
    candidate_facts = []
    for path_pairs in found_paths:
        if len(path_pairs) < 3:
            continue
        path = []
        cum_conf = 0
        for (node, conf) in path_pairs:
            path.append(node)
            cum_conf += conf

        if path[1] in black_list_relation:
            continue

        candidate_facts.append((path, cum_conf))

    candidate_facts = sorted(candidate_facts, key=lambda x: x[1], reverse=True)

    return candidate_facts

def is_word(token):
    if len(token) == 1 and alphabet.match(token) == None:
        return False
    return True

def create_mapping(sentence, return_pt=False, nlp = None, tokenizer=None, use_NER=False, linker=None,
                   use_noun_chunks=True):
    '''Create a mapping
        nlp: spacy model
        tokenizer: huggingface tokenizer
    '''
    doc = nlp(sentence)

    tokens = list(doc)

    chunk2id = {}

    start_chunk = []
    end_chunk = []
    noun_chunks = []

    ner_ranges = list()
    linked_entities = defaultdict(dict)

    if use_NER:
        for chunk in doc.ents:
            noun_chunks.append(chunk.text)
            start_chunk.append(chunk.start)
            end_chunk.append(chunk.end)
            if use_noun_chunks:
                # lets keep track of the ranges, so if we are using noun chunks,
                # we can ignore noun chunks that overlap with NERs
                ner_ranges.append(set(range(chunk.start, chunk.end+1)))

            if linker:
                try:
                    umls_ent, confidence = chunk._.kb_ents[0]
                    disamb_ent = str(linker.kb.cui_to_entity[umls_ent].__getattribute__('concept_id')) + "_" +linker.kb.cui_to_entity[umls_ent].__getattribute__('canonical_name')
                    linked_entities[chunk.text]['text'] = disamb_ent
                    linked_entities[chunk.text]['confidence'] = confidence
                except:
                    if len(str(chunk.text).strip()) > 0:
                        linked_entities[chunk.text]['text'] = "UnLinked_{}".format(chunk.text)
                        linked_entities[chunk.text]['confidence'] = 0




    if use_noun_chunks:
        for chunk in doc.noun_chunks:
            curr_range = range(chunk.start, chunk.end+1)
            overlap_found = False
            for rng in ner_ranges:
                if rng.intersection(curr_range):
                    #found overlap between noun chunks and NER chunks. ignore this one
                    overlap_found = True
                    break
            if not overlap_found:
                if linker:
                    linked_entities[chunk.text]['text'] = "UnLinked_{}".format(chunk.text)
                    linked_entities[chunk.text]['confidence'] = 0
                noun_chunks.append(chunk.text)
                start_chunk.append(chunk.start)
                end_chunk.append(chunk.end)

    sentence_mapping = []
    token2id = {}
    mode = 0 # 1 in chunk, 0 not in chunk
    chunk_id = 0
    for idx, token in enumerate(doc):
        if idx in start_chunk:
            mode = 1
            sentence_mapping.append(noun_chunks[chunk_id])
            token2id[sentence_mapping[-1]] = len(token2id)
            chunk_id += 1
        elif idx in end_chunk:
            mode = 0

        if mode == 0:
            sentence_mapping.append(token.text)
            token2id[sentence_mapping[-1]] = len(token2id)


    token_ids = []
    tokenid2word_mapping = []

    for token in sentence_mapping:
        subtoken_ids = tokenizer(str(token), add_special_tokens=False)['input_ids']
        tokenid2word_mapping += [ token2id[token] ]*len(subtoken_ids)
        token_ids += subtoken_ids

    tokenizer_name = str(tokenizer.__str__)
    if 'GPT2' in tokenizer_name:
        outputs = {
            'input_ids': token_ids,
            'attention_mask': [1]*(len(token_ids)),
        }

    else:
        outputs = {
            'input_ids': [tokenizer.cls_token_id] + token_ids + [tokenizer.sep_token_id],
            'attention_mask': [1]*(len(token_ids)+2),
            'token_type_ids': [0]*(len(token_ids)+2)
        }

    if return_pt:
        for key, value in outputs.items():
            outputs[key] = torch.from_numpy(np.array(value)).long().unsqueeze(0)
    
    return outputs, tokenid2word_mapping, token2id, noun_chunks, linked_entities

def compress_attention(attention, tokenid2word_mapping, operator=np.mean):

    new_index = []
    
    prev = -1
    for idx, row in enumerate(attention):
        token_id = tokenid2word_mapping[idx]
        if token_id != prev:
            new_index.append( [row])
            prev = token_id
        else:
            new_index[-1].append(row)

    new_matrix = []
    for row in new_index:
        new_matrix.append(operator(np.array(row), 0))

    new_matrix = np.array(new_matrix)

    attention = np.array(new_matrix).T

    prev = -1
    new_index=  []
    for idx, row in enumerate(attention):
        token_id = tokenid2word_mapping[idx]
        if token_id != prev:
            new_index.append( [row])
            prev = token_id
        else:
            new_index[-1].append(row)

    
    new_matrix = []
    for row in new_index:
        new_matrix.append(operator(np.array(row), 0))
    
    new_matrix = np.array(new_matrix)

    return new_matrix.T

def index2word(tokenid2word_mapping, token2id):
    tokens = []
    prev = -1
    for token_id in tokenid2word_mapping:
        if token_id == prev:
            continue

        tokens.append(token2id[token_id])
        prev = token_id

    return tokens



if __name__ == '__main__':
    import en_core_web_sm
    from transformers import AutoTokenizer, BertModel
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    encoder = BertModel.from_pretrained('bert-base-cased')
    nlp = en_core_web_sm.load()

    sentence = 'Rolling Stone wrote: “No other pop song has so thoroughly challenged artistic conventions”'
    sentence = 'Dylan sing "Time They Are Changing"'
    inputs, tokenid2word_mapping, token2id, noun_chunks  = create_mapping(sentence, return_pt=True, nlp=nlp, tokenizer=tokenizer)

    outputs = encoder(**inputs, output_attentions=True)
    print(noun_chunks, tokenid2word_mapping, token2id)
