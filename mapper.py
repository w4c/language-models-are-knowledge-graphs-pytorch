from constant import invalid_relations_set


def disambiguate_entities(val, emb, emv_type='wiki', type='REL'):
    if type=='REL':
        return emb.wiki(str(val), emv_type)
    try:
        return emb[str(val)]['text']
    except:
        return str("Unknown_{}".format(val))


def Map(head, relations, tail, top_first=True, best_scores = True, emb=False, type='REL'):
    if head == None or tail == None or relations == None:
        return {}
    if not emb:
        emb_path = "../../Documents/wiki_2020/generated"

    if type == 'REL':

        # head_p_e_m = emb.wiki(str(head), 'wiki')
        head_p_e_m = disambiguate_entities(head, emb, type=type)
        if head_p_e_m is None:
            return {}
        tail_p_e_m = disambiguate_entities(tail, emb, type=type)
        # tail_p_e_m = emb.wiki(str(tail), 'wiki')

        if tail_p_e_m is None:
            return {}
        tail_p_e_m = tail_p_e_m[0][0]
        head_p_e_m = head_p_e_m[0][0]
    else:
        head_p_e_m = disambiguate_entities(head, emb, type=type)
        tail_p_e_m = disambiguate_entities(tail, emb, type=type)


    valid_relations = [ r for r in relations if r not in invalid_relations_set and r.isalpha() and len(r) > 1 ]
    if len(valid_relations) == 0:
        return {}
    return { 'h': head_p_e_m, 't': tail_p_e_m, 'r': '_'.join(valid_relations)  }

def deduplication(triplets):
    unique_pairs = []
    pair_confidence = []
    for t in triplets:
        key = '{}\t{}\t{}'.format(t['h'], t['r'], t['t'])
        conf = t['c']
        if key not in unique_pairs:
            unique_pairs.append(key)
            pair_confidence.append(conf)
    
    unique_triplets = []
    for idx, unique_pair in enumerate(unique_pairs):
        h, r, t = unique_pair.split('\t')
        unique_triplets.append({ 'h': h, 'r': r, 't': t , 'c': pair_confidence[idx]})

    return unique_triplets


if __name__ == "__main__":
    emb = GenericLookup("entity_word_embedding", save_dir=sqlite_path, table_name="embeddings")
    p_e_m = emb.wiki("Bob", 'wiki')[:10]
    print(p_e_m)
