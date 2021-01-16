from process import parse_sentence
from mapper import Map, deduplication
from transformers import AutoTokenizer, AutoModel
import argparse
import spacy
from tqdm import tqdm
import json
from REL.db.generic import GenericLookup

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Process lines of text corpus into knowledgraph')
parser.add_argument('input_filename', type=str, help='text file as input')
parser.add_argument('output_filename', type=str, help='output text file')
parser.add_argument('--language_model',default='bert-base-cased', 
                    choices=[ 'bert-large-uncased', 'bert-large-cased', 'bert-base-uncased','roberta-large',
                              'bert-base-cased', 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl',
                              'emilyalsentzer/Bio_ClinicalBERT','bionlp/bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16',
                              "allenai/biomed_roberta_base",
                              'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
                              ],
                    help='which language model to use')
parser.add_argument('--use_cuda', default=True, 
                        type=str2bool, nargs='?',
                        help="Use cuda?")
parser.add_argument('--include_text_output', default=False, 
                        type=str2bool, nargs='?',
                        help="Include original sentence in output")
parser.add_argument('--threshold', default=0.05,
                        type=float, help="Any attention score lower than this is removed")
parser.add_argument('--use_ner', default=False,
                    type=str2bool, help="Use Named Entities (defaults to using spacy named entity)")
parser.add_argument('--use_nouns', default=True,
                    type=str2bool, help="Use noun chunks (defaults to using spacy) for anchor nodes")
parser.add_argument('--spacy_model',default='en_core_sci_lg',
                    choices=['en_core_sci_lg', 'en_core_web_md', 'en_ner_bc5cdr_md'],
                    help='which spacy model to use')
parser.add_argument('--entity_linker',default='REL',
                    choices=['REL', 'SCISPACY'],
                    help='which entity linker to use')

parser.add_argument('--REL_embeddings_path', default=False,
                    type=str, help="REL trained embeddings - embedding lookup for entity linking")
parser.add_argument('--scispacy_kb_name', default='umls',
                    choices=['umls'],
                    type=str, help="Knowledge Base to link using SciSpacy Entity Linker")

args = parser.parse_args()

use_cuda = args.use_cuda


'''Create
Tested language model:

1. bert-base-cased

2. gpt2-medium

Basically any model that belongs to this family should work

'''

language_model = args.language_model
print("Language Model: {}".format(language_model))
nlp = spacy.load(args.spacy_model)
linker = None
if args.entity_linker == 'SCISPACY' and args.scispacy_kb_name:
    from scispacy.linking import EntityLinker
    linker = EntityLinker(resolve_abbreviations=True, name=args.scispacy_kb_name)
    nlp.add_pipe(linker)


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(language_model)
    encoder = AutoModel.from_pretrained(language_model)

    encoder.eval()
    if use_cuda:
        encoder = encoder.cuda()    
    input_filename = args.input_filename
    output_filename = args.output_filename
    include_sentence = args.include_text_output
    emb_path = args.REL_embeddings_path

    print(args)
    print(f"args.use_ner:{args.use_ner}")

    with open(input_filename, 'r') as f, open(output_filename, 'w') as g:
        for idx, line in enumerate(tqdm(f)):
            sentence  = line.strip()
            if len(sentence):
                valid_triplets = []
                all_disamb_ents = dict()
                for sent in nlp(sentence).sents:
                    print(sent)
                    # Match
                    triplets_lst , disamb_ents = parse_sentence(sent.text, tokenizer, encoder, nlp, args, use_cuda=use_cuda,
                                                                linker=linker)
                    for triplets in triplets_lst:
                        valid_triplets.append(triplets)
                    all_disamb_ents.update(disamb_ents)
                if len(valid_triplets) > 0:
                    # Map
                    mapped_triplets = []
                    if args.entity_linker == 'REL':
                        emb = GenericLookup("entity_word_embedding", save_dir=emb_path, table_name="embeddings")
                    else:
                        emb = all_disamb_ents
                    for triplet in valid_triplets:
                        head = triplet['h']
                        tail = triplet['t']
                        relations = triplet['r']
                        conf = triplet['c']
                        if conf < args.threshold:
                            continue

                        mapped_triplet = Map(head, relations, tail, emb=emb, type=args.entity_linker)
                        if 'h' in mapped_triplet:
                            mapped_triplet['c'] = conf
                            mapped_triplets.append(mapped_triplet)
                    output = { 'line': idx, 'tri': deduplication(mapped_triplets) }

                    if include_sentence:
                        output['sent'] = sentence
                    if len(output['tri']) > 0:
                        g.write(json.dumps( output )+'\n')