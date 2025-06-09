!pip install -q biopython scispacy spacy nltk
!pip install -q https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_ner_bionlp13cg_md-0.5.1.tar.gz

import os
from Bio import Entrez
import spacy
import scispacy
from collections import Counter
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

Entrez.email = "adepu.shriya@gmail.com"  # Replace with your email

def fetch_pubmed_abstracts(query, max_results=5):
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    id_list = record["IdList"]
    abstracts = []
    for pmid in id_list:
        handle = Entrez.efetch(db="pubmed", id=pmid, rettype="abstract", retmode="text")
        abstract_text = handle.read()
        abstracts.append(abstract_text)
    return abstracts

abstracts = fetch_pubmed_abstracts("kinase inhibitors", max_results=5)

nlp = spacy.load("en_ner_bionlp13cg_md")

def extract_compound_contexts(text, nlp):
    doc = nlp(text)
    compounds = []
    sentences = sent_tokenize(text)
    results = []

    for sent in sentences:
        sent_doc = nlp(sent)
        entities = [ent.text for ent in sent_doc.ents if ent.label_ in ["CHEMICAL", "DRUG"]]
        if entities:
            compounds.extend(entities)
            results.append((sent, entities))
    return results, compounds

all_results = []
all_compounds = []

for abs_text in abstracts:
    results, compounds = extract_compound_contexts(abs_text, nlp)
    all_results.extend(results)
    all_compounds.extend(compounds)

compound_freq = Counter(all_compounds)

def highlight_compounds(sentence, compounds):
    highlighted = sentence
    for comp in compounds:
        highlighted = highlighted.replace(comp, f"**{comp.upper()}**")
    return highlighted

print("\n=== Extracted compound mentions and context sentences ===")
for sent, compounds in all_results:
    highlighted = highlight_compounds(sent, compounds)
    summary = f"Mentions {', '.join(compounds)} with related context."
    print(f"\nSentence:\n{highlighted}\nSummary: {summary}")

print("\n=== Compound mention frequency ===")
for comp, freq in compound_freq.most_common():
    print(f"{comp}: {freq}")
