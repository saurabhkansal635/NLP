import spacy

nlp = spacy.load(r'en_core_web_sm')


sentence = "Saurabh Contributions june 2019 $1 billion"

doc = nlp(sentence)

for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)