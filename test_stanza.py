from stanza.server import CoreNLPClient
from nltk import tokenize

text = "Elli Manning is the brother of Peyton Manning"
#with CoreNLPClient(annotators=["tokenize","ssplit","pos","lemma","depparse","natlog","openie"], be_quiet=False) as client:
with CoreNLPClient(annotators=['tokenize','ssplit','pos','lemma','ner', 'parse', 'depparse','coref','openie'],memory='8G', be_quiet=False) as client:
    text = 'Tom is a smart boy. He knows a lot of things.'
    ann = client.annotate(text)
    modified_text = tokenize.sent_tokenize(text)
    for coref in ann.corefChain:
        antecedent = []
        for mention in coref.mention:
            phrase = []
            for i in range(mention.beginIndex, mention.endIndex):
                phrase.append(ann.sentence[mention.sentenceIndex].token[i].word)
            if antecedent == []:
                antecedent = ' '.join(word for word in phrase)
            else:
                anaphor = ' '.join(word for word in phrase)
                modified_text[mention.sentenceIndex] = modified_text[mention.sentenceIndex].replace(anaphor, antecedent)

    modified_text = ' '.join(modified_text)

    #print(modified_text)
