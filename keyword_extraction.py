import yake
import spacy
import os
import argparse
import nltk
import numpy as np
from multi_rake import Rake
from parse_data import parse_corp_tokenized
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from outline_extraction_text_rank import outline_by_textrank
import pytextrank

nltk.download('stopwords')
nltk_stopwords = stopwords.words('german')
stopwords = nltk_stopwords + [x.capitalize() for x in nltk_stopwords]
print(stopwords)


def extract_keywords_textrank(texts: list, top_k=20, lemmatize=True, stop_words=stopwords):
    nlp = spacy.load("de_core_news_sm")
    nlp.add_pipe("textrank")
    kyws = []
    i = 1
    for text in texts:
        print(i)
        i+=1
        doc = nlp(text)
        # top-ranked phrases in the document
        if lemmatize:
            doc_kyws = []
            lemmas = set()
            for phrase in doc._.phrases[:top_k * 2]:
                if len(doc_kyws) == top_k:
                    break
                # get the spans;
                spans = doc[phrase.chunks[0].start: phrase.chunks[0].end]

                # some chunks contain the same term repeated several times (represents several occurrences?)
                spans_repeat = [doc[x.start: x.end] for x in phrase.chunks]
                '''print('Chunks:', phrase.chunks, type(phrase.chunks))
                print('Spans wo repeat:', spans)
                print('Spans orig:', spans_repeat)
                print(spans[0].lemma_)'''

                # create the lemmatized version of this span
                span_lemmatized = []
                for tok in spans:
                    if tok.lemma_ in stop_words:
                        continue
                    span_lemmatized.append(tok.lemma_)

                #print('Lemmatized span', span_lemmatized)
                # if such a lemma has already been seen, don't add it to keywords
                # handles cases such as diese Arme	0.03798767039576459, meine Arme	0.03798767039576459
                if ''.join(span_lemmatized) in lemmas:
                    continue

                lemmas.add(''.join(span_lemmatized))
                doc_kyws.append((phrase.text, phrase.rank, phrase.count))
            kyws.append(doc_kyws)
        else:
            kyws.append([(phrase.text, phrase.rank, phrase.count) for phrase in doc._.phrases[:top_k]])
    return kyws


def tf_idf(txts: list, top_k=20, pos_tagging=True):
    nlp = spacy.load("de_core_news_sm")
    top_k_terms = []
    tf = TfidfVectorizer(analyzer='word', stop_words=stopwords, lowercase=False)
    X = tf.fit_transform(txts)
    ind2term = dict((v, k) for k, v in tf.vocabulary_.items())  # reverse mapping from term-index to index-term
    for row in X.toarray():
        indices = np.argsort(-row)  # sort in descending order
        ix = indices[:top_k * 2]  # indices of top_k elements
        top_k_values = row[ix]  # get top-scoring keys
        top_terms_doc = []
        if pos_tagging:
            for i, idx in enumerate(ix):
                if len(top_terms_doc) == top_k:
                    break
                doc = nlp(ind2term[idx])  # pos tags
                # exclude auxiliary verbs, particles, adpositions and adverbs
                # TODO: add more tags?
                for x in doc:
                    if x.pos_ == 'AUX' or x.pos_ == 'PART' or x.pos_ == 'ADP' or x.pos_ == 'ADV':
                        continue
                    # print(x.pos_, x.text)
                    top_terms_doc.append((ind2term[idx], top_k_values[i]))
        else:
            top_terms_doc = [(ind2term[idx], top_k_values[i]) for i, idx in enumerate(ix)]
        top_k_terms.append(top_terms_doc)
        '''print(indices)
		print(ix)
		print(top_k_values)
		print('-----')
		print()'''
    return top_k_terms


def extract_keywords_rake(texts: list, top_k=20):
    from rake_nltk import Rake
    kyws = []
    # Extraction given the list of strings where each string is a sentence.
    for txt in texts:
        r = Rake(language="german")
        r.extract_keywords_from_text(txt)
        keyw_tup = r.get_ranked_phrases_with_scores()[:top_k]
        kyws.append([(x[1], x[0]) for x in keyw_tup])  # turns (score, keyw) into (keyw, score)
    return kyws


def extract_keywords_yake(texts: list, top_k=20):
    kws = []
    for doc in texts:
        kw_extractor = yake.KeywordExtractor(top=top_k, lan='de')
        kws.append(kw_extractor.extract_keywords(doc))
    return kws


# returns weird n-grams that occur together once
def extract_keywords_multi_rake(texts: list, top_k=20):
    rake = Rake(language_code="de")
    return [rake.apply(txt)[:top_k] for txt in texts]


def extract_keywords_keybert(docs: list):
    kw_model = KeyBERT(model="paraphrase-multilingual-MiniLM-L12-v2")
    return [kw_model.extract_keywords(t, stop_words=stopwords) for t in docs]


def write_keywords_to_file(out_file: str, ids: list, keywords: list):
    for i, doc in enumerate(keywords):
        tmp_out = out_file.format(ids[i])
        with open(tmp_out, 'w', encoding='utf8') as f:
            for t in doc:
                f.write(str(t[0]) + '\t' + str(t[1]))
                f.write('\n')


def print_results(res: list):
    for r in res:
        print(r)
    print('---------------------------------')
    print()


def write_outline(outfile: str, outline):
    with open(outfile, 'w', encoding='utf8') as f:
        f.write(outline)

def apply_keyw_extraction(algorithm:str, docs, top_k):
    if algorithm == 'rake':
        feats = extract_keywords_rake(docs, top_k=top_k)
    elif algorithm == 'yake':
        feats = extract_keywords_yake(docs, top_k=top_k)
    elif algorithm == 'keybert':
        feats = extract_keywords_keybert(docs)
    elif algorithm == 'multirake':
        feats = extract_keywords_multi_rake(docs, top_k=top_k)
    elif algorithm == 'tfidf':
        feats = tf_idf(docs, top_k=top_k)
    elif algorithm == 'textrank':
        feats = extract_keywords_textrank(docs, top_k=top_k)
    else:
        raise ValueError('Unsupported algorithm.')
    return feats

def perform_extr(out_path, docs, idxs, algorithm, top_k=20, print_to_screen=True):
    out_path = out_path + algorithm + '/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    keywords = apply_keyw_extraction(algorithm, docs, top_k=top_k)

    write_keywords_to_file(out_path + '{}.txt', idxs, keywords)

    if print_to_screen:
        print_results(keywords)



if __name__ == '__main__':
    '''parser = argparse.ArgumentParser(description='Extract outlines from draco_texts.')
	parser.add_argument('action', choices=['dracor', 'dta'])
	parser.add_argument('in_file', type=str,
						help='Path to the corpus with plays.')
'''
    # dta_txts, idx = parse_corp_tokenized('../test_folder/parsed_dta_with_scene', keep_speakers=True, corpus='dta', add_newline=True)

    '''dr_txt, idx = parse_corp_tokenized('../test_folder/parsed_dracor_speech_by_scenes', keep_speakers=True, corpus='dracor', add_newline=True)

	print(len(dr_txt), len(idx))'''

    # print(dr_txt[0])
    '''
	txts = dr_txt[0:2]
	ids = idx[0:2]
	k = 20
	algs = ['tfidf', 'textrank', 'rake', 'yake', 'keybert', 'multirake',  'textrank']


	# extract keywords directly from the play
	path_out = 'test_folder/kw_out/dracor/'
	for a in algs:
		perform_extr(path_out, txts, ids, a, top_k=k, print_to_screen=True)'''

    '''# extract outlines + keywords from outlines
	path_out = '../outline_extraction/test_folder/kw_outline_out/dracor/'
	outline_out = 'test_folder/outline/dracor/'
	if not os.path.exists(outline_out):
		os.makedirs(outline_out)
	outlines = []
	for i,x in enumerate(txts):
		outl = outline_by_textrank(x)
		write_outline(outline_out + str(ids[i]) + '.txt', outl)
		outlines.append(outl)
		print('Outline:', outl)

	for a in algs:
		perform_extr(path_out, outlines, ids, a, top_k=k, print_to_screen=True)'''

    # test keyword extraction
    with open('../output/outline/wo_speakers/dracor/$id_ger000090_1.txt', 'r', encoding='utf8') as f:
        tx = f.readlines()
        print(tx)
        docs_kw = extract_keywords_textrank(tx)
        for doc in docs_kw:
            for kw in doc:
                print(kw)