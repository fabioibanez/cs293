import os
from pathlib import Path

import pandas as pd
import spacy
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models import Phrases, CoherenceModel
from gensim.models.phrases import Phraser
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from ldamallet import LdaMallet

# Load spacy model for lemmatization
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

stop_words = set(STOPWORDS)
stop_words.update(['inaudible', 'crosstalk'])

custom_stop_words = {"inaudible", "crosstalk"}

ncte_single_utterances = pd.read_csv('ncte_single_utterances.csv')
paired_annotations = pd.read_csv('paired_annotations.csv')
student_reasoning = pd.read_csv('student_reasoning.csv')
transcript_metadata = pd.read_csv('transcript_metadata.csv')

df = ncte_single_utterances.merge(
    student_reasoning[['comb_idx', 'NCTETID', 'student_reasoning']],
    on='comb_idx',
    how='left'
)
df = df.merge(
    paired_annotations[['exchange_idx', 'student_on_task', 'teacher_on_task', 'high_uptake', 'focusing_question']].rename(columns={
        'exchange_idx': 'comb_idx',
    }),
    on='comb_idx',
    how='left'
)

student_df = df[df['speaker'].isin(['student', 'multiple students'])]
# collapse student utterances by OBSID (one document per session)
student_docs = student_df.groupby('OBSID')['text'].apply(" ".join).reset_index()




# run preprocessing and bigram models to tokenize, remove stop words, and create bigrams
def preprocess(text):
    """Tokenize, remove stopwords, and lemmatize text."""
    doc = nlp(text)
    result = []
    for token in doc:
        # Lemmatize and filter
        if (
            not token.is_punct and 
            token.lemma_.isalpha() and
            token.lemma_.lower() not in custom_stop_words
            ):
            result.append(token.lemma_.lower())
    return result



student_texts = student_docs['text'].astype(str).apply(preprocess).tolist()
student_bigram = Phrases(student_texts, min_count=5, threshold=10)
student_bigram_mod = Phraser(student_bigram)
student_texts_bigrams = [student_bigram_mod[doc] for doc in student_texts]


student_dictionary = Dictionary(student_texts_bigrams)
student_dictionary.filter_extremes(no_below=5, no_above=0.8)
# Explicitly remove unwanted placeholder tokens from the dictionary
bad_tokens = ['multiple_comment', 'multiple_answer', 'multiple_question']
bad_ids = [student_dictionary.token2id[tok] for tok in bad_tokens if tok in student_dictionary.token2id]
if bad_ids:
    student_dictionary.filter_tokens(bad_ids=bad_ids)
    student_dictionary.compactify()
student_corpus = [student_dictionary.doc2bow(text) for text in student_texts_bigrams]

def coherence_score(model, texts, dictionary):
    cm_cv = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
    cm_umass = CoherenceModel(model=model, corpus=student_corpus, dictionary=dictionary, coherence='u_mass')
    cm_uci = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_uci')
    cm_npmi = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_npmi')
    return cm_cv.get_coherence(), cm_umass.get_coherence(), cm_uci.get_coherence(), cm_npmi.get_coherence()


def run_gensim_lda(num_topics_list):
    results = []
    for k in num_topics_list:
        lda_model = LdaModel(
            corpus=student_corpus,
            id2word=student_dictionary,
            num_topics=k,
            random_state=42,
            alpha='auto',
            eta='auto'
        )
        coh_cv, coh_umass, coh_uci, coh_npmi = coherence_score(lda_model, student_texts_bigrams, student_dictionary)
        results.append((k, coh_cv, coh_umass, coh_uci, coh_npmi, lda_model))
        print(f"Gensim LDA k={k} coherence cv={coh_cv:.4f} coherence u_mass={coh_umass:.4f} coherence uci={coh_uci:.4f} coherence npmi={coh_npmi:.4f}")
    return results


def run_mallet_lda(num_topics_list, mallet_path):
    mallet_binary = Path(mallet_path)
    if not mallet_binary.exists():
        print(f"Mallet not found at {mallet_binary}. Set MALLET_PATH to your mallet binary.")
        return []

    results = []
    for k in num_topics_list:
        mallet_model = LdaMallet(
            str(mallet_binary),
            corpus=student_corpus,
            num_topics=k,
            id2word=student_dictionary,
            iterations=1000,
            random_seed=42
        )
        mallet_model.save(f'mallet_lda_k{k}.model')
        coh_cv, coh_umass, coh_uci, coh_npmi = coherence_score(mallet_model, student_texts_bigrams, student_dictionary)
        results.append((k, coh_cv, coh_umass, coh_uci, coh_npmi, mallet_model))
        print(f"Mallet LDA k={k} coherence cv={coh_cv:.4f} coherence u_mass={coh_umass:.4f} coherence uci={coh_uci:.4f} coherence npmi={coh_npmi:.4f}")
    return results


if __name__ == "__main__":
    topic_range = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    mallet_path = os.getenv("MALLET_PATH", "/usr/local/bin/mallet")
    mallet_path = '/opt/homebrew/bin/mallet'

    # print("Running Gensim LDA on student texts...")
    # gensim_results = run_gensim_lda(topic_range)
    # gensim_coherence = [res[1] for res in gensim_results]

    print("\nRunning Mallet LDA (if available) on student texts...")
    mallet_results = run_mallet_lda(topic_range, mallet_path)
    mallet_coherence = [res[1] for res in mallet_results]

    # if gensim_results:
    #     best_gensim = max(gensim_results, key=lambda x: x[1])
    #     print(gensim_coherence)
    #     print(f"\nBest Gensim LDA: k={best_gensim[0]} coherence={best_gensim[1]:.4f}")
    #     for idx, topic in best_gensim[2].print_topics(-1):
    #         print(f"Topic {idx}: {topic}")

    if mallet_results:
        best_mallet = max(mallet_results, key=lambda x: x[1])

        print(f"\nBest Mallet LDA: k={best_mallet[0]} coherence cv={best_mallet[1]:.4f} u_mass={best_mallet[2]:.4f} uci={best_mallet[3]:.4f} npmi={best_mallet[4]:.4f}")
        for idx, topic in best_mallet[5].print_topics(-1):
            print(f"Topic {idx}: {topic}")


        #print all mallet topics for all topic_range
        for k, coh_cv, coh_umass, coh_uci, coh_npmi, model in mallet_results:

            print(f"\nMallet LDA Topics for k={k} (coherence cv={coh_cv:.4f}, u_mass={coh_umass:.4f}, uci={coh_uci:.4f}, npmi={coh_npmi:.4f}):")
            for idx, topic in model.print_topics(-1):
                print(f"Topic {idx}: {topic}")

        # save mallet coherence scores to a csv
        coherence_df = pd.DataFrame(mallet_results, columns=['num_topics', 'coherence_cv', 'coherence_umass', 'coherence_uci', 'coherence_npmi', 'model'])
        coherence_df.to_csv('mallet_lda_coherence_scores.csv', index=False)