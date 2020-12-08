import spacy
from spacy.lang.pt.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
import argparse
from rouge import rouge
import pandas as pd
import re
import unicodedata


def summarization(text, language, nb_sentences):
    import en_core_web_sm
    nlp = en_core_web_sm.load()

    doc = nlp(text)

    corpus = [sent.text.lower() for sent in doc.sents]

    cv = CountVectorizer(stop_words=list(STOP_WORDS))
    cv_fit = cv.fit_transform(corpus)
    word_list = cv.get_feature_names()
    count_list = cv_fit.toarray().sum(axis=0)

    """
    The zip(*iterables) function takes iterables as arguments and returns an iterator. 
    This iterator generates a series of tuples containing elements from each iterable. 
    Let's convert these tuples to {word:frequency} dictionary"""

    word_frequency = dict(zip(word_list, count_list))

    val = sorted(word_frequency.values())

    # Check words with higher frequencies
    higher_word_frequencies = [
        word for word, freq in word_frequency.items() if freq in val[-3:]]
    print("\nWords with higher frequencies: ", higher_word_frequencies)

    # gets relative frequencies of words
    higher_frequency = val[-1]
    for word in word_frequency.keys():
        word_frequency[word] = (word_frequency[word]/higher_frequency)

    # SENTENCE RANKING: the rank of sentences is based on the word frequencies
    sentence_rank = {}
    for sent in doc.sents:
        for word in sent:
            if word.text.lower() in word_frequency.keys():
                if sent in sentence_rank.keys():
                    sentence_rank[sent] += word_frequency[word.text.lower()]
                else:
                    sentence_rank[sent] = word_frequency[word.text.lower()]
            else:
                continue

    top_sentences = (sorted(sentence_rank.values())[::-1])
    top_sent = top_sentences[:nb_sentences]

    # Mount summary
    summary = []
    for sent, strength in sentence_rank.items():
        if strength in top_sent:
            summary.append(sent)

    # return orinal text and summary
    return text, summary

def read_excel():
    """Reads excel file and return two lists of strings, the first contains human summaries, the second is the actual news"""

    def data_preprocessing(text_line):
        """Pre processing the data so as to obtain clean sentences to generate tokens.
          Doing lower casing,normalization,removing special characters etc. """
        text_line = str(text_line).lower()
        text_line = unicodedata.normalize('NFKD', text_line).encode('ascii', 'ignore').decode(
            'utf-8', 'ignore')  # for converting é to e and other accented chars
        text_line = re.sub(r"http\S+", "", text_line)
        text_line = re.sub(r"there's", "there is", text_line)
        text_line = re.sub(r"i'm", "i am", text_line)
        text_line = re.sub(r"he's", "he is", text_line)
        text_line = re.sub(r"she's", "she is", text_line)
        text_line = re.sub(r"it's", "it is", text_line)
        text_line = re.sub(r"that's", "that is", text_line)
        text_line = re.sub(r"what's", "that is", text_line)
        text_line = re.sub(r"where's", "where is", text_line)
        text_line = re.sub(r"how's", "how is", text_line)
        text_line = re.sub(r"\'ll", " will", text_line)
        text_line = re.sub(r"\'ve", " have", text_line)
        text_line = re.sub(r"\'re", " are", text_line)
        text_line = re.sub(r"\'d", " would", text_line)
        text_line = re.sub(r"\'re", " are", text_line)
        text_line = re.sub(r"won't", "will not", text_line)
        text_line = re.sub(r"can't", "cannot", text_line)
        text_line = re.sub(r"n't", " not", text_line)
        text_line = re.sub(r"n'", "ng", text_line)
        text_line = re.sub(r"'bout", "about", text_line)
        text_line = re.sub(r"'til", "until", text_line)
        text_line = re.sub(r"\"", "", text_line)
        text_line = re.sub(r"\'", "", text_line)
        text_line = re.sub(r' s ', "", text_line)
        text_line = re.sub(r"&39", "", text_line)
        text_line = re.sub(r"&34", "", text_line)
        text_line = re.sub(
            r"[\[\]\\0-9()\"$#%/@;:<>{}`+=~|.!?,-]", "", text_line)
        text_line = re.sub(r"&", "", text_line)
        text_line = re.sub(r"\\n", "", text_line)
        text_line = text_line.strip()
        return text_line

    raw_data = pd.read_excel('./files/news_inshort.xlsx')

    short_summary, actual_news = pd.DataFrame(), pd.DataFrame()

    short_summary['short'] = raw_data['short']
    actual_news['long'] = raw_data['long']

    short_summary['short'] = short_summary['short'].apply(
        lambda x: data_preprocessing(x))
    actual_news['long'] = actual_news['long'].apply(
        lambda x: data_preprocessing(x))

    human_summaries = short_summary['short'].astype(str).values.tolist()
    complete_news = actual_news['long'].astype(str).values.tolist()

    return human_summaries, complete_news


if __name__ == '__main__':

    # Read the excel file
    human_summaries, complete_news = read_excel()

    # This list will hold the generated summaries
    output_summaries = []

    for news in complete_news[:10]:
        text, summary = summarization(news, 'english', 1)
        output_summaries.append(summary[0].text)

    # Evaluate the first N summaries with ROUGE
    output_summaries = [text for text in output_summaries[:10]]
    human_summaries = [text for text in human_summaries[:10]]

    # Run ROUGE evaluation metric
    print(rouge(output_summaries, human_summaries))
