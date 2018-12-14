import pandas
import gensim.corpora as corpora
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from gensim.utils import simple_preprocess
from gensim.models import LdaMulticore
from gensim.models.wrappers import LdaMallet
from gensim.models import CoherenceModel

# import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')

NLTK_STEMMER = SnowballStemmer('english')


def lemmatize_fast(text):
    return NLTK_STEMMER.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


def load_stop_words(language, removal_words=[]):
    """

    :param language:
    :param removal_words:
    :return:
    """
    stop_words = stopwords.words(language)
    stop_words.extend(removal_words)
    return stop_words


def pre_process_text(text):
    """

    :param text:
    :return:
    """
    stop_words = load_stop_words('english')
    tokens = []
    try:
        for token in simple_preprocess(text, deacc=True):
            if token not in stop_words:
                tokens.append(lemmatize_fast(token))
        return tokens
    except:
        return []


def load_data(file_path, text_columns):
    """

    :param file_path:
    :param text_columns:
    :return:
    """
    data_frame = pandas.read_csv(file_path)
    for column in text_columns:
            data_frame[column] = data_frame[column].map(pre_process_text)
    return data_frame


def build_corpus(data_frame):
    dictionary = corpora.Dictionary(data_frame)
    corpus = [dictionary.doc2bow(text) for text in data_frame]
    return dictionary, corpus


def get_text_data(filepath, text_column):
    data_frame = load_data(filepath, text_columns=[text_column])
    dictionary, corpus = build_corpus(data_frame[text_column])
    return data_frame, dictionary, corpus


def get_tokens(data_frame, text_column):
    return data_frame[text_column].values.tolist()


def lda_multicore(corpus, id2word, num_topics):
    lda_model = LdaMulticore(corpus=corpus,
                             id2word=id2word,
                             num_topics=num_topics,
                             random_state=100,
                             chunksize=100,
                             passes=10,
                             workers=4,
                             per_word_topics=True)
    return lda_model


def lda_mallet(mallet_path, corpus, dictionary, num_topics):
    lda_model = LdaMallet(mallet_path, corpus=corpus, id2word=dictionary, num_topics=num_topics)
    return lda_model


def print_topics(lda_model):
    for idx, topic in lda_model.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic))


def build_lda_model(model_type, corpus, dictionary, num_topics):
    if model_type == 'Mallet':
        return lda_mallet('C:\\Users\\Yash\mallet\\bin\\mallet', corpus, dictionary, num_topics)
    if model_type == 'Multicore':
        return lda_multicore(corpus, dictionary, num_topics)


def get_coherence_scores(tokens, corpus, dictionary, topic_steps):
    scores = []
    models = []

    for i in topic_steps:
        print('Building Model with {} #Topics'.format(i))
        model = build_lda_model('Mallet', corpus, dictionary, num_topics=i)
        models.append(model)
        coherence = CoherenceModel(model=model, texts=tokens, dictionary=dictionary, coherence='c_v')
        scores.append(coherence.get_coherence())
        print('Coherence {}'.format(coherence.get_coherence()))
    return models, scores

from optparse import OptionParser

parser = OptionParser()
parser.add_option("-p", "--path", dest="mallet",
                  help="path to mallet")

