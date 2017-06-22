# -*- coding: utf-8 -*-
import csv
import string
from nltk import stem, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import cPickle
from nltk.corpus import stopwords
from sklearn import preprocessing
from django.utils.encoding import smart_unicode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import numpy as np

"""
    Classe de métodos úteis para o processamento de texto.
"""


def read_file(path):
    """
        Método responsável por ler os arquivos csv da base de dados e retornar os reviews e labels
        para serem trabalhados
    :param path: endereço do arquivo a ser lido
    :return: database = base com todos os reviews lidos pelo método
             labels = os labels referentes a cada um dos reviews
    """
    database = []
    labels = []
    with open(path, "rb") as csv_file:
        next(csv_file)
        spamreader = csv.reader(csv_file, delimiter = str(u','))
        for row in spamreader:
            row = map(lambda x: x if x != '' else '0', row)
            database.append(row[0])
            labels.append(row[1:3])

    return database, labels

def merge_lists(lists):
    """
        Método responsável por unir 3 listas em uma só
    :param lists: listas a serem unidas
    :return: uma lista com as informações contidas nas listas prévias
    """
    return lists[0] + lists[1] + lists[2] + lists[3]

def find_ngrams(input_list, n):
    """
        Método que calcula o n_gram para uma lista de sentença/frase
    :param input_list: lista com as sentenças/frase a serem extraídos
    :param n: tamanho do n_gram
    :return: retorna uma lista com os n_grams das sentenças/frase extraídos
    """
    return zip(*[input_list[i:] for i in range(n)])

def generate_n_gram_for_a_sentence(sentence = str, range = int):
    """
        Método que calcula o n_gram para uma lista de sentença/frase
    :param sentence: sentença/frase a ter seu n gram extraído
    :param range: tamanho do n gram
    :return: uma lista com o n gram da sentença/frase
    """
    vectorizer = CountVectorizer(ngram_range=(range, range))
    analyzer = vectorizer.build_analyzer()
    return analyzer(sentence)

def generate_n_gram_database(database = [], n = int, file_name = str):
    """
        Método responsável por extrai as características do tipo n gram de uma base de texto.
    :param database: Lista com todas as sentenças a terem seu n gram extraído
    :param n: tamanho do n gram
    :param file_name: arquivo onde será salvo o resultado
    :return: a base com o n gram extraído
    """
    n_gram_database = []
    for sentence in database:
        n_gram_sentence = generate_n_gram_for_a_sentence(sentence=sentence, range=n)
        n_gram_database.append(n_gram_sentence)
    cPickle.dump(n_gram_database, open(file_name, 'wb'))
    return n_gram_database

def vectorize_database_tfidf(database):
    """
        Método que vetoriza a base se dados usando a técnica de TF-IDF
    :param database: vetor com as frases utilizadas para compor a base de dados
    :return: matriz com os dados extraídos em forma de frequência Tf-IDF
    """

    database = map(lambda x: x.lower(), database)
    database_ = [tokenize(sentence) for sentence in database]

    pt_stop_words = set(stopwords.words('portuguese'))
    pt_stop_words.add('pra')
    pt_stop_words.add('para')
    vectorizer = TfidfVectorizer(max_df=0.75, max_features=5000, lowercase=False, min_df=2, stop_words=pt_stop_words,
                                 ngram_range=(1, 4),
                                 use_idf=True)

    data = vectorizer.fit_transform(database_)

    return data.todense(), vectorizer

def vectorize_database_hash(database):
    """
        Método que vetoriza a base se dados usando a técnica de hash
    :param database: vetor com as frases utilizadas para compor a base de dados
    :return: matriz com os dados extraídos em forma de frequência Hash
    """
    pt_stop_words = set(stopwords.words('portuguese'))
    vectorizer = HashingVectorizer(n_features=2000,
                                   stop_words=pt_stop_words, ngram_range=(2, 6), lowercase=True,
                                   non_negative=False, norm='l2',
                                   binary=False)
    data = vectorizer.fit_transform(database)

    return data

def split_database(database=[], labels =[], test_size =float):
    """
        Método que separa a base em vetores de treino e teste de acordo com uma porcentagem,
        dada no intervalo [0-1]
    :param database: matriz de dados já extraídos utilizando alguma técnica de extração
    :param labels: Classificação referente a cada um dos vetores da base de dados
    :param test_size: tamanho da base de treino dada em forma de porcentagem no intervalo [0-1].
    :return: a base dividida em 4 matrizes que são: base de treino, base de testes,
        labels de treino, labels de teste. Sempre nessa ordem
    """
    database = np.array(database)
    labels = np.array(labels)
    return train_test_split(database, labels, test_size=test_size)

def load_database(metodo = 'tfidf'):
    """
        Método que lê os arquivos csv da base de dados e já devole a base vetorizada com o método TF-IDF
        por definição mas utilizando a variável metoto e mudando para hash o retorno é utilizando o métoro
        de Hash
    :param metodo: variável que define o método estatístico utilizado para extrair as características
    :return: database: base de dados extraído utilizando o método chamado com a variável metodo
             labels: rótulos correspondentes para cada vetor da matriz de base
             vectorizer: instância responsável por transformar os novos inputs em
    """
    import os
    database = []
    labels = []
    root = "textClassification/SentimentAnalisys/Data"
    for path_to_file in os.listdir(root):
        data, labe = read_file(os.path.join(root, path_to_file))
        database.append(data)
        labels.append(labe)
    database = merge_lists(database)
    labels = merge_lists(labels)
    labels = np.array(labels)
    labels = labels
    replace_data(labels, 'Pro', 'Product')
    replace_data(labels, 'Prod', 'Product')
    replace_data(labels, ' ', '0')

    database, vectorizer = vectorize_database_tfidf(database)
    return database, labels, vectorizer

def replace_data(list_labels, itens_to_replace, replacement_value):
    """
        Método que retira as lábels erradas causadas por algum problema de digitação pela label correta
    :param list_labels: labels a serem ajustadas
    :param itens_to_replace: item que deve ser removido
    :param replacement_value: item que deve ser substituido
    :return: a lista de labels com os valores corretos.
    """
    indices_to_replace = [i for i,x in enumerate(list_labels) if x[0]==itens_to_replace]
    for i in indices_to_replace:
        list_labels[i][0] = replacement_value

def encoding_labels(labels, labels_to_encode):
    """
        Método que transforma as labels de categóricas em números para poder gerar a classificação
    :param labels: conjunto de rótulos
    :param labels_to_encode: exemplo de cada rotulo que existe no conjunto de rótulos.
    :return: o vetor com as classe a qual cada um pertence em formato de inteiros.
    """
    le = preprocessing.LabelEncoder()
    le.fit(labels_to_encode)
    return le.transform(labels)

def training_models(train_dataset, labels_for_train_dataset_1, labels_for_train_dataset_2):
    """
        Método responsável por treinar os classificadores utilizados no sistema.
    :param train_dataset: lista de sentenças utilzadas para o treinamento dos classificadores
    :param labels_for_train_dataset_1: rótulos para treinar o classsificador que identifica se a sentença é de produto
    :param labels_for_train_dataset_2: rótulos para treinar o classsificador que identifica se a sentença é de loja
    :return: os classificadores treinados na seguninte ordem: classificador para produto, classificador para loja
    """
    labels_1 = encoding_labels(labels_for_train_dataset_1, ['0', 'Product'])
    labels_2 = encoding_labels(labels_for_train_dataset_2, ['0', 'Store'])

    svm = LinearSVC()
    svm2 = LinearSVC()

    svm.fit(train_dataset, labels_1)
    svm2.fit(train_dataset, labels_2)

    return svm, svm2


def load_raw_database():
    """
        Método que lê os arquivos csv da base de dados e já devole a base vetorizada com o método TF-IDF
        por definição mas utilizando a variável metoto e mudando para hash o retorno é utilizando o métoro
        de Hash
    :param metodo: variável que define o método estatístico utilizado para extrair as características
    :return: database: base de dados extraído utilizando o método chamado com a variável metodo
             labels: rótulos correspondentes para cada vetor da matriz de base
             vectorizer: instância responsável por transformar os novos inputs em
    """
    import os
    database = []
    labels =[]
    root = "textClassification/SentimentAnalisys/Data"
    for path_to_file in os.listdir(root):
        data, labe = read_file(os.path.join(root, path_to_file))
        database.append(data)
        labels.append(labe)
    database = merge_lists(database)
    labels = merge_lists(labels)
    labels = np.array(labels)
    labels = labels

    return database, labels.tolist()

def tokenize(text):
    """
        Méotodo responsável por realizar os segunintes passos com uma data sentença:
            - tokenize a sentença, ou seja, reparti-la em tokens ou em cada um dos elementos da sentença.
            - remoção de pontuação
            - processo de Stemming, em que a palavra é reduzida a seu radical.
            - a sentença é novamente unida em uma única string já processada.
    :param text: sentença a qual se deseja classificar.
    :return: a senteça processada com apenas radicais de cada token.
    """
    steemming = stem.RSLPStemmer()
    tokens = word_tokenize(smart_unicode(text))
    tokens = [i for i in tokens if i not in string.punctuation]
    stems = [steemming.stem(token) for token in tokens]
    stems = join_strings(stems)
    return stems

def join_strings(list_of_strings):
    """
        Método para transformar tokens em uma única sentença
    :param list_of_strings: Lista com os tokens
    :return: sentença formada pela união dos tokens
    """
    return " ".join(list_of_strings)