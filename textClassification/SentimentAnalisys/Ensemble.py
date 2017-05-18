# -*- coding: utf-8 -*-
# Author: Diego Santos

#joblib para persistencia do classificador
from sklearn.externals import joblib

from Util import *


def update_classifiers(patter_to_add, label = []):
    """
        Método responsável por atualizar os pesos do classificador utilizando novas entradas não apresentados
    :param patter_to_add: padrão novo a ser adicionado
    :param label: rótulo do novo padrão
    :return: classificadores treinados
    """

    database, labels = load_database()
    database.append(patter_to_add)
    labels.append(label)


class Esenmble():

    def __init__(self):
        self.cls_product, self.cls_store = self.load_classifiers()
        self.database, self.labels, self.vectorizer = load_database()

    def load_classifiers(self):

        svm_product = joblib.load('textClassification/SentimentAnalisys/ClassifierWeigths/svm-02.pkl')
        svm_store = joblib.load('textClassification/SentimentAnalisys/ClassifierWeigths/svm2-02.pkl')

        return svm_product, svm_store

    def prediction(self, pattern):
        pattern = self.vectorizer.transform([pattern])
        pattern = pattern.todense()
        is_product = self.cls_product.predict(pattern)
        is_store = self.cls_store.predict(pattern)

        return [is_product, is_store]

