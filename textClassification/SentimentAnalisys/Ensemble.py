# -*- coding: utf-8 -*-
# Author: Diego Santos

#joblib para persistencia do classificador
from sklearn.externals import joblib
from textClassification.models import Comment

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


class Ensemble():

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

    def update_classifiers(self):
        new_patterns = Comment.objects.filter(is_to_update=True)
        for pattern in new_patterns:
            temp = [pattern.comment]
            label1 = '0'
            label2 = '0'
            if pattern.is_product:
                label1 = 'Product'
            if pattern.is_store:
                label2 = 'Store'
            temp_label =[label1, label2]

            self.database.append(pattern.comment)
            self.labels.append(temp_label)

        for i in range(30):

            extracted_database, self.vectorizer = vectorize_database_tfidf(database=self.database)

            svm1 = LinearSVC()
            svm2 = LinearSVC()

            svm1 = self.train_svm(svm1, extracted_database, self.labels[:, 0])
            svm2 = self.train_svm(svm1, extracted_database, self.labels[:, 1])

    def predict(self, patter, svm1, svm2):
        is_product = svm1.predict(patter)
        is_loja = svm2.predict(patter)

        return [is_product[0], is_loja[0]]

    def test(self, patters, labels):
        labels_1 = encoding_labels(labels[:, 0], ['0', 'Product', 'Pro', 'Prod'])
        labels_2 = encoding_labels(labels[:, 1], ['0', 'Store'])
        acertos = 0
        for i in range(len(patters)):
            a = [labels_1[i], labels_2[i]]
            b = self.predict(patters[i])
            temp = cmp(b, a)
            if temp == 0:
                acertos += 1
        return float(float(acertos)/float(len(patters)))

    def train_svm(self, svm, database, labels):
        extracted_database, vectorizer = vectorize_database_tfidf(database=database)
        X_train, X_test, y_train, y_test = split_database(extracted_database, labels, 0.1)

        svm.fit(X_train, y_train)

        return svm