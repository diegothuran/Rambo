# -*- coding: utf-8 -*-
from rest_framework import serializers

from SentimentAnalisys.Ensemble import Ensemble
from SentimentAnalisys.Util import tokenize
from .models import Comment


class CommentSerializer(serializers.ModelSerializer):
    """
    Classe responsável por serializar os cometários passando-os para json
    """
    class Meta:
        model = Comment
        fields = '__all__'


    def create(self, request, *args, **kwargs):
        """
            Método responsável por analisar cada comentário
        :return: A classe para a qual o comentário pertence
        """
        ensemble = Ensemble()
        instance = Comment()
        instance.comment = request.get('comment', instance.comment)
        classification = ensemble.prediction(instance.comment)

        if classification[0] == 1:
            instance.is_product = True
        else:
            instance.is_product = False
        if classification[1] == 1:
            instance.is_store = True
        else:
            instance.is_store = False
        instance.save()

        return instance

class UpdateSerializer(serializers.ModelSerializer):

    #Ensemble().update_classifiers()

    class Meta:
        model = Comment
        fields = '__all__'


