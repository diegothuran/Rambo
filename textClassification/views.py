# -*- coding: utf-8 -*-
from rest_framework import generics
from rest_framework.views import APIView
from rest_framework.reverse import reverse
from rest_framework.response import Response
from .models import Comment
from .serializers import CommentSerializer, UpdateSerializer
from textClassification.SentimentAnalisys import Ensemble


class CommentListView(generics.ListCreateAPIView):
    queryset = Comment.objects.all()
    serializer_class = CommentSerializer

class CommentDetail(generics.RetrieveUpdateDestroyAPIView):
    queryset = Comment.objects.all()
    serializer_class = CommentSerializer

class UpdateListView(generics.ListCreateAPIView):
    queryset = Comment.objects.all()
    serializer_class = UpdateSerializer