# -*- coding: utf-8 -*-
from django.shortcuts import render
from rest_framework import status
from rest_framework import generics
from django.contrib.auth.models import User
from rest_framework.response import Response
from .models import Comment
from .serializers import CommentSerializer


class CommentListView(generics.ListCreateAPIView):
    queryset = Comment.objects.all()
    serializer_class = CommentSerializer

class CommentDetail(generics.RetrieveUpdateDestroyAPIView):
    queryset = Comment.objects.all()
    serializer_class = CommentSerializer