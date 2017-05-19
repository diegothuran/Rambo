from django.conf.urls import include, url
from django.contrib import admin
from . import views

app_name = 'sentiment'

urlpatterns = [
    url(r'^comments/$', views.CommentListView.as_view()),
    url(r'^comments/(?P<pk>[0-9]+)/$', views.CommentDetail.as_view()),
]