# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models

class Comment(models.Model):
    comment = models.TextField(verbose_name="Review")
    is_product = models.BooleanField(blank=True, verbose_name="É de produto?")
    is_store = models.BooleanField(blank=True, verbose_name="É de Loja?")
    is_correct = models.BooleanField(blank=True, default=True, verbose_name="Está correto?")
    is_to_update = models.BooleanField(blank=True, default=False, verbose_name="Utilizar na atualização?")

    #def __str__(self):
    #    return u'Model: %s' % self.comment

    def __unicode__(self):
        return u'Model: %s' % self.comment
