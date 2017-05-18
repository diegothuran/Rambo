# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models

class Comment(models.Model):
    comment = models.CharField(max_length=250)
    is_product = models.BooleanField(blank=True)
    is_store = models.BooleanField(blank=True)
    is_correct = models.BooleanField(blank=True, default=True)
    is_to_update = models.BooleanField(blank=True, default=False)

    def __str__(self):
        return self.comment[:40]
