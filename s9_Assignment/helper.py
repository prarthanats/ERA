# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 15:16:34 2023
@author: prarthana.ts
"""

from torchsummary import summary

def model_summary(model, input_size):
    summary(model, input_size)