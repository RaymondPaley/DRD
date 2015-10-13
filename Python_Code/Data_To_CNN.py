# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 20:40:58 2015

@author: tiruviluamala
"""

import os
import glob
import csv

os.chdir('/Users/tiruviluamala/Desktop/DRD')

names = glob.glob("train/*")[0:500]
    
y = [None] * 500
    
    
for index, name in enumerate(names):
    f = open('trainLabels.csv')
    csv_f = csv.reader(f)    
    for row in csv_f:
        if row[0] == names[index].split('/')[1].split('.')[0]:
            y[index] = row[1]
            
y = numpy.asarray(y, dtype='int32')