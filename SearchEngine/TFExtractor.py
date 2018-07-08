# -*- coding: utf-8 -*-

import numpy as np


def TFExtractor(text,keywordlist):
    '''
    % text: a vector of input keywords
    % keywordlist: a vector of size 1*m, storing keywords identified from images in the database

    % tVec: the feature vector generated from text by matching the keyword list of the data base
    '''

    m = len(text)
    n = len(keywordlist)

    tVec = np.zeros((n))

    for i in range(0, m):
        for j in range(0,n):
            if text[i] == keywordlist[j]:
                tVec[j] = 1
                break




    return tVec