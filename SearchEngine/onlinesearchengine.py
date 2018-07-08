# -*- coding: utf-8 -*-

import numpy as np
import scipy.io as io
from TFExtractor import TFExtractor
from VFExtractor import VFExtractor
from insertionranking import insertionranking


def onlinesearchengine(direc,text,k,root_files):



    flag_v=1
    flag_t=1

    #check for any special cases. Also, such checks could be done before calling this function
    file = open(root_files + 'keywordlist.txt', 'r')
    keywordlist = file.read().split()

    # count_index = -1
    # for x in keywordlist:
    #     count_index += 1
    #     if 'vitamin' in x:
    #         checkpoint = 1



    if len(text)>0: #indicating having input keywords
        Qt = TFExtractor(text,keywordlist)

        # check_sum = sum(Qt)
        # le = Qt.size
        # for i in range(0,le):
        #     if Qt[i] == 1:
        #         check_index = 1

        if sum(Qt)==0: #if no matching keywords in the database
             print('0:No relevant products found...')
             return

    else: #%if no input keywords
        Qt= 0 #np.zeros((len(keywordlist)))
        flag_t=0


    if len(direc)==0: #%if no input image, do following checks
        flag_v=0
        if flag_t==0: #%if no input keywords, indicating no input received
             print('1:Input image and/or keywords are required!')
             return


    #%build visual features
    normalization = io.loadmat(root_files + 'normalization.mat')
    normalization = normalization['normalization']
    if flag_v == 1:
        Qv = VFExtractor(direc,normalization)
        Qv = np.concatenate([Qv, 1 - Qv], 1)
    else:
        Qv= 0 #np.zeros((normalization.shape[1])) #%the colume number of normalization here is the dimension of visual features

    #%call searching function

       # %load necessary information
    J = io.loadmat(
        root_files + 'J.mat')
    J = J['J']
    J = J[0, 0]

    Wv = io.loadmat(
       root_files + 'Wv.mat')
    Wv = Wv['Wv']

    Wt = io.loadmat(
       root_files + 'Wt.mat')
    Wt = Wt['Wt']

    L = io.loadmat(
       root_files + 'L.mat')
    L = L['L']

    kVF = io.loadmat(
       root_files + 'kVF.mat')
    kVF = kVF['kVF']

    kTF = io.loadmat(
       root_files + 'kTF.mat')
    kTF = kTF['kTF']

    VF = io.loadmat(
       root_files + 'VF.mat')
    VF = VF['VF']

    TF = io.loadmat(
       root_files + 'TF.mat')
    TF = TF['TF']

    indexbase = io.loadmat(
       root_files + 'indexbase.mat')
    indexbase = indexbase['indexbase']



    index,simlist = insertionranking(Qv,Qt,flag_v,flag_t,J,Wv,Wt,L,kVF,kTF,VF,TF,indexbase,k)


    #%filter the results

    # if flag_v == 0:
    #     Filter = 0.5
    # elif flag_t ==0:#if only text available, products with similarity of 0.5 are deemed to be relevant
    #     Filter = 0.85
    # else:
    #     Filter = 0.8
    #
    #
    # for i in range(0,len(index)):
    #     if simlist[i]<Filter:
    #         index=index[0:i]
    #         simlist=simlist[0:i]
    #         break


    #output returned results

    if len(index)==0:
        print('2:No relevant products found...')
        return

    return index, simlist
