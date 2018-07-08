# -*- coding: utf-8 -*-

import numpy as np

def insertionranking(Qv,Qt,flag_v,flag_t,J,Wv,Wt,L,kVF,kTF,VF,TF,indexbase,k):
    '''
    %--------------------------------------------------------------------------
    % Qv: visual feature vector of the query of size 1*225
    % Qt: textual feature vector of the query of size 1*m (m is the number of keywords)
    % J: number of clusters
    % Wv: weight vectors of visual features for J clusters of size J*255
    % Wt: weight vectors of textual features for J clusters of size J*m
    % kVF: a vector of boolean values of size J*255, where 1 indicates the corresponding visual feature is a key feature, and 0 otherwise.
    % kTF: a vector of size J*m, figuring out the key textual features
    % L: a vector of size J*1 storing the number of data objects in each of the J clusters

    % VF,TF: the respective visual and textual feature vectors of the indexed images in the indexing structure of size n*225 and n*m (n is the number of indexed images)
    % indexbase: a vector of size n*1, storing the indexes in the data base of images ordered in VF and TF
    % Note that images are ordered according to the clusters they assigned to.
    % E.g. 1~L(1) in the database store the images belonging to the first cluster



    % k: the length of the rank list, i.e. k images are supposed to be returned as the search result
    % index: the index of the image at each rank position in the data set
    % simlist: similarities of images to the query
    %--------------------------------------------------------------------------
    '''







    #%-------------initialization-----------------------------------------------------

    m,n=VF.shape
    if k>m: #m is the total number of indexed images
       k=m


    simlist = np.zeros((k)) # similarities of the k images most similar to the query
    simlist=simlist-1

    index = np.zeros((k)) #%store the indexes of images in the database at given ranking positions

    if flag_v + flag_t == 2:
        rv = 0.1  # the weight of visual features
        rt = 1 - rv  # the weight of textual features
    else:
        rv = flag_v
        rt = flag_t

    #%-------------initialization-----------------------------------------------------



    #%---------obtain the descending order of clusters according to the similarities to the query -----------------

    simc=np.zeros((J)) #%store similarities of clusters to the query
    simc=simc-1  #%here -1 is used to avoid the case when sim =0, which incurs an error in ranking


    indexc=np.zeros((J))  #%store indexes of clusters at the given rank position


    for i in range(0, J): #%similarity calculation and ranking
        numerator_V =0
        numerator_T =0

        Mj_V=0 #% similarity for visual features
        Mj_T=0 #% similarity for textual features

        if flag_v == 1:
            m,n=Wv.shape
            for j in range(0, n): #currently we disable the key feature selection mode because the dataset is too small to select effective key features
                 #numerator_V = numerator_V + kVF[i,j]*min(Qv[j],Wv[i,j])   %kVF indicates counting the key features only
                numerator_V = numerator_V + min(Qv[j],Wv[i,j])

            denominator = 0
            for j in range(0, n):
                # denominator = denominator + kVF[i,j]*Qv[j]
                denominator = denominator + Qv[j]

            Mj_V = numerator_V/denominator#% similarity for visual features


        if flag_t == 1:
            m,n=Wt.shape
            for j in range(0, n):
                # numerator_T = numerator_T + kTF[i,j]*min(Qt[j],Wt[i,j])
                 numerator_T = numerator_T + min(Qt[j],Wt[i,j])

            denominator1 = 0
            denominator2 = 0
            for j in range(0, n):
                # denominator = denominator + kTF[i,j]*Qt[j]
                 denominator1 = denominator1 + Qt[j]
                 denominator2 = denominator2 + Wt[i,j]

            Mj_T =  0.9 * numerator_T/(0.00001+denominator1)+0.1 * numerator_T/(0.00001+denominator2) #% similarity for textual features, consider both hit of query and number of words in product

        #%overall similarity of the ith cluster to the query
        if flag_v == 0:
           sim = Mj_T
        elif flag_t ==0:
            sim = Mj_V
        else:
             sim = rv*Mj_V + rt*Mj_T

        #%ranking
        for j in range(0, J):
            if sim > simc[j]:
                for l in range(1,J-j):
                    simc[J-l]=simc[J-l-1]
                    indexc[J-l]=indexc[J-l-1]

                simc[j]=sim
                indexc[j]=i
                break

    #%---------obtain the descending order of clusters according to the similarities to the query -----------------



    #%---------obtain the rank list of images to the query--------------------------------------------

    m,n=VF.shape
    count = 0 #%count the number of continuous data objects that cannot be included into the rank list
    limit = min(2*k,m) #% if the rank list remain unchanged for the following 'limit' data objects, the ranking algorithm stops
    flag = 0 #%used to jump out two loops to end the ranking algorithm if limit is reached

    for i in range(0,J):
         if flag == 1:
              break

         m=0 #%index of the first image in the database of the given cluster
         n=0 #%index of the last image in the database of the given cluster


         for j in range(0,int(indexc[i])+1): #sum number of images in clusters in front the given cluster to obtain the index of last image of the cluster

             n=n+int(L[0,j])

         m=n-int(L[0,int(indexc[i])])+1

         m=m-1 #index of python arrays starts from 0
         n=n-1

         for p in range(m,n+1): #%calculate the similarities of images of the indexc(i)-th cluster  to the query and insert into the ranking list
            numerator_V =0
            numerator_T =0

            Mj_V=0 #% similarity for visual features
            Mj_T=0 #% similarity for textual features

            if flag_v == 1:
                mm,nn=Wv.shape
                for j in range(0, nn): #currently we disable the key feature selection mode because the dataset is too small to select effective key features
                 #numerator_V = numerator_V + kVF[i,j]*min(Qv[j],Wv[i,j])   %kVF indicates counting the key features only
                    numerator_V = numerator_V + min(Qv[j],VF[p,j])

                denominator = 0
                for j in range(0, nn):
                    # denominator = denominator + kVF[i,j]*Qv[j]
                    denominator = denominator + Qv[j]

                Mj_V = numerator_V/denominator#% similarity for visual features


            if flag_t == 1:
                mm,nn=Wt.shape
                for j in range(0, nn):
                    # numerator_T = numerator_T + kTF[i,j]*min(Qt[j],Wt[i,j])
                    numerator_T = numerator_T + min(Qt[j],TF[p,j])

                denominator1 = 0
                denominator2 = 0
                for j in range(0, nn):
                    # denominator = denominator + kTF[i,j]*Qt[j]
                    denominator1 = denominator1 + Qt[j]
                    denominator2 = denominator2 + TF[p,j]

                Mj_T = 0.9 * numerator_T/(0.00001+denominator1)+0.1 * numerator_T/(0.00001+denominator2) #% similarity for textual features

            #%overall similarity of the ith cluster to the query
            if flag_v == 0:
                sim = Mj_T
            elif flag_t ==0:
                sim = Mj_V
            else:
                sim = rv*Mj_V + rt*Mj_T


            #%find the ranking position of the p-th image to the query
            if sim<simlist[k-1]:
                count = count +1

    #%-------------------------------------------------------------------------
    #            if count >=limit:         %Note that the algorithm cannot effectively learn the key features from small data sets (<10000 products)
    #                flag = 1            %Because the similarity between query and clusters are not well estimated.
    #                break               %In this case, go through the whole data set instead of stop checking the similarity of products in the clusters that are deemed as low relevant clusters
    #
    #%-------------------------------------------------------------------------

            else:
                count = 0 #%if the ranking list is changed, clear the count
                for j in range(0,k):
                    if sim > simlist[j]:
                        for l in range(1,k-j):
                            simlist[k-l]=simlist[k-l-1]
                            index[k-l]=index[k-l-1]

                        simlist[j]=sim
                        index[j]=indexbase[0,p]
                        break



    #%---------obtain the rank list of images to the query--------------------------------------------


    return index,simlist

