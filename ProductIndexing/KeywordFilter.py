
import numpy as np
import scipy.io as io
import io as fileio


def KeywordFilter(root_index,save_index):

#the following codes are for a revision of the indexing data


     file = open(root_index + 'keyword_statistics.txt', 'r', encoding="utf8")
     keywordstatistics = file.read().split()
     keywordstatistics = keywordstatistics[1:]
     keywordstatistics = list(map(int, keywordstatistics))

     file = open(root_index + 'keywordlist.txt', 'r', encoding="utf8")
     keywordlist = file.read().split()
     number_of_keywords = len(keywordlist)

     newkeywordlist = []

     for x in range(0,number_of_keywords):
         if  keywordstatistics[x] > 5:
             newkeywordlist.append(keywordlist[x])

     mask = np.asarray(keywordstatistics)


     TF = io.loadmat(root_index + 'TF.mat')
     TF = TF['TF']
     m,n = TF.shape
     new_TF = np.zeros((m,len(newkeywordlist)))
     for x in range(0,m):
        temp = TF[x,:]
        temp = temp[ np.where(mask > 5)]
        new_TF[x,:] = temp

     TF = new_TF


     io.savemat(save_index + 'TF.mat', {'TF': TF})



     keyword_path = save_index + 'keywordlist.txt'
     with fileio.open(keyword_path, 'w', encoding='utf-8') as file:
         file.write('%s' % newkeywordlist[0])
         del newkeywordlist[0]
         for item in newkeywordlist:
             file.write(' %s' % item)


     keyword_statistics_path = save_index + 'keyword_statistics.txt'
     newkeywordstatistics = mask[ np.where(mask > 5)]
     with fileio.open(keyword_statistics_path, 'w', encoding='utf-8') as file:
        file.write('%d' % (len(newkeywordlist)+1))
        for number in newkeywordstatistics:
            file.write(' %d' % number)












     # Wt = io.loadmat(root_index + 'Wt.mat')
     # Wt = Wt['Wt']
     # m,n = Wt.shape
     # new_Wt = np.zeros((m,n))
     # for x in range(0,m):
     #     temp = Wt[x,:]
     #     temp = temp[ np.where(mask > 1)]
     #     new_Wt[x,:] = temp
     # Wt = new_Wt
     #
     #
     #
     # kTF = io.loadmat(root_index + 'kTF.mat')
     # kTF = kTF['kTF']
     # new_kTF = np.zeros((m,n))
     # for x in range(0,m):
     #     temp = kTF[x,:]
     #     temp = temp[ np.where(mask > 1)]
     #     new_kTF[x,:] = temp
     # kTF = new_kTF
     #
     #
     #
     #
     # TF = io.loadmat(root_index + 'TF.mat')
     # TF = TF['TF']
     # m,n = TF.shape
     # new_TF = np.zeros((m,n))
     # for x in range(0,m):
     #     temp = TF[x,:]
     #     temp = temp[ np.where(mask > 1)]
     #     new_TF[x,:] = temp
     # TF = new_TF


     #
     # io.savemat(save_index + 'Wt.mat', {'Wt': Wt})
     # io.savemat(save_index + 'kTF.mat', {'kTF': kTF})
     # io.savemat(save_index + 'TF.mat', {'TF': TF})

