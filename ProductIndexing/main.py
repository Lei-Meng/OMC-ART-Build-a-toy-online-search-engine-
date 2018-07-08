#!/usr/bin/python3
# -*- coding:utf8 -*-


from VFExtractor import VFExtractor
from TFExtractor import TFExtractor
from KeywordFilter import KeywordFilter
from GHFusionART import GHFART
from Backup import Backup
from RemoveFiles import RemoveFiles

import scipy.io as io
import numpy as np
import os


if __name__ == "__main__":



#---------------------------------------Clarification of Pathes-----------------------------------------------------------
    #path to the pre-defined father folder
    root = 'C:\\Users\\lily\\Desktop\\'

    #path to existing database
    database_root = root + 'database\\'
    database_image_path = database_root + 'images\\'
    database_keyword_path = database_root + 'product_keywords.txt'

    #path to data of new products
    new_product_root = root + 'new_products\\'
    new_product_image_path = new_product_root + 'images\\'
    new_product_caption_path = new_product_root + 'product_keywords.txt'

    #path to save the features of new products
    feature_path = root + 'feature_data\\'

    # path to current indexing base
    indexing_path = root + 'indexing_data\\'

    # path to the backup of the indexing base before including new products
    backup_path = root + 'last_time_backup\\'
#--------------------------------------------------------------------------------------------------


#----------------------------------Check and backup before indexing----------------------------------------------------------------

    #check whether there is already an indexing base
    if os.listdir(indexing_path):
        present_of_indexing_base = 1
        Backup(indexing_path,backup_path)
    else:
        present_of_indexing_base = 0

#----------------------------------extract features of new products----------------------------------------------------------------

    #cleanining the feature folder
    RemoveFiles(feature_path)

    VF = VFExtractor(present_of_indexing_base,new_product_image_path,feature_path,indexing_path,database_image_path)
    TF = TFExtractor(present_of_indexing_base, new_product_caption_path, feature_path,indexing_path,database_keyword_path)






    #conduct indexing of new products in existing indexing base
    # indexing_path = 'C:\\Users\\lily\\Desktop\\indexing_data\\'
    # VF = io.loadmat(
    #     feature_path + 'VF.mat')
    # VF = VF['VF']
    #
    # TF = io.loadmat(
    #     feature_path + 'TF.mat')
    # TF = TF['TF']
    #
    GHFART(present_of_indexing_base,VF, TF,indexing_path)






    # #post processing of TF
    # file_path = 'C:\\Users\\lily\\Desktop\\feature_data\\'
    # save_path = 'C:\\Users\\lily\\Desktop\\temp\\'
    # KeywordFilter(file_path,save_path)

    print('end')
