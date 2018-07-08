
import io
import numpy as np
import scipy.io as sio



def TFExtractor(present_of_indexing_base, new_product_caption_path, feature_path,indexing_path,database_keyword_path):


    # Read lines of captions from the txt file of new product captions
    with io.open(new_product_caption_path, encoding='utf-8') as file:
        lines = [word for line in file for word in line.split('\t')] #note that each line of words should have a '\n' at the end

    #Important parameters
    number_of_objects = len(lines)  # number of products
    max_number_of_keywords = 20000 #can be set according to real situations; assume no more than 20k new unique keywords in the new products

    # create textual features of new products
    if present_of_indexing_base: #if there exists indexing base

        #store the captions of new products to database by appending to the existing caption file
        with io.open(database_keyword_path, 'a', encoding='utf-8') as file:
            for item in lines:
                file.write('%s' % item)

        #load keyword list and statistics of existing indexing base
        file = open(indexing_path + 'keywordlist.txt', 'r')
        keywordlist = file.read().split()

        keyword_statistics = sio.loadmat(indexing_path + 'keyword_statistics.mat')
        keyword_statistics = keyword_statistics['keyword_statistics']

        number_of_keywords_in_indexing_base = len(keywordlist)
        number_of_keywords = len(keywordlist)

        keyword_statistics = np.concatenate((keyword_statistics, np.zeros((keyword_statistics.shape[0],max_number_of_keywords))), axis=1)
        TF_of_new_products = np.zeros((number_of_objects, number_of_keywords + max_number_of_keywords))

        #scan keywords of new products to create textual features
        # and if having new keywords
        # update indexing base in terms of keywordlist, statistics, Wt,TF.

        for i in range(0, number_of_objects): #process each line of product captions
            print('Processing caption of product %d' % i)
            caption = lines[i].split()

            flag_presence_in_caption = np.zeros((number_of_keywords + max_number_of_keywords))  # 0 indicate the first appearance of the keyword x in this caption

            for x in caption: #for each keyword in the caption of product i
                if x in keywordlist: #if the keyword x is in existing indexing base
                    index_of_keyword = keywordlist.index(x)
                    #update keyword statistics
                    keyword_statistics[0,index_of_keyword] += 1 #the totoal frequency + 1
                    if not flag_presence_in_caption[index_of_keyword]:
                        keyword_statistics[1, index_of_keyword] += 1 #the frequency in different products + 1
                        flag_presence_in_caption[index_of_keyword] = 1

                    # ceate textual features
                    TF_of_new_products[i,index_of_keyword] = 1

                else: #if x is a new keyword
                    keywordlist.append(x)
                    number_of_keywords += 1
                    keyword_statistics[0,number_of_keywords - 1] = 1
                    keyword_statistics[1, number_of_keywords - 1] = 1
                    flag_presence_in_caption[number_of_keywords - 1] = 1

                    TF_of_new_products[i, number_of_keywords - 1] = 1

        #obtain and save the final TF of new products
        TF_of_new_products = TF_of_new_products[:,0:number_of_keywords]
        sio.savemat(feature_path + 'TF_of_new_products.mat', {'TF_of_new_products': TF_of_new_products})

        #update indexing base if new keywords involved

        keyword_statistics = keyword_statistics[:,0:number_of_keywords]
        sio.savemat(indexing_path + 'keyword_statistics.mat', {'keyword_statistics': keyword_statistics})

        if number_of_keywords > number_of_keywords_in_indexing_base:

            with io.open(indexing_path + 'keywordlist.txt', 'w', encoding='utf-8') as file:
                file.write('%s' % keywordlist[0])
                del keywordlist[0]
                for item in keywordlist:
                    file.write(' %s' % item)

            TF_database = sio.loadmat(indexing_path + 'TF_database.mat')
            TF_database = TF_database['TF_database']

            Wt = sio.loadmat(indexing_path + 'Wt.mat')
            Wt = Wt['Wt']


            temp = np.zeros((TF_database.shape[0],number_of_keywords-number_of_keywords_in_indexing_base))
            TF_database = np.concatenate((TF_database, temp), axis=1)
            temp = np.zeros((Wt.shape[0], number_of_keywords - number_of_keywords_in_indexing_base))
            Wt = np.concatenate((Wt, temp), axis=1)

            sio.savemat(indexing_path + 'TF_database.mat', {'TF_database': TF_database})
            sio.savemat(indexing_path + 'Wt.mat', {'Wt': Wt})


    else: #if there is no existing indexing base

        #Create the caption file in database
        with io.open(database_keyword_path, 'w', encoding='utf-8') as file:
            for item in lines:
                file.write('%s' % item)

        #create related data files
        keywordlist = []
        number_of_keywords = 0
        keyword_statistics = np.zeros((2,max_number_of_keywords)) # row 1 for total number of appearance; row 2 for number of products with the word
        number_of_objects = len(lines)
        TF_of_new_products = np.zeros((number_of_objects, max_number_of_keywords))

        #create TF of new products and statistics of keywords in captions
        for i in range(0,number_of_objects):
            print('Processing caption of product %d' % i)
            caption = lines[i].split()

            flag_presence_in_caption = np.zeros((max_number_of_keywords))  # 0 indicate the first appearance of the keyword x in this caption

            for x in caption:
                if x in keywordlist:
                    index_of_keyword = keywordlist.index(x)
                    keyword_statistics[0,index_of_keyword] += 1
                    if not flag_presence_in_caption[index_of_keyword]:
                        keyword_statistics[1, index_of_keyword] += 1
                        flag_presence_in_caption[index_of_keyword] = 1
                    TF_of_new_products[i,index_of_keyword] = 1
                else:
                    keywordlist.append(x)
                    number_of_keywords += 1
                    keyword_statistics[0,number_of_keywords-1] = 1
                    keyword_statistics[1, number_of_keywords - 1] = 1
                    flag_presence_in_caption[number_of_keywords - 1] = 1
                    TF_of_new_products[i,number_of_keywords - 1] = 1



        #save TF, keyword list, keyword statistics
        TF_of_new_products = TF_of_new_products[:, 0:number_of_keywords]
        keyword_statistics = keyword_statistics[:,0:number_of_keywords]
        sio.savemat(feature_path + 'TF_of_new_products.mat', {'TF_of_new_products': TF_of_new_products})
        sio.savemat(indexing_path + 'keyword_statistics.mat', {'keyword_statistics': keyword_statistics})

        with io.open(indexing_path + 'keywordlist.txt', 'w', encoding='utf-8') as file:
            file.write('%s' % keywordlist[0])
            del keywordlist[0]
            for item in keywordlist:
                file.write(' %s' % item)

    return TF_of_new_products