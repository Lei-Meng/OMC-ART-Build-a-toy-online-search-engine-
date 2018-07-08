

import numpy as np
import os
from ColorMoment import ColorMoment
import scipy.io as sio

def VFExtractor(present_of_indexing_base,new_product_image_path,feature_path,indexing_path,database_image_path):

    # Read all images
    image_files = os.listdir(new_product_image_path)

    # Extract visual features
    number_of_images = len(image_files)
    number_of_visual_features = 225 #here we use color histogram (225D) solely for an example

    VF_of_new_products = np.zeros((number_of_images,number_of_visual_features))

    if present_of_indexing_base: #get the number of current products to save the new products to database
        VF_database = sio.loadmat(indexing_path + 'VF_database.mat')
        VF_database = VF_database['VF_database']
        index_of_image_in_database, _ = VF_database.shape  # get number of current products
    else:
        index_of_image_in_database = 0


    for i in range(0, number_of_images):
        print('Processing image %d' %i)

        # Visual feature extraction
        VF_of_new_products[i, :] = ColorMoment(new_product_image_path + image_files[i])

        #rename and move to database
        index_of_image_in_database += 1
        os.rename(new_product_image_path + image_files[i], database_image_path + str(index_of_image_in_database) + ".jpg")



    # Do normalization

    #Get the max, min of new products
    normalization_of_new_products = np.zeros((2, number_of_visual_features))
    normalization_of_new_products[0, :] = np.min(VF_of_new_products, axis=0)
    normalization_of_new_products[1, :] = np.max(VF_of_new_products, axis=0)

    #a,b,c are variables for normalization
    a = np.zeros((1, number_of_visual_features))
    b = np.zeros((1, number_of_visual_features))
    c = np.zeros((1, number_of_visual_features))

    if present_of_indexing_base: #if there exists an indexing base

       #load related data from indexing base in case the update of visual features
       normalization = sio.loadmat(indexing_path + 'normalization.mat')
       normalization = normalization['normalization']

       #check whether existing max, min bounds of features should change for new products
       min_new = np.minimum(normalization_of_new_products[0, :], normalization[0, :])
       max_new = np.maximum(normalization_of_new_products[1, :], normalization[1, :])

       if sum((normalization[0, :]-min_new)+(max_new-normalization[1, :])) > 0: #if changed

           Wv = sio.loadmat(indexing_path + 'Wv.mat')
           Wv = Wv['Wv']

           #update visual feature values of products and cluster weights in existing indexing base
           #according to equation(12)-(13) in my paper "Online Multimodal Co-indexing and Retrieval of Weakly Labeled Web Image Collections"
           #which are in forms of a*(bx+c): We do not use a/b to avoid the case that the initial max,min are the same
           a[0,:] = 1 / (max_new - min_new)
           b[0,:] = normalization[1, :] - normalization[0, :]
           c[0,:] = (normalization[0, :] - min_new)
           m1 = Wv.shape[0]
           m2 = VF_database.shape[0]
           normalizer_a = np.concatenate([a,a], 1) #note that the dimentionality of Wv and VF_database is doubled in GHF-ART by complement coding
           normalizer_a1 = np.repeat(normalizer_a, m1, axis=0) #for updating cluster weights
           normalizer_a2 = np.repeat(normalizer_a, m2, axis=0) #for updating product features
           normalizer_b = np.concatenate([b, b], 1)
           normalizer_b1 = np.repeat(normalizer_b, m1, axis=0)
           normalizer_b2 = np.repeat(normalizer_b, m2, axis=0)
           normalizer_c = np.concatenate([c, c], 1)
           normalizer_c1 = np.repeat(normalizer_c, m1, axis=0)
           normalizer_c2 = np.repeat(normalizer_c, m2, axis=0)

           #save the updated data to indexing base
           Wv_new = normalizer_a1 * (normalizer_b1 * Wv + normalizer_c1)
           VF_database_new = normalizer_a2 * (normalizer_b2 * VF_database + normalizer_c2)
           sio.savemat(indexing_path + 'Wv.mat', {'Wv': Wv_new})
           sio.savemat(indexing_path + 'VF_database.mat', {'VF_database': VF_database_new})

           normalization_new = np.array([min_new,max_new])
           sio.savemat(indexing_path + 'normalization.mat', {'normalization': normalization_new})

           #normalizing visual features of new products with existing normalization records in indexing base
           a[0,:] = max_new - min_new
           b[0,:] = min_new
           normalizer_a = np.repeat(a, number_of_images, axis=0)
           normalizer_b = np.repeat(b, number_of_images, axis=0)

           VF_of_new_products = (VF_of_new_products - normalizer_b) / normalizer_a

           #save visual features of new products for indexing procedures
           sio.savemat(feature_path + 'VF_of_new_products.mat', {'VF_of_new_products': VF_of_new_products})

       else: #if no change in max min boundaries
           a[0,:] = normalization[1, :] - normalization[0, :]
           b[0,:] = normalization[0,:]
           normalizer_a = np.repeat(a, number_of_images, axis=0)
           normalizer_b = np.repeat(b, number_of_images, axis=0)

           # note that we do the following to avoid the case that the initial max,min are the same
           # any value for such features is fine (see the update equations above for reasons) and we use 0 for convention
           index = np.where( (normalization[1, :] - normalization[0, :]) > 0)
           VF_of_new_products[:,index] = (VF_of_new_products[:,index] - normalizer_b[:,index]) / normalizer_a[:,index]
           index = np.where((normalization[1, :] - normalization[0, :]) == 0)
           VF_of_new_products[:,index] = 0

           # save visual features of new products for indexing procedures
           sio.savemat(feature_path + 'VF_of_new_products.mat', {'VF_of_new_products': VF_of_new_products})

    else: #if no existing indexing base, use the max,min of new products for normalization

        a[0,:] = normalization_of_new_products[1,:] - normalization_of_new_products[0,:]
        b[0,:] = normalization_of_new_products[0,:]
        normalizer_a = np.repeat(a, number_of_images, axis=0)
        normalizer_b = np.repeat(b, number_of_images, axis=0)

        index = np.where(normalization_of_new_products[1,:] - normalization_of_new_products[0,:] > 0)
        VF_of_new_products[:,index] = (VF_of_new_products[:,index] - normalizer_b[:,index]) / normalizer_a[:,index]
        index = np.where(normalization_of_new_products[1,:] - normalization_of_new_products[0,:] == 0)
        VF_of_new_products[:,index] = 0

        # save normalization and visual features of new products for indexing procedures
        sio.savemat(indexing_path + 'normalization.mat', {'normalization': normalization_of_new_products})
        sio.savemat(feature_path + 'VF_of_new_products.mat', {'VF_of_new_products': VF_of_new_products})

    return VF_of_new_products