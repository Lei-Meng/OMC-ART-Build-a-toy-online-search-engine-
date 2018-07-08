
import scipy.io as io
import numpy as np

def GHFART(present_of_indexing_base,M,N,indexing_path):
    '''
    % M: numpy arrary; m*n1 matrix of visual features; m is number of objects and n1 is number of visual features
    % N: numpy array; m*n2 matrix of textual features; n is the number of words
    '''
#-----------------------------------------------------------------------------------------------------------------------
# Input parameters
    alpha = 0.01 # no need to tune; used in choice function; to avoid too small cluster weights (resulted by the learning method of ART; should be addressed sometime); give priority to choosing denser clusters
    beta = 0.6 # has no significant impact on performance with a moderate value of [0.4,0.7]

    # the two rhos need carefully tune; used to shape the inter-cluster similarity; rho_v = 0.7 indicates an object will not be clustered to a cluster with visual similarity lower than 0.7
    rho_v = 0.6
    rho_t = 0.01

# Input parameters
#-----------------------------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------------------------------
# Initialization

    #complement coding
    M = np.concatenate([M,1-M], 1)
    #Note that no complement coding for textual features N


    #get data sizes
    row, colV = M.shape
    _, colT = N.shape #note row, i.e. the number of objects, is the same for the two matrices


# Initialization
# -----------------------------------------------------------------------------------------------------------------------



# -----------------------------------------------------------------------------------------------------------------------
# Clustering process

    print("algorithm starts")

    # load cluster settings
    if present_of_indexing_base:  # if indexing base exists, load them

        #load the features of products in database for generating VF/TF in cluster order for search engine
        VF_database = io.loadmat(indexing_path + 'VF_database.mat')
        VF_database = VF_database['VF_database']
        current_number_of_products = VF_database.shape[0]
        VF_database = np.concatenate([VF_database, M], axis=0)

        TF_database = io.loadmat(indexing_path + 'TF_database.mat')
        TF_database = TF_database['TF_database']
        TF_database = np.concatenate([TF_database, N], axis=0)

        #load parameters for clustering
        J = io.loadmat(
            indexing_path + 'J.mat')
        J = J['J']
        J = J[0, 0]

        Wv = io.loadmat(
            indexing_path + 'Wv.mat')
        Wv = Wv['Wv']
        Wv = np.concatenate([Wv,np.zeros((row,colV))], axis=0) #The maximum number of clusters increased equals to number of new products


        Wt = io.loadmat(
            indexing_path + 'Wt.mat')
        Wt = Wt['Wt']
        Wt = np.concatenate([Wt, np.zeros((row, colT))], axis=0)

        L = io.loadmat(
            indexing_path + 'L.mat')
        L = L['L']
        L = np.concatenate([L, np.zeros((1,row))], axis=1)

        Assign = io.loadmat(
            indexing_path + 'Assign.mat')
        Assign = Assign['Assign']
        Assign = np.concatenate([Assign, np.zeros((1,row))],axis=1)

        gamma = io.loadmat(indexing_path + 'gamma.mat') # No need to tune. Gamma will be self-adapted and no need to tune; it is the weights for fusing visual and textual similarities
        gamma = gamma['gamma']
        gamma = gamma[0, 0]


        start_product = 0 #Process from the first new product in the for loop below
        start_assign = current_number_of_products # record the assignment of new products in Assign

    else: # if not products yet, create initial cluster with the first product of new products

        #save features of new products as feature data of products in current database
        VF_database = M
        TF_database = N

        #initialize other parameters
        Wv = np.zeros((row, colV))
        Wt = np.zeros((row, colT))
        J = 0  # number of clusters
        L = np.zeros((1,row))  # note the maximun number of cluster is row; record the sizes of clusters for the learning of textual features

        gamma = 0.5  # No need to tune. Gamma will be self-adapted and no need to tune; it is the weights for fusing visual and textual similarities

        # record cluster assignment of objects in database
        Assign = np.zeros((1,row), dtype=np.int)  # the cluster assignment of objects

        #first cluster
        print('Processing product 0')
        Wv[0, :] = M[0, :]
        Wt[0, :] = N[0, :]
        J = 1
        Assign[0,0] = J-1 #note that python array index trickily starts from 0
        L[0,J-1] = 1

        start_product = 1 #Process from the first new product in the for loop below
        start_assign = 0



    # for update of gamma
    # note that these are also the initialization
    max_number_of_clusters = Wv.shape[0]
    Difference_V = np.zeros(max_number_of_clusters)
    Difference_T = np.zeros(max_number_of_clusters)
    AvgDif_V = 0
    AvgDif_T = 0
    R_V = 1 # robustness of  visual features
    R_T = 1 # robustness of  textual features


    #processing other objects
    for n in range(start_product,row-start_product):

        print('Processing product %d' % n)

        T_max = -1 #the maximun choice value
        winner = -1 #index of the winner cluster

        #compute the similarity with all clusters; find the best-matching cluster
        for j in range(0,J):

            #compute the match function
            Mj_numerator_V = 0
            Mj_numerator_T = 0

            for i in range(0,colV):
                Mj_numerator_V = Mj_numerator_V + min(M[n, i], Wv[j, i])

            for i in range(0,colT):
                Mj_numerator_T = Mj_numerator_T + min(N[n, i], Wt[j, i])

            Mj_V = Mj_numerator_V / sum(M[n,:])
            Mj_T = Mj_numerator_T / (0.00001 + sum(N[n, :])) #note that the addition of 0.00001 is for possible practical engineering requirement
                                                            # can avoid the case when all words of a product are filtered in some procedures

            if Mj_V >= rho_v and Mj_T >= rho_t:
                #compute choice function
                Tj = (1-gamma) * Mj_numerator_V / (alpha + sum(Wv[j, :])) + gamma * Mj_numerator_T / (alpha + sum(Wt[j,:]))

                if Tj > T_max:
                    T_max = Tj
                    winner = j


        #Cluster assignment process
        if winner == -1: #indicates no cluster passes the vigilance parameter - the rhos
            #create a new cluster
            J = J + 1
            Wv[J - 1, :] = M[n, :]
            Wt[J - 1, :] = N[n, :]
            Assign[0,start_assign+n] = J - 1
            L[0,J - 1] = 1

            #update vigilance parameter gamma
            gamma = ((R_T) ** (J / (J + 1))) / ((R_T) ** (J / (J + 1)) + (R_V) ** (J / (J + 1))) # gamma is for textual features and that for visual feature is 1-gamma

        else: #if winner is found, do cluster assignment and update cluster weights and gamma

            #variables for computing gamma
            Wv_old = np.copy(Wv[winner, :])
            Wt_old = np.copy(Wt[winner, :])
            Pattern_V = M[n, :]
            Pattern_T = N[n, :]

            #update cluster weights
            for i in range(0, colV):
                Wv[winner, i] = beta * min(Wv[winner, i], M[n, i]) + (1 - beta) * Wv[winner, i]

            for i in range(0, colT):
                Wt[winner, i] = L[0,winner] / (L[0,winner] + 1) * (Wt[winner, i] + N[n, i] / L[0,winner])

            #cluster assignment
            Assign[0,start_assign+n] = winner
            L[0,winner] += 1

            #update gamma
            Wv_now = Wv[winner, :]
            Wt_now = Wt[winner, :]

            NewDif_V = (L[0,winner] - 1) / L[0,winner] / sum(Wv_now) * (sum(Wv_old) * Difference_V[winner] + sum(abs(Wv_old - Wv_now)) + 1 / (L[0,winner]-1) * sum(abs(Wv_now-Pattern_V)))
            AvgDif_V = AvgDif_V + (NewDif_V - Difference_V[winner]) / J
            Difference_V[winner] = NewDif_V
            R_V = np.exp(-AvgDif_V)

            NewDif_T = (L[0,winner] - 1) / L[0,winner] / sum(Wt_now) * (sum(Wt_old) * Difference_T[winner] + sum(abs(Wt_now - ((L[0,winner] - 1) / L[0,winner]) * Wt_old)) + 1 / (L[0,winner] - 1) * sum(abs(Wt_now - Pattern_T)))
            AvgDif_T = AvgDif_T + (NewDif_T - Difference_T[winner]) / J
            Difference_T[winner] = NewDif_T
            R_T = np.exp(-AvgDif_T)

            gamma = R_T / (R_V + R_T)

# Clustering process
# -----------------------------------------------------------------------------------------------------------------------

    print("algorithm ends")

    #Clean indexing data
    Wv = Wv[0 : J, :]
    Wt = Wt[0 : J, :]
    L = L[:,0 : J]

    #construction of kVF, kTF, VF, TF, indexbase

    #kVF and kTF
    kVF = np.zeros((J, colV))
    kTF = np.zeros((J, colT))
    meanv = np.zeros(J)
    meant = np.zeros(J)

    for i in range(0,J):
        meanv[i] = np.mean(Wv[i, :])
        meant[i] = np.mean(Wt[i, :])

    for i in range(0,J):
        for j in range(0,colV):
            if Wv[i,j] > meanv[i]:
                kVF[i,j] = 1

        for j in range(0,colT):
            if Wt[i,j] > meant[i]:
                kTF[i,j] = 1


    # Build indexbase to arrange products in terms of clusters
    number_of_products = VF_database.shape[0]
    indexbase = np.zeros(number_of_products, dtype= np.int) #link between product position in database and in our indexing base
    count = np.zeros(J, dtype= np.int) #to record the number of arranged products in each cluster



    for i in range(0, number_of_products): #scan the products in database
        m = 0 # store the i-th product to m-th postion in indexbase
        for j in range(0,int(Assign[0,i])):
            m += int(L[0,j])
        m = m + int(count[int(Assign[0,i])])
        indexbase[m] = i
        count[int(Assign[0,i])] += 1


    # Organize VF and TF features of products in database in the order of clusters for search engine
    VF = np.zeros((number_of_products, colV))
    TF = np.zeros((number_of_products, colT))
    for i in range(0,number_of_products):
        VF[i,:]=VF_database[indexbase[i], :]
        TF[i,:]=TF_database[indexbase[i], :]


    # Store indexing base structure files

    io.savemat(indexing_path + 'VF_database.mat', {'VF_database': VF_database})
    io.savemat(indexing_path + 'TF_database.mat', {'TF_database': TF_database})

    io.savemat(indexing_path + 'J.mat', {'J': J})
    io.savemat(indexing_path + 'gamma.mat', {'gamma': gamma})
    io.savemat(indexing_path + 'Wv.mat', {'Wv': Wv})
    io.savemat(indexing_path + 'Wt.mat', {'Wt': Wt})
    io.savemat(indexing_path + 'L.mat', {'L': L})
    io.savemat(indexing_path + 'kVF.mat', {'kVF': kVF})
    io.savemat(indexing_path + 'kTF.mat', {'kTF': kTF})
    io.savemat(indexing_path + 'VF.mat', {'VF': VF})
    io.savemat(indexing_path + 'TF.mat', {'TF': TF})
    io.savemat(indexing_path + 'indexbase.mat', {'indexbase': indexbase})
    io.savemat(indexing_path + 'Assign.mat', {'Assign': Assign})


    return 0

