def cramer_rao(A, signal, angles, locations):
    
    import numpy as np
    
    noise = 1
    N_sample = signal.shape[1]
    S = (signal.dot(signal.conj().T))/N_sample
    P_A = np.identity(12) - A.dot(np.linalg.pinv(A.conj().T.dot(A))).dot(A.conj().T)
    U = S.dot(np.linalg.pinv((A.conj().T).dot(A).dot(S) + (noise)*np.identity(3))).dot((A.conj().T).dot(A).dot(S))

    # Direction of arrivals
    K_l = np.array([np.sin(angles), np.cos(angles)])

    der_V = [np.zeros((12,1), dtype=complex),np.zeros((12,1),dtype=complex),np.zeros((12,1),dtype=complex)]
    V = [np.zeros((12,6), dtype=complex),np.zeros((12,6),dtype=complex),np.zeros((12,6),dtype=complex)]

    #for i in range(0,len(angles)):    
    #    for j in range(0,len(locations)):
    #        first_part = np.exp(1j*np.pi*locations[j].dot(K_l[:,i]))
    #        coef = locations[j][1]*K_l[:,i][1]-locations[j][0]*K_l[:,i][0]
    #        second_part = np.array([np.pi*1j*coef, np.exp(1j*np.pi*1*np.sin(angles[i]))*(coef*np.pi*1j+np.pi*K_l[:,i][1])])
    #        A_k = np.transpose(np.array(first_part*second_part, ndmin=2))
    #        der_V[i][2*j:2*j+2, j:j+1] = A_k[0:2]

    #       second_part = np.array([1, np.exp(1j*np.pi*1*np.sin(angles[i]))])
    #       A_k = np.transpose(np.array(first_part*second_part, ndmin=2))
    #       V[i][2*j:2*j+2, j:j+1] = A_k[0:2]
     
    
    for i in range(0,len(angles)):    
        for j in range(0,len(locations)):
            first_part = np.exp(1j*np.pi*locations[j].dot(K_l[:,i]))
            coef = locations[j][1]*K_l[:,i][1]-locations[j][0]*K_l[:,i][0]
            #second_part = np.array([np.pi*1j*coef, np.exp(1j*np.pi*1*np.sin(angles[i]))*(coef*np.pi*1j+np.pi*K_l[:,i][1])])
            second_part = np.array([0, np.pi*j*np.cos(angles[i])*np.exp(j*np.pi*np.sin(angles[i]))])
            A_k = np.transpose(np.array(first_part*second_part, ndmin=2))
            der_V[i][2*j:2*j+2, 0:1] = A_k[0:2]

            second_part = np.array([1, np.exp(1j*np.pi*1*np.sin(angles[i]))])
            V_k = np.transpose(np.array(second_part, ndmin=2))
            V[i][2*j:2*j+2, j:j+1] = V_k[0:2]
    
    D_theta = np.squeeze(np.array([der_V[0],der_V[1],der_V[2]])).T
    D_xi = []
    D_zheta = []

    for i in range (0,len(locations)):
        e_k = np.zeros((6,1))
        e_k[i] = 1
        D_xi.append(np.squeeze(np.array([V[0].dot(e_k),V[1].dot(e_k),V[2].dot(e_k)])))
        D_zheta.append(1j*D_xi[i])

    D_xi = np.concatenate([D_xi[i] for i in range(5)]).T
    D_zheta = np.concatenate([D_zheta[i] for i in range(5)]).T
    D = np.concatenate((np.concatenate((D_xi,D_zheta), axis = 1), D_theta), axis=1)
    
    # shape should be 33*33 of both of them
    first_matrix = np.kron(np.ones((11,1)).dot(np.ones((1,11))), U)
    second_matrix = (D.conj().T.dot(P_A).dot(D)).T
    cramer_rao = (1/(2*N_sample))*np.mean(np.diag(1/(np.real(np.multiply(first_matrix,second_matrix))))[0:3])
    
    return cramer_rao
    
    
