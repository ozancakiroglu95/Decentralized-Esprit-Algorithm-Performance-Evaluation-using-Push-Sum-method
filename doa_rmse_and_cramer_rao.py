def rmse_and_cramer_rao(SNR_range, N_samples_range, iteration, A, angles, locations, K, method_code, return_name):    
    
    """ return rmse or cramer rao according to number of signal snapshots and received signal SNR

    Parameters
    ----------
    SNR_range: tuple (x, y)
        SNR first and last value. Give x=y if you want results according to N_samples_range
    N_samples_range: tuple (a, b)
        Number of samples first and last value. Give a=b if you want results according to SNR_range
    iteration: int 
        Iteration number of Average Consensus or Push-Sum algorithm
    method_code: integer
        "1" is average consensus d-Esprit, "2" is push-sum d-Esprit, "3" conventional Esprit
    return_name: string 
        "rmse" returns rmse, "cramer" returns cramer
        
    Returns
    -------
    rmse_or_cramer: numpy array
        returns cramer rao or rmse with the length specified in SNR_range or N_samples_range
    """
    
    import numpy as np
    from stochastic_cramer_rao import cramer_rao
    
    N_samples_zero = N_samples_range[0]
    SNR_zero = SNR_range[0]
    
    if SNR_range[1] == SNR_range[0]+1:
        snr_dB = SNR_range[1]
        if return_name == "rmse":
            MSE = np.zeros(N_samples_range[1]-N_samples_range[0])
        elif return_name == "cramer":
            cramer = np.zeros(N_samples_range[1]-N_samples_range[0])

    elif N_samples_range[1] == N_samples_range[0]+1:
        N_samples = N_samples_range[1]
        if return_name == "rmse":
            MSE = np.zeros(SNR_range[1]-SNR_range[0])
        elif return_name == "cramer":
            cramer = np.zeros(SNR_range[1]-SNR_range[0])
        
    for snr_dB in range(SNR_range[0],SNR_range[1]):
        
        for N_samples in range(N_samples_range[0], N_samples_range[1]):
        
            for i in range(500):

                # Signal(A*s) to noise(n) ratio
                received_snr = 10**(snr_dB/20)
                ratio_As_to_s = 1/4
                snr = received_snr*ratio_As_to_s

                # Source signal implementation (shape: (3,500))
                signal = np.random.normal(0,np.sqrt(snr),(3,N_samples))
                #w = np.atleast_2d([np.pi/3, np.pi/4, np.pi/5]).T
                #signal = (np.sqrt(snr))*np.exp(1j*w*(np.atleast_2d(np.arange(1,N_samples+1))))

                # Received signal power on sensors
                signal_power = sum(sum(np.abs(A.dot(signal))**2))/(12*N_samples)

                # Noise signal implementation (shape: (12,500))
                noise = np.random.normal(0,np.sqrt(0.5),(12,N_samples)) + 1j*np.random.normal(0,np.sqrt(0.5),(12,N_samples))
                noise_power  = sum(sum(np.abs(noise)**2))/(12*N_samples)
                if i == 0:
                    print()
                    print("SIGNAL POWER")
                    print(signal_power)
                    print("NOISE POWER")
                    print(noise_power)
                    print("SIGNAL TO NOISE RATIO")
                    print(signal_power/noise_power)

                # Received signal (shape: (12,500))
                z = A.dot(signal) + noise

                # Sample covariance matrix
                R_sample = z.dot(z.conj().T)/N_samples

                # Eigenvalue and eigenvectors
                w_sample, v_sample = np.linalg.eig(R_sample)
                
                if i == 0 and snr_dB == -20:
                    print()
                    print("EIGENVALUES OF SAMPLE COVARIANCE MATRIX")
                    print(w_sample[0])
                    print(w_sample[1])
                    print(w_sample[2])
                    print(w_sample[3])

                # Sensor Selection Matrix (shape: (12,6))
                T = np.array([[1,0,0,0,0,0],
                              [1,0,0,0,0,0],
                              [0,1,0,0,0,0],
                              [0,1,0,0,0,0],
                              [0,0,1,0,0,0],
                              [0,0,1,0,0,0],
                              [0,0,0,1,0,0],
                              [0,0,0,1,0,0],
                              [0,0,0,0,1,0],
                              [0,0,0,0,1,0],
                              [0,0,0,0,0,1],
                              [0,0,0,0,0,1]])

                # Push-Sum Matrix (shape: (6,6))
                P_push = np.array([[0.2,0.2,0.2,0  ,0  ,0],
                              [0.2,0.2,0.2,0  ,0  ,0],
                              [0.6,0.6,0.2,0.2,0  ,0],
                              [0  ,0  ,0.4,0.2,0.2,0.2],
                              [0  ,0  ,0  ,0.2,0.2,0.2],
                              [0  ,0  ,0  ,0.4,0.6,0.6]])

                # Average-Consensus Matrix (shape: (6,6))
                P_ave = np.array([[0.17,0.5,0.33,0  ,0  ,0],
                                  [0.5,0.17,0.33,0  ,0  ,0],
                                  [0.33,0.33,0.01,0.33,0  ,0],
                                  [0  ,0  ,0.33,0.01,0.33,0.33],
                                  [0  ,0  ,0  ,0.33,0.17,0.5],
                                  [0  ,0  ,0  ,0.33,0.5,0.17]])

                # Weight Vector  (shape: (6,1))
                w = np.atleast_2d([1,1,1,1,1,1]).T
                
                if method_code == 1:

                    # Average Consensus Covariance Matrix Estimation      
                    R_ave_con = K * np.multiply(T.dot(np.linalg.matrix_power(P_ave,iteration)).dot(T.T), R_sample)
                    R = R_ave_con
                
                if method_code == 2:
                    # Push-Sum Covariance Matrix Estimation
                    R_push_numerator = np.multiply(T.dot(np.linalg.matrix_power(P_push,iteration)).dot(T.T), R_sample)
                    R_push_denominator = T.dot(np.linalg.matrix_power(P_push,iteration)).dot(w).dot(np.ones((1,6))).dot(T.T)

                    # Push Sum Covariance Matrix (shape: (12,12))
                    R_push = K*np.multiply(R_push_numerator, (1/(R_push_denominator)))
                    R = R_push

                if method_code == 3:
                    # Conventional ESPRIT Algorithm      
                    R = R_sample               

                w_push, v_push = np.linalg.eig(R)

                # Upper group selection matrix J_up
                J_up = np.kron(np.eye(6),np.array([1,0]))

                # Lower group selection matrix J_down
                J_down = np.kron(np.eye(6),np.array([0,1]))

                # Push-Sum estimated signal eigenvector matrices
                U_s_push = v_push[:,:3]

                # Upper signal eigenvectors
                U_s_up = J_up.dot(U_s_push)

                # Lower signal eigenvectors
                U_s_down = J_down.dot(U_s_push)

                # Matrix including knowledge about DOAs of the source signals
                psi = np.linalg.inv((U_s_up.conj().T).dot(U_s_up)).dot((U_s_up.conj().T)).dot(U_s_down)

                w2, v2 = np.linalg.eig(psi)
                doa = []
                doa.append(np.arcsin(np.angle(w2[0])/np.pi)*360/(2*np.pi))
                doa.append(np.arcsin(np.angle(w2[1])/np.pi)*360/(2*np.pi))
                doa.append(np.arcsin(np.angle(w2[2])/np.pi)*360/(2*np.pi))

                if i == 0:
                    print()
                    print("  DOAs of the source signals in degrees with SNR: " + str(snr_dB) )
                    print("  DOAs of the source signals in degrees with N_samples: " + str(N_samples) )
                    print("****************************************************************")
                    print("****************************************************************")
                    print("DOA of the first source signal:   " + str(doa[0]))
                    print("DOA of the second source signal:   " + str(doa[1]))
                    print("DOA of the third source signal:   " + str(doa[2]))

                diff_1 = min(abs(doa[0]-(angles*360/(2*np.pi))))
                diff_2 = min(abs(doa[1]-(angles*360/(2*np.pi))))
                diff_3 = min(abs(doa[2]-(angles*360/(2*np.pi))))
                
                if SNR_range[1] == SNR_range[0] + 1:
                    if return_name == "rmse":
                        MSE[N_samples - N_samples_zero] = MSE[N_samples - N_samples_zero]+1/3*1/500*((diff_1)**2+(diff_2)**2+(diff_3)**2)
                        if i == 499: 
                            print("RMSE")
                            print(np.sqrt(MSE[N_samples - N_samples_zero]))
                    elif return_name == "cramer":
                        cramer[N_samples - N_samples_zero] = cramer[N_samples - N_samples_zero]+(1/500)*np.sqrt(cramer_rao(A, signal, angles, locations))*360/(2*np.pi)
                        if i == 499: 
                            print("Cramer Rao Bound")
                            print(np.sqrt(cramer[N_samples - N_samples_zero]))

                elif N_samples_range[1] == N_samples_range[0] + 1:
                    if return_name == "rmse":
                        MSE[snr_dB - SNR_zero] = MSE[snr_dB - SNR_zero]+1/3*1/500*((diff_1)**2+(diff_2)**2+(diff_3)**2)
                        if i == 499:
                            print("RMSE")
                            print(np.sqrt(MSE[snr_dB - SNR_zero]))
                    elif return_name == "cramer":
                        cramer[snr_dB - SNR_zero] = cramer[snr_dB - SNR_zero]+(1/500)*np.sqrt(cramer_rao(A, signal, angles, locations))*360/(2*np.pi)
                        if i == 499:
                            print("Cramer Rao Bound")
                            print((cramer[snr_dB - SNR_zero]))
                            
    if return_name == "rmse":
        return np.sqrt(MSE)
    elif return_name == "cramer":
        return cramer
    