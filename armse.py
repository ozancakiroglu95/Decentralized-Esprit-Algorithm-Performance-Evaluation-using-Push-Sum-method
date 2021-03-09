def average_consensus_armse(SNR_range, N_samples_range, iteration, A, angles, locations, K, L, M):

    import numpy as np
    
    N_samples_zero = N_samples_range[0]
    SNR_zero = SNR_range[0]
    
    
    if SNR_range[1] == SNR_range[0] + 1:
        ARMSE = np.zeros(N_samples_range[1]-N_samples_range[0])
                
    elif N_samples_range[1] == N_samples_range[0] + 1:
        ARMSE = np.zeros(SNR_range[1]-SNR_range[0])
    
    
    for snr_dB in range(SNR_range[0],SNR_range[1]):
        
        for N_samples in range(N_samples_range[0], N_samples_range[1]):
    
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
            
            # right eigenvectors of R
            w1, u = np.linalg.eig(R_sample)
            
            # Upper group selection matrix J_up
            J_up = np.kron(np.eye(6),np.array([1,0]))

            # Lower group selection matrix J_down
            J_down = np.kron(np.eye(6),np.array([0,1]))

            # Push-Sum estimated signal eigenvector matrices
            U_s = u[:,:3]

            # Upper signal eigenvectors
            U_s_up = J_up.dot(U_s)

            # Lower signal eigenvectors
            U_s_down = J_down.dot(U_s)

            # Matrix including knowledge about DOAs of the source signals
            psi = np.linalg.inv((U_s_up.conj().T).dot(U_s_up)).dot((U_s_up.conj().T)).dot(U_s_down)
            
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

            # Average-Consensus Matrix (shape: (6,6))
            P_ave = np.array([[0.17,0.5,0.33,0  ,0  ,0],
                              [0.5,0.17,0.33,0  ,0  ,0],
                              [0.33,0.33,0.01,0.33,0  ,0],
                              [0  ,0  ,0.33,0.01,0.33,0.33],
                              [0  ,0  ,0  ,0.33,0.17,0.5],
                              [0  ,0  ,0  ,0.33,0.5,0.17]])
            
            w2, r_l = np.linalg.eig(psi)
            doa = []
            doa.append(np.arcsin(np.angle(w2[0])/np.pi)*360/(2*np.pi))
            doa.append(np.arcsin(np.angle(w2[1])/np.pi)*360/(2*np.pi))
            doa.append(np.arcsin(np.angle(w2[2])/np.pi)*360/(2*np.pi))

            print()
            print("  DOAs of the source signals in degrees with SNR: " + str(snr_dB) )
            print("  DOAs of the source signals in degrees with N_samples: " + str(N_samples) )
            print("****************************************************************")
            print("****************************************************************")
            print("DOA of the first source signal:   " + str(doa[0]))
            print("DOA of the second source signal:   " + str(doa[1]))
            print("DOA of the third source signal:   " + str(doa[2]))


            v, V = np.linalg.eig(psi.T)
            a, beta = np.linalg.eig(P_ave)

            mse_error = np.zeros(3, dtype=np.complex128)
                
            for k in range(3):
                
                for n in range(50):
                    
                    # left eigenvectors of psi
                    q_l = V[:, k].T
                    gamma_H = q_l.dot(np.linalg.inv(U_s_up.conj().T.dot(U_s_up))).dot(U_s_up.conj().T).dot(J_up - w2[k].conj()*J_down)
                    mu_H = q_l.dot(np.linalg.inv(U_s_up.conj().T.dot(U_s_up))).dot(U_s_up.conj().T).dot(J_down - w2[k]*J_up)

                    inner_exp_1 = np.zeros((M,M))
                    for i in range(1,L+1):
                        for j in range(1,M+1):
                            if i != j:
                                inner_exp_1 = inner_exp_1 + (1/N_samples)*((w1[i-1]*w1[j-1])/((w1[i-1]-w1[j-1])**2))*(r_l[:,k].reshape((3,1)).dot(r_l[:,k].reshape((3,1)).conj().T))[i-1,i-1]*(u[:,i-1]).dot(u[:,i-1].conj().T)

                    h_i = np.zeros((12,1))
                    h_j = np.zeros((12,1))
                    for i in range(1,L+1):
                        for j in range(1,L+1):
                            for m in range(2,K+1):
                                h_i = h_i + K*a[m-1]**iteration*np.diag(T.dot(beta[:,m-1])).dot(R_sample).dot(np.diag(T.dot(beta[:,m-1])).conj().T).dot(u[:,i-1])
                                h_j = h_j + K*a[m-1]**iteration*np.diag(T.dot(beta[:,m-1])).dot(R_sample).dot(np.diag(T.dot(beta[:,m-1])).conj().T).dot(u[:,j-1])
                            B_i = (np.delete(u, i-1, 1)).dot(np.linalg.pinv(np.diag(np.delete((w1-w1[i-1]), i-1)))).dot(np.delete(u, i-1, 1).conj().T)
                            B_j = (np.delete(u, j-1, 1)).dot(np.linalg.pinv(np.diag(np.delete((w1-w1[j-1]), j-1)))).dot(np.delete(u, j-1, 1).conj().T)
                            inner_exp_1 = inner_exp_1 + (r_l[:,k].reshape((3,1)).dot(r_l[:,k].reshape((3,1)).conj().T))[i-1,j-1]*B_i.dot(h_i).dot(h_j.conj().T).dot(B_j.conj().T)

                    inner_exp_2 = np.zeros((M,M))
                    for i in range(1,L+1):
                        for j in range(1,L+1):
                            if i != j:
                                inner_exp_2 = inner_exp_2 - (1/N_samples)*((r_l[:,k].reshape((3,1)).dot(r_l[k].reshape((3,1)).T))[i-1,j-1]*w1[i-1]*w1[j-1]*u[:,i-1].dot(u[:,j-1].T))/((w1[i-1]-w1[j-1])**2)

                    h_i = np.zeros((12,1))
                    h_j = np.zeros((12,1))
                    for i in range(1,L+1):
                        for j in range(1,L+1):
                            for m in range(2,K+1):
                                h_i = h_i + K*a[m-1]**iteration*np.diag(T.dot(beta[:,m-1])).dot(R_sample).dot(np.diag(T.dot(beta[:,m-1])).conj().T).dot(u[:,i-1])
                                h_j = h_j + K*a[m-1]**iteration*np.diag(T.dot(beta[:,m-1])).dot(R_sample).dot(np.diag(T.dot(beta[:,m-1])).conj().T).dot(u[:,j-1])
                            B_i = (np.delete(u, i-1, 1)).dot(np.linalg.pinv(np.diag(np.delete((w1-w1[i-1]), i-1)))).dot(np.delete(u, i-1, 1).conj().T)
                            B_j = (np.delete(u, j-1, 1)).dot(np.linalg.pinv(np.diag(np.delete((w1-w1[j-1]), j-1)))).dot(np.delete(u, j-1, 1).conj().T)
                            inner_exp_2 = inner_exp_2 + (r_l[:,k].reshape((3,1)).dot(r_l[:,k].reshape((3,1)).T))[i-1,j-1]*B_i.dot(h_i).dot(h_j.T).dot(B_j.T)               

                    gamma_expectation = gamma_H.dot(inner_exp_1).dot(gamma_H.conj().T)
                    mu_expectation = mu_H.dot(inner_exp_2).dot(mu_H.T)
                    mse_error[k] = mse_error[k] + (1/50)*(gamma_expectation-np.real(((w2[k].conj())**2)*(mu_expectation))/(2*(np.pi*np.cos(np.arcsin(np.angle(w2[k])/np.pi)))**2))

            if SNR_range[1] == SNR_range[0] + 1:
                ARMSE[N_samples - N_samples_zero] = abs(np.sqrt(sum(mse_error)/3))
                print(ARMSE[N_samples - N_samples_zero])

            elif N_samples_range[1] == N_samples_range[0] + 1:
                ARMSE[snr_dB - SNR_zero] = abs(np.sqrt(sum(mse_error)/3))
                print(ARMSE[snr_dB - SNR_zero])
              
    return ARMSE
    


