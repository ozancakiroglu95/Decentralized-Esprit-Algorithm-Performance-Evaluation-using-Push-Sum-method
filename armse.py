def average_consensus_armse(SNR_range, N_samples_range, iteration, method, A, angles, locations, K, L, M):

    import numpy as np
    from scipy import linalg
    
    N_samples_zero = N_samples_range[0]
    SNR_zero = SNR_range[0]
    
    
    if SNR_range[1] == SNR_range[0] + 1:
        ARMSE = np.zeros(N_samples_range[1]-N_samples_range[0])
                
    elif N_samples_range[1] == N_samples_range[0] + 1:
        ARMSE = np.zeros(SNR_range[1]-SNR_range[0])
    
    
    for snr_dB in range(SNR_range[0],SNR_range[1]):
        
        for N_samples in range(N_samples_range[0], N_samples_range[1]):
            # Signal(A*s) to noise(n) ratio
            received_snr = 10 ** (snr_dB / 10)
            ratio_As_to_s = 1/4
            snr = received_snr * ratio_As_to_s

            mse_error = np.zeros(3, dtype=np.complex128)

            for n in range(50):

                # Source signal implementation (shape: (3,500))
                signal = np.random.normal(0, np.sqrt(snr), (3, N_samples))

                # Noise signal implementation (shape: (12,500))
                noise = np.random.normal(0, np.sqrt(0.5), (12, N_samples)) + 1j * np.random.normal(0, np.sqrt(0.5),(12, N_samples))

                # Received signal (shape: (12,500))
                z = A.dot(signal) + noise

                # Sample covariance matrix
                R_sample = z.dot(z.conj().T) / N_samples

                # right eigenvectors of R
                w1, u = np.linalg.eig(R_sample)

                # Upper group selection matrix J_up
                J_up = np.kron(np.eye(6), np.array([1, 0]))

                # Lower group selection matrix J_down
                J_down = np.kron(np.eye(6), np.array([0, 1]))

                # Push-Sum estimated signal eigenvector matrices
                U_s = u[:, :3]

                # Upper signal eigenvectors
                U_s_up = J_up.dot(U_s)

                # Lower signal eigenvectors
                U_s_down = J_down.dot(U_s)

                # Matrix including knowledge about DOAs of the source signals
                psi = np.linalg.inv((U_s_up.conj().T).dot(U_s_up)).dot((U_s_up.conj().T)).dot(U_s_down)

                # Sensor Selection Matrix (shape: (12,6))
                T = np.array([[1, 0, 0, 0, 0, 0],
                              [1, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 1],
                              [0, 0, 0, 0, 0, 1]])

                # Average-Consensus Matrix (shape: (6,6))
                P_ave = np.array([[0.17, 0.5 , 0.33, 0   , 0   , 0   ],
                                  [0.5 , 0.17, 0.33, 0   , 0   , 0   ],
                                  [0.33, 0.33, 0.01, 0.33, 0   , 0   ],
                                  [0   ,    0, 0.33, 0.01, 0.33, 0.33],
                                  [0   ,    0, 0   , 0.33, 0.17, 0.5 ],
                                  [0   ,    0, 0   , 0.33, 0.5 , 0.17]])

                # Push-Sum Matrix (shape: (6,6))
                P_push = np.array([[0.2, 0.2, 0.2, 0, 0, 0],
                                   [0.2, 0.2, 0.2, 0, 0, 0],
                                   [0.6, 0.6, 0.2, 0.2, 0, 0],
                                   [0, 0, 0.4, 0.2, 0.2, 0.2],
                                   [0, 0, 0, 0.2, 0.2, 0.2],
                                   [0, 0, 0, 0.4, 0.6, 0.6]])

                # Push-Sum Matrix (shape: (6,6))
                P_random = np.array([[0.3, 0.2, 0.8, 0.1, 0.3, 0.2],
                                   [0.2, 0.2, 0.1, 0.1, 0.3, 0.5],
                                   [0.6, 0.6, 0.2, 0.9, 0.3, 0.3],
                                   [0.5, 0.3, 0.4, 0.4, 0.3, 0.2],
                                   [0.7, 0.6, 0.9, 0.3, 0.3, 0.2],
                                   [0.9, 0.3, 0.2, 0.4, 0.1, 0.6]])

                # Weight Vector  (shape: (6,1))
                w = np.atleast_2d([1, 1, 1, 1, 1, 1]).T

                if method == "ave":
                    a, beta = np.linalg.eig(P_ave)
                elif method == "push":
                    a, beta = np.linalg.eig(P_push)

                idx = a.argsort()[::-1]
                a = a[idx]
                beta = beta[:, idx]

                v, V = np.linalg.eig(psi.T)
                idx = v.argsort()[::-1]
                v  = v[idx]
                V = V[:, idx]

                w2, r_l = np.linalg.eig(psi)
                idx = w2.argsort()[::-1]
                w2  = w2[idx]
                r_l = r_l[:, idx]

                doa = []
                doa.append(np.arcsin(np.angle(w2[0]) / np.pi) * 360 / (2 * np.pi))
                doa.append(np.arcsin(np.angle(w2[1]) / np.pi) * 360 / (2 * np.pi))
                doa.append(np.arcsin(np.angle(w2[2]) / np.pi) * 360 / (2 * np.pi))

                for k in range(3):
                    
                    # left eigenvectors of psi
                    q_l = V[:, k].reshape((1,3))
                    gamma_H = q_l.dot(np.linalg.pinv(U_s_up.conj().T.dot(U_s_up))).dot(U_s_up.conj().T).dot(J_up - w2[k].conj() * J_down)
                    mu_H = q_l.dot(np.linalg.pinv(U_s_up.conj().T.dot(U_s_up))).dot(U_s_up.conj().T).dot(J_down - w2[k] * J_up)

                    r_l_k = r_l[:, k].reshape((3, 1))
                    inner_exp_1 = np.zeros((M, M), dtype=np.complex128)
                    for i in range(L):
                        for j in range(M):
                            if i != j:
                                u_i = u[:, i].reshape((12,1))
                                inner_exp_1 = inner_exp_1 + ((1/N_samples) * (((w1[i] * w1[j]) / ((w1[i] - w1[j]) ** 2)) * (r_l_k.dot(r_l_k.conj().T))[i, i] * ((u_i).dot(u_i.conj().T))))

                    for i in range(L):
                        for j in range(L):
                            h_i = np.zeros((12, 1), dtype=np.complex128)
                            h_j = np.zeros((12, 1), dtype=np.complex128)
                            for m in range(1, K):
                                u_i = u[:, i].reshape((12, 1))
                                u_j = u[:, j].reshape((12, 1))
                                beta_m = beta[:, m].reshape((6, 1))
                                if method == "push":
                                    T_push_k = np.diag(np.squeeze(T.dot(beta_m)))
                                    T_push_l = np.diag(T.dot(np.linalg.pinv(beta).T)[:, m].conj().T)
                                    h_denominator = T.dot(np.linalg.matrix_power(P_push, iteration)).dot(w).dot(np.ones((1, 6))).dot(T.T)
                                    h_push_first = np.multiply((T_push_k.dot(R_sample).dot(T_push_l.conj().T)), (1 / h_denominator))
                                    h_i = h_i + K * (a[m] ** iteration)* h_push_first.dot(u_i)
                                    h_j = h_j + K * (a[m] ** iteration)* h_push_first.dot(u_j)
                                if method == "ave":
                                    h_ave_first = np.diag(np.squeeze(T.dot(beta_m))).dot(R_sample).dot(np.diag(np.squeeze(T.dot(beta_m))).conj().T)
                                    h_i = h_i + K * (a[m] ** iteration) * h_ave_first.dot(u_i)
                                    h_j = h_j + K * (a[m] ** iteration) * h_ave_first.dot(u_j)
                            B_i = (np.delete(u, i, 1)).dot(np.linalg.pinv(np.diag(np.delete((w1 - w1[i]), i)))).dot(np.delete(u, i, 1).conj().T)
                            B_j = (np.delete(u, j, 1)).dot(np.linalg.pinv(np.diag(np.delete((w1 - w1[j]), j)))).dot(np.delete(u, j, 1).conj().T)
                            inner_exp_1 = inner_exp_1 + (r_l_k.dot(r_l_k.conj().T))[i, j] * B_i.dot(h_i).dot(h_j.conj().T).dot(B_j.conj().T)

                    inner_exp_2 = np.zeros((M, M), dtype=np.complex128)
                    for i in range(L):
                        for j in range(L):
                            if i != j:
                                u_i = u[:, i].reshape((12, 1))
                                u_j = u[:, j].reshape((12, 1))
                                inner_exp_2 = inner_exp_2 - ((1/N_samples) * (((r_l_k.dot(r_l_k.T))[i, j] * w1[i] * w1[j] * u_i.dot(u_j.T)) /  ((w1[i] - w1[j])**2)))

                    for i in range(L):
                        for j in range(L):
                            h_i = np.zeros((12, 1), dtype=np.complex128)
                            h_j = np.zeros((12, 1), dtype=np.complex128)
                            for m in range(1, K):
                                u_i = u[:, i].reshape((12, 1))
                                u_j = u[:, j].reshape((12, 1))
                                beta_m = beta[:, m].reshape((6, 1))
                                if method == "push":
                                    T_push_k = np.diag(np.squeeze(T.dot(beta_m)))
                                    T_push_l = np.diag(T.dot(np.linalg.pinv(beta).T)[:, m].conj().T)
                                    h_denominator = T.dot(np.linalg.matrix_power(P_push, iteration)).dot(w).dot(np.ones((1, 6))).dot(T.T)
                                    h_push_first = np.multiply((T_push_k.dot(R_sample).dot(T_push_l.conj().T)), (1 / h_denominator))
                                    h_i = h_i + K * (a[m] ** iteration) * h_push_first.dot(u_i)
                                    h_j = h_j + K * (a[m] ** iteration) * h_push_first.dot(u_j)
                                if method == "ave":
                                    h_ave_first = np.diag(np.squeeze(T.dot(beta_m))).dot(R_sample).dot(np.diag(np.squeeze(T.dot(beta_m))).conj().T)
                                    h_i = h_i + K * (a[m] ** iteration) * h_ave_first.dot(u_i)
                                    h_j = h_j + K * (a[m] ** iteration) * h_ave_first.dot(u_j)
                            B_i = (np.delete(u, i, 1)).dot(np.linalg.pinv(np.diag(np.delete((w1 - w1[i]), i)))).dot(np.delete(u, i, 1).conj().T)
                            B_j = (np.delete(u, j, 1)).dot(np.linalg.pinv(np.diag(np.delete((w1 - w1[j]), j)))).dot(np.delete(u, j, 1).conj().T)
                            inner_exp_2 = inner_exp_2 + (r_l_k.dot(r_l_k.T))[i, j] * B_i.dot(h_i).dot(h_j.T).dot(B_j.T)
                    
                    gamma_expectation = gamma_H.dot(inner_exp_1).dot(gamma_H.conj().T)
                    mu_expectation = mu_H.dot(inner_exp_2).dot(mu_H.T)

                    mse_error[k] = mse_error[k] + (1/50) * ((gamma_expectation - np.real(((w2[k].conj()) ** 2) * (mu_expectation))) / (2 * ((np.pi * np.cos(np.arcsin(np.angle(w2[k]) / np.pi))) ** 2)))

            if SNR_range[1] == SNR_range[0] + 1:
                ARMSE[N_samples - N_samples_zero] = abs(np.sqrt((sum(mse_error)/3))*(360/(2*np.pi)))
                print(ARMSE[N_samples - N_samples_zero])

            elif N_samples_range[1] == N_samples_range[0] + 1:
                ARMSE[snr_dB - SNR_zero] = abs(np.sqrt((sum(mse_error)/3))*(360/(2*np.pi)))
                print(ARMSE[snr_dB - SNR_zero])

    return ARMSE
    


