import numpy as np


def dynprg_gains(A, B, Q, R, H, xi, T, dt=None):

    # Model
    A = np.array(A)
    B = np.array(B)
    Q = np.array(Q)
    R = np.array(R)
    H = np.array(H)
    xi = np.array(xi)
        
    if dt is None:
        N = int(T)        
    else:
        N = int(T / dt)
        A = np.eye(A.shape[0]) + dt * A
        B = dt * B
        Q = dt * Q
        R = dt * R

    # Number of states
    n_states = xi.shape[0]
    
    # Gains
    F = np.zeros((N, n_states))

    # Computes gains
    P = H
    for k in range(1, N + 1):
        F[N - k] = -np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
        P = (A + B @ F[N - k][np.newaxis, :]).T @ P @ \
            (A + B @ F[N - k][np.newaxis, :]) + F[N - k][np.newaxis, :].T \
            @ R[np.newaxis, :] @ F[N - k][np.newaxis, :] + Q

    return F


def dynprg_sim(A, B, Q, R, H, F, xi, T, dt=None):

    # Model
    A = np.array(A)
    B = np.array(B)
    Q = np.array(Q)
    R = np.array(R)
    H = np.array(H)
    xi = np.array(xi)
        
    if dt is None:
        N = int(T)        
    else:
        N = int(T / dt)
        A = np.eye(A.shape[0]) + dt * A
        B = dt * B
        Q = dt * Q
        R = dt * R

    # Number of states
    n_states = xi.shape[0]

    # Number of controls
    n_controls = B.shape[1]

    # States
    x = np.zeros((N, n_states))

    # Control
    u = np.zeros((N, n_controls))
    
    x[0] = xi
    for k in range(N - 1):
        u[k] = F[None, k] @ x[k, :]
        x[k + 1] = A @ x[k, :] + B @ u[k, :]
        #x[k + 1] = (A + B @ F[None, k]) @ x[k, :]

    return x, u


def riccati_k(A, B, Q, R, K, dt):

    R_inv = np.linalg.inv(R)
    
    return K + dt * (-K @ A + -A.T @ K + -Q + K @ B @ R_inv @ B.T @ K)


def riccati_gains(A, B, Q, R, H, T, dt):

    # Model
    A = np.array(A)
    B = np.array(B)
    Q = np.array(Q)
    R = np.array(R)
    H = np.array(H)

    # Number of points
    N = int(T / dt)

    # Gain vector
    K = np.zeros((N, A.shape[0], A.shape[0]))

    for n in range(1, N):
        K[n] = riccati_k(A, B, Q, R, K[n - 1], dt)

    return K

def riccati_sim(A, B, Q, R, H, K, xi, T, dt=None):

    # Model
    A = np.array(A)
    B = np.array(B)
    Q = np.array(Q)
    R = np.array(R)
    H = np.array(H)
    xi = np.array(xi)
        
    if dt is None:
        N = int(T)        
    else:
        N = int(T / dt)
        A = np.eye(A.shape[0]) + dt * A
        B = dt * B
        Q = dt * Q
        R = dt * R

    # Number of states
    n_states = xi.shape[0]

    # Number of controls
    n_controls = B.shape[1]

    # States
    x = np.zeros((N, n_states))

    # Control
    u = np.zeros((N, n_controls))

    R_inv = np.linalg.inv(R)
    x[0] = xi
    for k in range(N - 1):
        #u[k] = F[None, k] @ x[k, :]
        u[k] = R_inv @ B.T @ K[k] @ x[k, :]
        x[k + 1] = A @ x[k, :] + B @ u[k, :]
        #x[k + 1] = (A + B @ F[None, k]) @ x[k, :]
    u[N - 1] = R_inv @ B.T @ K[N - 1] @ x[N - 1, :]

    return x, u 
