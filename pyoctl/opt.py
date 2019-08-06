import numpy as np


def dynprg_gains(A, B, Q, R, H, T, dt=None):
    """Computes gains for dynamic programming.

    Parameters
    ----------
    A : list
        Coefficients of matrix A. Each element of the list should contain one
        line from the model's A matrix.

    B : list
        Coefficients of matrix B. Each element of the list should contain one
        line from the model's B matrix.

    Q : list
        Coefficients of matrix Q. Each element of the list should contain one
        line from the cost index's Q matrix.

    R : list
        Coefficients of matrix H. Each element of the list should contain one
        line from the cost index's R matrix.

    H : list
        Coefficients of matrix H. Each element of the list should contain one
        line from the cost index's H matrix.

    T : int, float
        Final time or number of points to simulate.

    dt : int, float, NoneType
        Discretization interval. If `None`, the system is considered to be
        discrete. If a number, the system is considered to be continuous and
        it is internally discretized. By default, is `None`.

    Returns
    -------
    F : np.ndarray
        Vector with state's gain for feedback.
        
    """
    # Model
    A = np.array(A)
    B = np.array(B)
    Q = np.array(Q)
    R = np.array(R)
    H = np.array(H)
        
    if dt is None:
        N = int(T)        
    else:
        N = int(T / dt)
        A = np.eye(A.shape[0]) + dt * A
        B = dt * B
        Q = dt * Q
        R = dt * R

    # Number of states
    n_states = A.shape[0]
    
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
    """Simulates a system's response with dynamic programming gains.

    Parameters
    ----------
    A : list
        Coefficients of matrix A. Each element of the list should contain one
        line from the model's A matrix.

    B : list
        Coefficients of matrix B. Each element of the list should contain one
        line from the model's B matrix.

    Q : list
        Coefficients of matrix Q. Each element of the list should contain one
        line from the cost index's Q matrix.

    R : list
        Coefficients of matrix H. Each element of the list should contain one
        line from the cost index's R matrix.

    H : list
        Coefficients of matrix H. Each element of the list should contain one
        line from the cost index's H matrix.

    F : np.ndarray
        Vector with state's gain for feedback. This vector must be of
        appropriate size, i.e., must be compatible with the number of
        simulation steps/points.
        
    xi : list
        States' initial conditions. Each element of the list corresponds to
        one state.

    T : int, float
        Final time or number of points to simulate.

    dt : int, float, NoneType
        Discretization interval. If `None`, the system is considered to be
        discrete. If a number, the system is considered to be continuous and
        it is internally discretized. By default, is `None`.

    Returns
    -------
    x : np.ndarray
        A matrix, where each column corresponds to one state.

    u : np.ndarray
        A matrix, where each column corresponds to one control signal.
        
    """
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

    return x, u


def riccati_k(A, B, Q, R, K, dt):
    """Computes one integration step for Riccati's gain equation. Numerical
    integration is performed backward in time.

    Parameters
    ----------
    A : np.ndarray
        Coefficients of matrix A.

    B : np.ndarray
        Coefficients of matrix B.

    Q : np.ndarray
        Coefficients of matrix Q.

    R : np.ndarray
        Coefficients of matrix H.

    K : np.ndarray
        Gain coefficients for the current step.

    dt : int, float
        Discretization interval.     

    Returns
    -------
    np.ndarray
        The value for K one step backward in time.
        
    """
    R_inv = np.linalg.inv(R)

    # Numerical integration from tf to t0 (backwards in time) changes the
    # sign from + dt * ( ) to - dt * ( )
    return K - dt * (-K @ A + -A.T @ K + -Q + K @ B @ R_inv @ B.T @ K)


def riccati_gains(A, B, Q, R, H, T, dt):
    """Computes gains from Riccati's equation.

    Parameters
    ----------
    A : list
        Coefficients of matrix A. Each element of the list should contain one
        line from the model's A matrix.

    B : list
        Coefficients of matrix B. Each element of the list should contain one
        line from the model's B matrix.

    Q : list
        Coefficients of matrix Q. Each element of the list should contain one
        line from the cost index's Q matrix.

    R : list
        Coefficients of matrix H. Each element of the list should contain one
        line from the cost index's R matrix.

    H : list
        Coefficients of matrix H. Each element of the list should contain one
        line from the cost index's H matrix.

    T : int, float
        Final time or number of points to simulate.

    dt : int, float
        Discretization interval. 

    Returns
    -------
    K : np.ndarray
        Vector with state's gain for feedback.
        
    """
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

    # Order is reversed since integration must be performed from tf to t0
    # (backwards in time) to meet K(tf) = H
    K[N - 1] = H
    for n in reversed(range(0, N - 1)):
        K[n] = riccati_k(A, B, Q, R, K[n + 1], dt)

    return K

def riccati_sim(A, B, Q, R, H, K, xi, T, dt=None):
    """Simulates a system's response with Riccati's gains.

    Parameters
    ----------
    A : list
        Coefficients of matrix A. Each element of the list should contain one
        line from the model's A matrix.

    B : list
        Coefficients of matrix B. Each element of the list should contain one
        line from the model's B matrix.

    Q : list
        Coefficients of matrix Q. Each element of the list should contain one
        line from the cost index's Q matrix.

    R : list
        Coefficients of matrix H. Each element of the list should contain one
        line from the cost index's R matrix.

    H : list
        Coefficients of matrix H. Each element of the list should contain one
        line from the cost index's H matrix.

    K : np.ndarray
        Vector with state's gain for feedback. This vector must be of
        appropriate size, i.e., must be compatible with the number of
        simulation steps/points.
        
    xi : list
        States' initial conditions. Each element of the list corresponds to
        one state.

    T : int, float
        Final time or number of points to simulate.

    dt : int, float, NoneType
        Discretization interval. If `None`, the system is considered to be
        discrete. If a number, the system is considered to be continuous and
        it is internally discretized. By default, is `None`.

    Returns
    -------
    x : np.ndarray
        A matrix, where each column corresponds to one state.

    u : np.ndarray
        A matrix, where each column corresponds to one control signal.
        
    """
    # Model
    A = np.array(A)
    B = np.array(B)
    Q = np.array(Q)
    R = np.array(R)
    H = np.array(H)
    xi = np.array(xi)

    # If model is not discrete, performs discretization
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
        u[k] = -R_inv @ B.T @ K[k] @ x[k, :]
        x[k + 1] = A @ x[k, :] + B @ u[k, :]

    # Computes the last u so we don't have a discontinuity
    u[N - 1] = R_inv @ B.T @ K[N - 1] @ x[N - 1, :]

    return x, u 
