import numpy as np


def dynprg_gains(A, B, Q, R, H, T, dt=None):
    """Computes gains for dynamic programming.

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

    H : np.ndarray
        Coefficients of matrix H.

    T : int, float
        Final time or number of points to simulate.

    dt : int, float, NoneType
        Discretization interval. If `None`, the system is considered to be
        discrete. If a number, the system is considered to be continuous and
        it is discretized before computing gains. By default, is `None`.

    Returns
    -------
    F : np.ndarray
        Vector with state's gain for feedback.
        
    """        
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


def dynprg_sim(A, B, Q, R, H, F, xi, T, dt=None, gain='dynamic',
               sat_control=False, u_sat=None):
    """Simulates a system's response with dynamic programming gains.

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

    H : np.ndarray
        Coefficients of matrix H.

    F : np.ndarray
        Vector with state's gain for feedback. This vector must be of
        appropriate size, i.e., must be compatible with the number of
        simulation steps/points.
        
    xi : np.ndarray
        States' initial conditions.

    T : int, float
        Final time or number of points to simulate.

    dt : int, float, NoneType
        Discretization interval. If `None`, the system is considered to be
        discrete. If a number, the system is considered to be continuous and
        it is discretized before simulation. By default, is `None`.

    gain : str
        Defines if gain is dynamic or static. If `dynamic`, each value of `F`
        is applied at each time step. If `static`, the first value of `F` is
        applied at all time steps. By default, is `dynamic`.

    sat_control : bool
        Defines if control signal is bounded. By default, is `False`.

    u_sat : list, NoneType
        Minimum and maximum value allowed for each control signal. Each
        element of the list corresponds to each control signal. Each element
        of the list should be another list, as [min, max] value allowed. By
        default, is `None`.

    Returns
    -------
    x : np.ndarray
        A matrix, where each column corresponds to one state.

    u : np.ndarray
        A matrix, where each column corresponds to one control signal.
        
    """        
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
    if sat_control is True:
        u_sat = np.array(u_sat)
    
    x[0] = xi
    if gain == 'dynamic':
        for k in range(N - 1):
            u[k] = F[None, k] @ x[k, :]
            if sat_control is True:
                u[k] = control_sat(u[k], u_sat)
            x[k + 1] = A @ x[k, :] + B @ u[k, :]

        # Computes the last u so we don't have a discontinuity
        u[N - 1] = F[None, N - 1] @ x[N - 1, :]
        if sat_control is True:
            u[N - 1] = control_sat(u[N - 1], u_sat)
    else:
        for k in range(N - 1):
            u[k] = F[None, 0] @ x[k, :]
            if sat_control is True:
                u[k] = control_sat(u[k], u_sat)
            x[k + 1] = A @ x[k, :] + B @ u[k, :]        

        # Computes the last u so we don't have a discontinuity
        u[N - 1] = F[None, 0] @ x[N - 1, :]
        if sat_control is True:
            u[N - 1] = control_sat(u[N - 1], u_sat)
            
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


def riccati_s(A, B, Q, R, K, s, r, dt):
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

    s : np.ndarray
        Gain coefficients for the current step.

    r : np.ndarray
        Reference for the current step.

    dt : int, float
        Discretization interval.     

    Returns
    -------
    np.ndarray
        The value for K one step backward in time.
        
    """
    R_inv = np.linalg.inv(R)
    
    return s + dt * (A.T - K @ B @ R_inv @ B.T) @ s - dt * Q @ r    


def riccati_gains(A, B, Q, R, H, T, dt):
    """Computes gains from Riccati's equation.

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

    H : np.ndarray
        Coefficients of matrix H.

    T : int, float
        Final time or number of points to simulate.

    dt : int, float
        Discretization interval. 

    Returns
    -------
    K : np.ndarray
        Vector with state's gain for feedback.
        
    """
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


def riccati_gains_tracking(A, B, Q, R, H, r, T, dt):
    """Computes gains from Riccati's equation.

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

    H : np.ndarray
        Coefficients of matrix H.

    r : np.ndarray
        Reference.

    T : int, float
        Final time or number of points to simulate.

    dt : int, float
        Discretization interval. 

    Returns
    -------
    K : np.ndarray
        Vector with state's gain for feedback.
        
    """
    # Number of points
    N = int(T / dt)

    # Gain vector
    K = np.zeros((N, A.shape[0], A.shape[0]))
    s = np.zeros((N, A.shape[0], 1))

    # Order is reversed since integration must be performed from tf to t0
    # (backwards in time) to meet K(tf) = H
    K[N - 1] = H
    for n in reversed(range(0, N - 1)):
        K[n] = riccati_k(A, B, Q, R, K[n + 1], dt)
    
    s[N - 1] = -H @ r[N - 1].reshape(-1, 1)
    for n in reversed(range(0, N - 1)):
        s[n] = riccati_s(A, B, Q, R, K[n + 1], s[n + 1],
                         r[n + 1].reshape(-1, 1), dt)
    
    return K, s


def riccati_sim(A, B, Q, R, H, K, xi, T, dt=None):
    """Simulates a system's response with Riccati's gains.

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

    H : np.ndarray
        Coefficients of matrix H.

    K : np.ndarray
        Vector with state's gain for feedback. This vector must be of
        appropriate size, i.e., must be compatible with the number of
        simulation steps/points.
        
    xi : np.ndarray
        States' initial conditions.

    T : int, float
        Final time or number of points to simulate.

    dt : int, float, NoneType
        Discretization interval. If `None`, the system is considered to be
        discrete. If a number, the system is considered to be continuous and
        it is discretized before simulation. By default, is `None`.

    Returns
    -------
    x : np.ndarray
        A matrix, where each column corresponds to one state.

    u : np.ndarray
        A matrix, where each column corresponds to one control signal.
        
    """
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


def control_sat(u, us):
    """Helper function to saturate a control signal.

    Warning: this function only works with 1-D control signals if `us` is a
    list

    Parameters
    ----------
    u : np.ndarray
        Vector with control signal.

    us : np.ndarray
        Array containing saturation values. Each line should contain the
        min and max value allowed for each control signal.
        
    Returns
    -------
    u : np.ndarray
        Vector with saturated control signal.
        
    """
    us = us.T
    u[u < us[0]] = us[0]
    u[u > us[1]] = us[1]
    
    return u
