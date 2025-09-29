from math import sin, cos, atan2, pi, sqrt
import numpy as np

# ---- Kalman Filter: model functions ----

# evolution model (f)
def motion_model(x, u, dt_pred):
    # x: estimated state (x, y, heading)
    # u: control input or odometry measurement in body frame (Vx, Vy, angular rate)
    
    xPred = tcomp(x, u, dt_pred)
    xPred[2, 0] = angle_wrap(xPred[2, 0])
    
    return xPred


# observation model (h)
def observation_model(xVeh, iFeature, Map):
    # xVeh: vecule state
    # iFeature: observed amer index
    # Map: map of all amers

    Dx = Map[0, iFeature] - xVeh[0, 0]
    Dy = Map[1, iFeature] - xVeh[1, 0]
    theta_k = xVeh[2, 0]

    z = np.array([
        [np.sqrt((Dx)**2 + (Dy)**2)],
        [np.arctan2(Dy, Dx) - theta_k]
    ])

    z[1, 0] = angle_wrap(z[1, 0])
    
    return z


# ---- Kalman Filter: Jacobian functions to be completed ----

# h(x) Jacobian wrt x
def get_obs_jac(xPred, iFeature, Map):
    # xPred: predicted state
    # iFeature: observed amer index
    # Map: map of all amers
    
    Dx = Map[0, iFeature] - xPred[0, 0]
    Dy = Map[1, iFeature] - xPred[1, 0]

    r2_k = Dx**2 + Dy**2
    r_k = np.sqrt(r2_k)

    jH = np.array([
        [-Dx/r_k, -Dy/r_k, 0],
        [Dy/r2_k, -Dx/r2_k, -1]
    ])

    return jH


# f(x,u) Jacobian wrt x
def F(x, u, dt_pred):
    # x: estimated state (x, y, heading)
    # u: control input (Vx, Vy, angular rate)
    # dt_pred: time step
    
    vx = u[0, 0]
    vy = u[1, 0]
    
    theta = x[2, 0]
    s = np.sin(theta)
    c = np.cos(theta)
    
    df_dx = np.array([
        [1, 0, -(vx * s + vy * c) * dt_pred],
        [0, 1,  (vx * c - vy * s) * dt_pred],
        [0, 0,  1]
    ])
    
    return df_dx


# f(x,u) Jacobian wrt w (noise on the control input u)
def G(x, u, dt_pred):
    # x: estimated state (x, y, heading) in ground frame
    # u: control input (Vx, Vy, angular rate) in robot frame
    # dt_pred: time step for prediction

    theta = x[2, 0]
    s = np.sin(theta)
    c = np.cos(theta)
    
    df_du = np.array([
        [c*dt_pred, -s*dt_pred,0],
        [s*dt_pred, c*dt_pred, 0],
        [0, 0,  dt_pred]
    ])

    return df_du

# ---- Utils functions ----
# Display error ellipses
def plot_covariance_ellipse(xEst, PEst, axes, lineType):
    """
    Plot one covariance ellipse from covariance matrix
    """

    Pxy = PEst[0:2, 0:2]
    eigval, eigvec = np.linalg.eig(Pxy)

    if eigval[0] >= eigval[1]:
        bigind = 0
        smallind = 1
    else:
        bigind = 1
        smallind = 0

    if eigval[smallind] < 0:
        print('Pb with Pxy :\n', Pxy)
        exit()

    t = np.arange(0, 2 * pi + 0.1, 0.1)
    a = sqrt(eigval[bigind])
    b = sqrt(eigval[smallind])
    x = [3 * a * cos(it) for it in t]
    y = [3 * b * sin(it) for it in t]
    angle = atan2(eigvec[bigind, 1], eigvec[bigind, 0])
    rot = np.array([[cos(angle), sin(angle)],
                    [-sin(angle), cos(angle)]])
    fx = rot @ (np.array([x, y]))
    px = np.array(fx[0, :] + xEst[0, 0]).flatten()
    py = np.array(fx[1, :] + xEst[1, 0]).flatten()
    axes.plot(px, py, lineType)


# fit angle between -pi and pi
def angle_wrap(a):
    while a > np.pi:
        a = a - 2 * np.pi
    while a < -np.pi:
        a = a + 2 * np.pi
    return a


# composes two transformations
def tcomp(tab, tbc, dt):
    assert tab.ndim == 2 # eg: robot state [x, y, heading]
    assert tbc.ndim == 2 # eg: robot control [Vx, Vy, angle rate]
    #dt : time-step (s)

    angle = tab[2, 0] + dt * tbc[2, 0] # angular integration by Euler

    angle = angle_wrap(angle)
    s = sin(tab[2, 0])
    c = cos(tab[2, 0])
    position = tab[0:2] + dt * np.array([[c, -s], [s, c]]) @ tbc[0:2] # position integration by Euler
    out = np.vstack((position, angle))

    return out