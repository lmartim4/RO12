"""
TP Kalman filter for mobile robots localization

authors: Goran Frehse, David Filliat, Nicolas Merlinge
"""

import matplotlib.pyplot as plt
import numpy as np

import simulation as sim
from kalman import F, G, get_obs_jac, motion_model, plot_covariance_ellipse, observation_model, angle_wrap, pi

import os
try:
    os.makedirs("outputs")
except:
    pass



# =============================================================================
# Main Program
# =============================================================================

# Init displays
show_animation = True
save = False

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14, 7))
ax3 = plt.subplot(3, 2, 2)
ax4 = plt.subplot(3, 2, 4)
ax5 = plt.subplot(3, 2, 6)

# ---- General variables ----

# Simulation time
Tf = 6000       # final time (s)
dt_pred = 1     # Time between two dynamical predictions (s)
dt_meas = 1     # Time between two measurement updates (s)

# Location of landmarks
nLandmarks = 30
Map = 140*(np.random.rand(2, nLandmarks) - 1/2)

# True covariance of errors used for simulating robot movements
QTrue = np.diag([0.01, 0.01, 1*pi/180]) ** 2
RTrue = np.diag([3.0, 3*pi/180]) ** 2

# Modeled errors used in the Kalman filter process
QEst = 1*np.eye(3, 3) @ QTrue
REst = 1*np.eye(2, 2) @ RTrue

# initial conditions
xTrue = np.array([[1, -40, -pi/2]]).T
xOdom = xTrue
xEst = xTrue
PEst = 10 * np.diag([1, 1, (1*pi/180)**2])

# Init history matrixes
hxEst = xEst
hxTrue = xTrue
hxOdom = xOdom
hxError = np.abs(xEst-xTrue)  # pose error
hxVar = np.sqrt(np.diag(PEst).reshape(3, 1))  # state std dev
htime = [0]

# Simulation environment
simulation = sim.Simulation(Tf, dt_pred, xTrue, QTrue, xOdom, Map, RTrue, dt_meas)

# Temporal loop
for k in range(1, simulation.nSteps):

    # Simulate robot motion
    simulation.simulate_world(k)

    # Get odometry measurements
    xOdom, u_tilde = simulation.get_odometry(k)

    # Kalman prediction

    Fk = F(xEst, u_tilde, dt_pred)
    Gk = G(xEst, u_tilde, dt_pred)

    xPred = motion_model(xEst,u_tilde,dt_pred)
    PPred = Fk @ PEst @ Fk.T + Gk @ QEst @ Gk.T

    # Get random landmark observation
    [z, iFeature] = simulation.get_observation(k)

    if z is not None:
        # Predict observation
        zPred = observation_model(xPred, iFeature, Map)

        # get observation Jacobian
        H = get_obs_jac(xPred, iFeature, Map)

        # compute Kalman gain - with dir and distance
        Innov = z - zPred # observation error (innovation)
        Innov[1, 0] = angle_wrap(Innov[1, 0])
        
        S = H @ PPred @ H.T + REst 
        K = PPred @ H.T @ np.linalg.inv(S)

        # Compute Kalman gain to use only distance
#        Innov = #...................       # observation error (innovation)
#        H = #...................
#        S = #...................
#        K = #...................

        # Compute Kalman gain to use only direction
#        Innov = #...................       # observation error (innovation)
#        Innov[1, 0] = angle_wrap(Innov[1, 0])
#        H = #...................           # observation error (innovation)
#        S = #...................
#        K = #...................

        # perform kalman update
        xEst =  xPred + K @ Innov
        xEst[2, 0] = angle_wrap(xEst[2, 0])

        PEst = (np.eye(3) - K @ H) @PPred
        
        
        PEst = 0.5 * (PEst + PEst.T)  # ensure symetry

    else:
        # there was no observation available
        xEst = xPred
        PEst = PPred

    # store data history
    hxTrue = np.hstack((hxTrue, simulation.xTrue))
    hxOdom = np.hstack((hxOdom, simulation.xOdom))
    hxEst = np.hstack((hxEst, xEst))
    err = xEst - simulation.xTrue
    err[2, 0] = angle_wrap(err[2, 0])
    hxError = np.hstack((hxError, err))
    hxVar = np.hstack((hxVar, np.sqrt(np.diag(PEst).reshape(3, 1))))
    htime.append(k*simulation.dt_pred)

    # plot every 15 updates
    if show_animation and k*simulation.dt_pred % 200 == 0:
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])

        ax1.cla()
        
        times = np.stack(htime)

        # Plot true landmark and trajectory
        ax1.plot(Map[0, :], Map[1, :], "*k")
        ax1.plot(hxTrue[0, :], hxTrue[1, :], "-k", label="True")

        # Plot odometry trajectory
        ax1.plot(hxOdom[0, :], hxOdom[1, :], "-g", label="Odom")

        # Plot estimated trajectory an pose covariance
        ax1.plot(hxEst[0, :], hxEst[1, :], "-r", label="EKF")
        ax1.plot(xEst[0], xEst[1], ".r")
        plot_covariance_ellipse(xEst,
                                PEst, ax1, "--r")

        ax1.axis([-70, 70, -70, 70])
        ax1.grid(True)
        ax1.legend()

        # plot errors curves
        ax3.plot(times, hxError[0, :], 'b')
        ax3.plot(times, 3.0 * hxVar[0, :], 'r')
        ax3.plot(times, -3.0 * hxVar[0, :], 'r')
        ax3.grid(True)
        ax3.set_ylabel('x (m)')
        ax3.set_xlabel('time (s)')
        ax3.set_title('Real error (blue) and 3 $\sigma$ covariances (red)')

        ax4.plot(times, hxError[1, :], 'b')
        ax4.plot(times, 3.0 * hxVar[1, :], 'r')
        ax4.plot(times, -3.0 * hxVar[1, :], 'r')
        ax4.grid(True)
        ax4.set_ylabel('y (m)')
        ax5.set_xlabel('time (s)')

        ax5.plot(times, hxError[2, :], 'b')
        ax5.plot(times, 3.0 * hxVar[2, :], 'r')
        ax5.plot(times, -3.0 * hxVar[2, :], 'r')
        ax5.grid(True)
        ax5.set_ylabel(r"$\theta$ (rad)")
        ax5.set_xlabel('time (s)')
        
        if save: plt.savefig(r'outputs/EKF_' + str(k) + '.png')
#        plt.pause(0.001)

plt.savefig('EKFLocalization.png')
print("Press Q in figure to finish...")
plt.show()
