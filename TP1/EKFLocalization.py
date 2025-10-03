"""
TP Kalman filter for mobile robots localization

authors: Goran Frehse, David Filliat, Nicolas Merlinge
"""

import matplotlib.pyplot as plt
from seed import *

import simulation as sim
from kalman import F, G, get_obs_jac, motion_model, plot_covariance_ellipse, observation_model, angle_wrap, pi

import os
try:
    os.makedirs("outputs")
except:
    pass

SENSOR_MODE = 'direction' # distance+direction

# Init displays
show_animation = True
save = True

fig = plt.figure(figsize=(14, 10))
ax1 = plt.subplot(1, 2, 1)
ax3 = plt.subplot(4, 2, 2)
ax4 = plt.subplot(4, 2, 4)
ax5 = plt.subplot(4, 2, 6)
ax6 = plt.subplot(4, 2, 8)

# ---- General variables ----

# Simulation time
Tf = 6000       # final time (s)
dt_pred = 1     # Time between two dynamical predictions (s)
dt_meas = 1     # Time between two measurement updates (s)

# Location of landmarks
nLandmarks = 4
Map = 140*(np.random.rand(2, nLandmarks) - 1/2)

# True covariance of errors used for simulating robot movements
QTrue = np.diag([0.1, 0.1, 1*pi/180]) ** 2
RTrue = np.diag([3, 3*pi/180]) ** 2

# Modeled errors used in the Kalman filter process
QEst = 1*np.eye(3, 3) @ QTrue
REst = 1*np.eye(2, 2) @ RTrue

# Calculate sigma values for display
QEst_sigmas = np.sqrt(np.diag(QEst))
REst_sigmas = np.sqrt(np.diag(REst))

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
hPTrace = [np.trace(PEst)]
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
        H = get_obs_jac(xPred, iFeature, Map)
        
        if SENSOR_MODE == 'distance+direction':
            # compute Kalman gain - with dir and distance
            Innov = z - zPred # observation error (innovation)
            Innov[1, 0] = angle_wrap(Innov[1, 0])
            
            S = H @ PPred @ H.T + REst 

        elif SENSOR_MODE == 'distance':
            # Compute Kalman gain to use only distance
            Innov = z[0:1, :] - zPred[0:1, :]
            H = H[0:1, :]
            S = H @ PPred @ H.T + REst[0:1, 0:1]
            
        elif SENSOR_MODE == 'direction':
            # Compute Kalman gain to use only direction
            Innov = z[1:2, :] - zPred[1:2, :]
            Innov[0, 0] = angle_wrap(Innov[0, 0])
            H = H[1:2, :]
            S = H @ PPred @ H.T + REst[1:2, 1:2]
        else:
            print("Invalid SENSOR_MODE")
        
        K = PPred @ H.T @ np.linalg.inv(S)

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
    hPTrace.append(np.trace(PEst))
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
        
        param_text = f'Simulation Parameters:\n'
        param_text += f'Landmarks: {nLandmarks}\n'
        param_text += f'dt_meas: {dt_meas} s\n'
        param_text += f'Sensor: {SENSOR_MODE}\n'
        param_text += f'QEst $\sigma$: [{QEst_sigmas[0]:.4f}, {QEst_sigmas[1]:.4f}, {QEst_sigmas[2]:.4f}]\n'
        param_text += f'REst $\sigma$: [{REst_sigmas[0]:.2f}, {REst_sigmas[1]:.4f}]'
        
        ax1.text(0.22, 1.13, param_text, transform=ax1.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax3.plot(times, hxError[0, :], 'b')
        ax3.plot(times, 3.0 * hxVar[0, :], 'r')
        ax3.plot(times, -3.0 * hxVar[0, :], 'r')
        ax3.grid(True)
        ax3.set_ylabel('x (m)')
        ax3.set_title('Real error (blue) and 3 $\sigma$ covariances (red)')

        ax4.plot(times, hxError[1, :], 'b')
        ax4.plot(times, 3.0 * hxVar[1, :], 'r')
        ax4.plot(times, -3.0 * hxVar[1, :], 'r')
        ax4.grid(True)
        ax4.set_ylabel('y (m)')

        ax5.plot(times, hxError[2, :], 'b')
        ax5.plot(times, 3.0 * hxVar[2, :], 'r')
        ax5.plot(times, -3.0 * hxVar[2, :], 'r')
        ax5.grid(True)
        ax5.set_ylabel(r"$\theta$ (rad)")
        
        ax6.plot(times, hPTrace, 'g', linewidth=2)
        ax6.grid(True)
        ax6.set_ylabel('Trace(P)')
        ax6.set_xlabel('time (s)')

        if save: plt.savefig(r'outputs/EKF_' + str(k) + '.png')
#        plt.pause(0.001)

plt.savefig('EKFLocalization.png')
print("Press Q in figure to finish...")
plt.show()