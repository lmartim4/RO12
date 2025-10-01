# ROB12 - TP 1

## Code Structure

For a better understanding and code organization the code provided by the professors to the TP was cutted in 3 smaller parts. 

```
kalman.py
simulation.py
EKFLocalization.py

seed.py (just for global seed setup) 
```

### kalman.py

This section of code contains the **Kalman Filter** it self. It contains then the model functions such as :

```python
def motion_model(x, u, dt_pred):
    # x: estimated state (x, y, heading)
    # u: control input or odometry measurement in body frame (Vx, Vy, angular rate)
    # dt_pred: time step
    
    # returns the angular integration of x and u using tcomp function

def observation_model(xVeh, iFeature, Map):
    # xVeh: vecule state
    # iFeature: observed amer index
    # Map: map of all amers

    # returns the observation model (slide 29)

def get_obs_jac(xPred, iFeature, Map):
    # xPred: predicted state
    # iFeature: observed amer index
    # Map: map of all amers
    
    # return observation jacobian (slide 29)
    
def F(x, u, dt_pred):
    # x: estimated state (x, y, heading)
    # u: control input (Vx, Vy, angular rate)
    # dt_pred: time step
    
    # returns the prediction model matrix F (slide 28)

def G(x, u, dt_pred):
    # x: estimated state (x, y, heading) in ground frame
    # u: control input (Vx, Vy, angular rate) in robot frame
    # dt_pred: time step for prediction

    # returns the linearization of model control/noise (calculated by hand)

    theta = x[2, 0]
    s = np.sin(theta)
    c = np.cos(theta)
    
    df_du = np.array([
        [c*dt_pred, -s*dt_pred,0],
        [s*dt_pred, c*dt_pred, 0],
        [0, 0,  dt_pred]
    ])

    return df_du

```

### simulation.py

**Simulation** models the robot by generating control inputs, updating the true position, simulating noisy odometry for the EKF, and providing occasional noisy landmark measurements.


```python
class Simulation:        
    def get_robot_control(self, k):
        # generate  sin trajectory
        return np.array([[0, 0.025,  0.1*np.pi / 180 * sin(3*np.pi * k / self.nSteps)]]).T
        
    def simulate_world(self, k):
        # updates true position

    def get_odometry(self, k):
        # simulates the robot's noisy motion by propagating the odometry state with the control input and then adding Gaussian noise from QTrue, returning both the updated noisy state and noisy control, which is what the EKF uses instead of the true state.

    def get_observation(self, k):
        # the robot sometimes gets a measurement of a random landmark. That measurement is noisy, and sometimes it may not exist at all (depending on timing and validity conditions).
```

### EKFLocalization

This is the main program. It is responsible for using the other files to implement the Extended Kalman Filter algorithm for mobile robot localization, simulating robot motion with odometry and landmark observations. This program is also responsible for configurating all simulation parameters and ploting the simulation into a nice screen. It tracks errors and covariance, and visualizes trajectories, 3σ bounds, and covariance evolution in real time, supporting distance, direction, or combined sensor modes.

```python
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
```
## Experiments

In the following subsections some experiments will be conducted in order to get a better understanding and intuition on how important each paramenter is and how well can this EKF can perform under non ideal conditions. 

The code provided was a bit modified so that the plots will also inform the state of the parameters under study: 

- Landmark count
- Delta time between measurements
- Available sensors (distance/direction/both]
- Odometry noise/precision
- Sensor noise/precision

Also, I added the trace of P plot so that we can see the general converging.

I used the *save* parameter to generate the intermediare figures and then made animated GIFs

### Baseline

![](/gifs/baseline.gif)

### Sensor measurment interval
| 5s | 50s |
| --- | --- |
| ![](/gifs/dt_meas/dt_meas_5.gif) | ![](/gifs/dt_meas/dt_meas_50.gif) |

### QEst Noise

| x , y (times 10) | $\theta$ (times 10) |
| --- | --- |
| ![](/gifs/QEst/xy_10.gif) | ![](/gifs/QEst/theta_10.gif) |


| x , y , $\theta$ (times 10) | x , y , $\theta$ (times 100) |
| --- | --- |
| ![](/gifs/QEst/xy_theta_10.gif) | ![](/gifs/QEst/xy_theta100.gif) |

### REst Noise

| distance (times 10) | direction (times 100) |
| --- | --- |
| ![](/gifs/REst/distance_10.gif) | ![](/gifs/REst/angle_10.gif) |

distance , direction (times 100)
![](/gifs/REst/angle_distance_10.gif)

### Trou de measures

![](/gifs/trou_sim/sensor_problems.gif)

### Range Only

![](/gifs/dist_only/1.gif)
![](/gifs/dist_only/2.gif.gif)
![](/gifs/dist_only/3.gif)
![](/gifs/dist_only/30.gif.gif)

### Direction Only

![](/gifs/direc_only/1.gif)
![](/gifs/direc_only/2.gif.gif)
![](/gifs/direc_only/3.gif)
![](/gifs/direc_only/30.gif.gif)