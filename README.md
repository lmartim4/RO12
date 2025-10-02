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

This first experiment shows that the filter is actually working. In green we can see what the odometry alone would estimate, beeing completelly blinded from the landmarks measurements. There we can see that from the first seconds it has already a poor estimation, completelly diverging from the actual positioning.

![](/gifs/baseline.gif)

### Trou de measures

In this experiment we are simulating what happens if the landmarking sensoring stop suddenly [from time 2500s up to 3500s]. It is impressive how fast the algorithm fixes the new odometry error, showing that it in fact a roboust method.

![](/gifs/trou_sim/sensor_problems.gif)

### Sensor measurment interval

In a similar way of the experiment above, we can investigate how the algorithm performs when this "trou de measures" happens more frequently. Until the last experiment all simulation was happening with pose prediction and landmark measuring with the same frequency. Increasing this ratio, we can simulate 1 landmark measurement for every 5s and 50s.

| 5s | 50s |
| --- | --- |
| ![](/gifs/dt_meas/dt_meas_5.gif) | ![](/gifs/dt_meas/dt_meas_50.gif) |

We can see a similar behaviour to the last experiment where when there is no landmark we end up having to trust the odometry blindly. However, as it is for a small amount of time the filter is indeed able to quickly "reset" the error. 

Is is also possible to see that the $3\sigma$ confiance interval grows. It is clearer to confirm it by checking the trace(P) plot. The 5s interval "converges" much quicker than the "50s"



### QEst Noise

The QEst represents the covariance of errors used for the robot odometry. Therefore in the baseline experiment its value was $[0.01, 0.01, \pi/180]$.
It is then the error associated with the x-axis, y-axis and direction.

Some experiments were conducted in order to visualize how important those parameters are for the EFK performance. Worse odometries forces the filter to relie on the landmark sensoring and vice-versa. 

The figures on the left has a worse odometry than the baseline and better on the right. We start by checking the X-axis and Y-axis precision, then we move to direction.

| $\sigma_x$ and $\sigma_y$ $\times 10$ | $\sigma_x$ and $\sigma_y$ $/ 10$ |
| --- | --- |
| ![](/gifs/QEst/xy_10.gif) | ![](/gifs/QEst/x_y_div_ten.gif)|

| $\sigma_\theta\times 10$ | $\sigma_\theta/10$ |
| --- | --- |
| ![](/gifs/QEst/theta_10.gif) | ![](/gifs/QEst/theta_div_ten.gif) |

By analysing the figures above we can that the performace gain atributed to the direction odometry is brutal. Both x and y odomtries seems to play a imporant role in local prediction, we can see the trajectorie becomes much smoother. However to final position estimation a better direction estimation is crucial. As the original odometry already has a smooth movement it seems that Kalman is already trusting the local prediction to the robots odometry which seems to make sense once that our X and Y precision has and ordem of $10^{-2}$ m (centimeters). We'll better explore this on the next experiments.

### REst Noise

This section we'll be analysing what happens when we increase and decrease the sensors precision.

The following figures will compare our baseline to improved sensors. As mentioned earlier we suspected that our local odometry was already "good enough" leaving us with a smooth trajectory. Improving our sensor distance precision by a factor of ten we can see that the X and Y estimation has a smaller uncertainty but having such a better distance sensoring does not really improve direction estimation, which was already low anyway. Later on this document we'll investigate the behaviour of each sensor isolated, so that we can dig down on this hypotesis.

| Baseline | $\sigma_{distance}/10$ |
| --- | --- |
| ![](/gifs/baseline.gif) | ![](/gifs/REst/dist_sur_10.gif) |

In the following figures we can see that improving our direction sensor really clamps our already high precision direction estimation. It also improoved the X and Y precisions as we can see on the plot by having a thinner gap of the red lines on X and Y plots 

| Baseline | $\sigma_{direction}/10$ |
| --- | --- |
| ![](/gifs/baseline.gif) | ![](/gifs/REst/direction_sur_10.gif) |


In the following figures we are comparing worse sensors to better sensors relative to our baseline by a factor of ten. Once again the direction sensoring seems to be the one promoting the greatest benefits.

| $\sigma_{distance}\times10$ | $\sigma_{distance}/10$ | 
| --- | --- |
| ![](/gifs/REst/distance_times_10.gif) | ![](/gifs/REst/dist_sur_10.gif) |

| $\sigma_{direction}\times10$ | $\sigma_{direction}/10$ | 
| --- | --- |
| ![](/gifs/REst/direction_times_10.gif) | ![](/gifs/REst/direction_sur_10.gif) |

Overall comparison for both precisions compared to the baseline

| $\sigma_{distance}$ and $\sigma_{direction}\times10$ | Baseline | $\sigma_{distance}$ and $\sigma_{direction}/10$ | 
| --- | --- | --- |
| ![](/gifs/REst/angle_distance_times10.gif) | ![](/gifs/baseline.gif) | ![](/gifs/REst/dist_direct_sur_10.gif) |

### Sensor Isolation Investigation

For this part of the study we'll be investigating how each sensor alone performs on the absence of the other. We'll also search how important QEst and REst becomes in each scenario.  

|          |  Range-Only  | Direction-Only  |
|-------   |-----|-----|
| **QEst** |     |     |
| **REst** |     |     |

### Range Only

As shown previously in this document having the x-axis and y-axis odometry more accurate doesn't make a noticiable difference. In this range-only section we'll be able to testify the downsides of not having a direction sensoring. We'll start by changing the odometry precision it self and then change the precision of the distance sensor to see how well it performs. On the end we'll be checking how important the landmarks are and search a landmark amount threshold that won't make any great difference by adding more. 

#### QEst tunning

The figure on the right shows the previously conducted experiment on improiving QEst for x and y. The figure on the left shows the same exact experiemnt but without the direction sensor (range-only).

We can attest the direction sensor benefits for both direction and position estimations. If you take a look on he X,Y axis it is clear that the gap between the red lines are considerably smaller, so is the case for the direction plot.

The behaviour is analogous to higher $\sigma_x$ and $\sigma_y$.

| RANGE ONLY ($\sigma_x , \sigma_y$ / 10) | BOTH ($\sigma_x , \sigma_y$ / 10) |
| --- | --- |
| ![](/gifs/dist_only/QEst/x_y_div_ten.gif) | ![](/gifs/QEst/x_y_div_ten.gif) |


The brutal deviation from odometry due to poor direction estimation seems to be confirmed. In the following gif we can verify that by making the direction odometry 10x smaller made the odometry estimation much better. Even tough the general performance of the EKF does not seem to be that better. It is indeed much smoother and is explained by the new direction odomertry precision.

In the following figures we'll see what happens if the QEst had a much smaller $\sigma_\theta$.
We'll see that is the odometry alone is still not enough. Having both sensor is still better and improves overall estimation quality. It is clearly visible on the direction error estimation. Having both sensor provides a much better estimation.

| RANGE ONLY ($\sigma_\theta$ / 10) | BOTH ($\sigma_\theta$ / 10) |
| --- | --- |
| ![](/gifs/dist_only/QEst/theta_div_ten.gif) | ![](/gifs/QEst/theta_div_ten.gif) |


#### REst tunning




#### Landmarks 

The following test will vary the landmarks count. Our objective is to verify if only 3 landmarks are still enough for a good EKF performance as it was verified on the dual sensor setup.

| Landmarks = 1 | Landmarks = 2 |
| --- | --- |
| ![](/gifs/dist_only/landmarks/1.gif) | ![](/gifs/dist_only/landmarks/2.gif) |

| Landmarks = 3 | Landmarks = 30 |
| --- | --- |
| ![](/gifs/dist_only/landmarks/3.gif) | ![](/gifs/dist_only/landmarks/30.gif) |

#### Results

When only the distance sensor was used for measurements, no configuration led to improved direction estimation. All experiments showed similar performance, with errors around [-0.5, 0.5] rad. Varying the number of landmarks revealed that performance did not improve beyond three landmarks. With two landmarks, the performance was quite similar overall, although in some regions the Kalman filter lost track slightly, which is expected due to the triangulation problem.

To improve estimation, we varied QEst and REst. Adjusting the QEst parameters directly influenced the odometry estimation quality, leading to some improvement. However, even after reducing it by a factor of 10, the overall performance remained similar to before, suggesting that enhancing odometry precision is not an effective strategy.

On the other hand, improving the quality of the range sensor produced significant improvements. It allowed the Kalman filter to drastically reduce the estimation error.

### Direction Only

#### QEst tunning

#### REst tunning

#### Landmarks

| Landmarks = 1 | Landmarks = 2 |
| --- | --- |
| ![](/gifs/direc_only/1.gif) | ![](/gifs/direc_only/2.gif) |

| Landmarks = 3 | Landmarks = 30 |
| --- | --- |
| ![](/gifs/direc_only/3.gif) | ![](/gifs/direc_only/30.gif) |

#### Results