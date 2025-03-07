from casadi import *
import numpy as np
import math
import timeit

from functions import Auxiliary

class MPC(object):
    def __init__(self, T, N, state):
        # ==============================
        #  CONFIGURATION
        # ==============================
        self.T = T                  # The total prediction time horizon.
        self.N = N                  # The number of discrete time steps used for prediction.

        # Number of state variables (x, y, theta, vt, vr)
        # Number of control inputs (Acceleration A, Steering Phi)
        self.nr_states = 5; self.nr_controls = 2

        self.sol = None

        # ==============================
        # SYSTEM CONSTRAINT PARAMETERS
        # ==============================
        self.configs = {'max_A': 10.0,  'max_Phi': 1.5, 'max_vt': 10.0,  'max_vr':0.5,
                        'min_A': -10.0, 'min_Phi': -1.5, 'min_vt':-10.0, 'min_vr':-0.5}

        # Distance constraints
        self.min_distance = 45
        self.max_distance = 60

        self.prev_target_x = 0
        self.prev_target_y = 0

        # ==============================
        # CONSTANTS
        # ==============================
        # State cost gain
        self.alpha = 10
        self.beta = 1

        # ==============================
        # ROBOT CHARACTERISTICS
        # ==============================
        # (Section for robot dynamics?)

        # Aux variables
        self.status = ""
        self.current_distance = 0;

    def solve_uav_to_ugv(self, state, gp):
        """
        Solves the mpc optimization problem
        """
        # ==============================
        # INITIAL CONDITIONS
        # ==============================
        # The optimization problem starts from the latest state measurements
        self.init_x = [state['x']]; self.init_y = [state['y']]; self.init_theta = [state['theta']]
        self.init_vt = [state['vt']]; self.init_vr = [state['vr']]

        # Calculate the current distance to the target
        self.current_distance = (state['x'] - gp[0]) ** 2 + (state['y'] - gp[1]) ** 2

        opti = Opti() # Optimization problem

        # ==============================
        # DECISION VARIABLES
        # ==============================
        X = opti.variable(self.nr_states, self.N+1)     # state trajectory: 5 state variables over (N+1) time steps
        U = opti.variable(self.nr_controls, self.N)     # control trajectory (throttle)

        # The cost function will be accumulated dynamically
        objective = 0

        # ==============================
        # STATE VARIABLES
        # ==============================
        self.pos_x = X[0,:]
        self.pos_y = X[1,:]
        self.pos_theta = X[2,:]
        self.vel_t = X[3,:]
        self.vel_r = X[4,:]

        # ==============================
        # CONTROL VARIABLES
        # ==============================
        self.A = U[0, :]
        self.Phi = U[1, :]

        # ==============================
        # SYSTEM DYNAMICS
        # ==============================
        # Unicycle model
        f = lambda x,u: vertcat(x[3]*cos(x[2]),     # dx/dt = vt*cos(theta)
                                x[3]*sin(x[2]),     # dy/dt = vt*sin(theta)
                                x[4],               # dtheta/dt = vr
                                u[0],               # dvt/dt = A
                                u[1])               # dvr/dt = phi

        dt = self.T/self.N          # Time interval: length of a control interval
        for k in range(self.N):     # loop over control intervals
            # Runge-Kutta 4 integration
            k1 = f(X[0:self.nr_states,k],         U[0:self.nr_controls,k])
            k2 = f(X[0:self.nr_states,k]+dt/2*k1, U[0:self.nr_controls,k])
            k3 = f(X[0:self.nr_states,k]+dt/2*k2, U[0:self.nr_controls,k])
            k4 = f(X[0:self.nr_states,k]+dt*k3,   U[0:self.nr_controls,k])
            x_next = X[0:self.nr_states,k] + dt/6*(k1+2*k2+2*k3+k4)
            opti.subject_to(X[0:self.nr_states,k+1]==x_next) # close the gaps

        # ==============================
        # SYSTEM CONSTRAINTS
        # ==============================
        # Enforce control input limits (Acceleration & Steering)
        # and velocity limits (Translational & Rotational)
        opti.subject_to(opti.bounded(self.configs['min_A'],    self.A     , self.configs['max_A']))
        opti.subject_to(opti.bounded(self.configs['min_Phi'],  self.Phi   , self.configs['max_Phi']))
        opti.subject_to(opti.bounded(self.configs['min_vt'],   self.vel_t , self.configs['max_vt'] ))
        opti.subject_to(opti.bounded(self.configs['min_vr'],   self.vel_r , self.configs['max_vr']))

        # ---- boundary conditions --------
        # ==============================
        # INITIAL STATE CONSTRAINTS
        # ==============================

        # The first state in the trajectory matches the current robot state
        opti.subject_to(self.pos_x[0]==self.init_x)
        opti.subject_to(self.pos_y[0]==self.init_y)
        opti.subject_to(self.pos_theta[0]==self.init_theta)
        opti.subject_to(self.vel_t[0]==self.init_vt)
        opti.subject_to(self.vel_r[0]==self.init_vr)

        # initialize variables x and y
        for k in range(1,self.N+1):
            opti.set_initial(self.pos_x[k], self.init_x)
            opti.set_initial(self.pos_y[k], self.init_y)

        # ==============================
        # DEFINE COST FUNCTION
        # ==============================

        # "Desired distance" (midpoint of the comfort region range)
        desired_distance = (self.min_distance + self.max_distance) / 2

        # Check if the target is moving
        target_moving = (abs(gp[0] - self.prev_target_x) > 1e-3 or abs(gp[1] - self.prev_target_y) > 1e-3)
        #print(arctan2(gp[1] - state["y"] + epsilon, gp[0] - state["x"]))

        #print(f"{self.init_vr=:}")
        #print(f"{self.init_vt=:}")
        #print(f"{self.init_theta=:}")

        # Define the cost function
        objective = 0
        for k in range(0, self.N + 1):

            # garantee soft constraint for FOV for all stages
            angle_to_target = arctan2(gp[1] - self.pos_y[k], gp[0] - self.pos_x[k])
            angle_error = atan2(sin(angle_to_target - self.pos_theta[k]), cos(angle_to_target - self.pos_theta[k]))
            objective += 0.01 * angle_error**2


            if self.current_distance > self.max_distance ** 2:
                # Stage 1: Minimize distance to the target
                self.status = "Pursuit"
                objective += self.alpha * (self.pos_x[k] - (gp[0])) ** 2
                objective += self.alpha * (self.pos_y[k] - (gp[1])) ** 2

            elif self.current_distance < self.min_distance ** 2:
                # Stage 2: Maximize distance to the target
                self.status = "Retreat"
                objective -= self.alpha * (self.pos_x[k] - gp[0]) ** 2
                objective -= self.alpha * (self.pos_y[k] - gp[1]) ** 2

            else:
                # ==============================
                # FOV CONSTRAINT
                # ==============================
                if target_moving:
                    self.status = "Pursuit"
                    objective += 0.001 * (self.vel_t[k] ** 2)
                else:
                    # Add dead zone
                    self.status = "Stopped"
                    objective += 10.0 * (self.vel_t[k] ** 2 + self.vel_r[k] ** 2)

            # Update the target's previous position
            self.prev_target_x = gp[0]
            self.prev_target_y = gp[1]

        opti.minimize(objective)

        # ==============================
        # SOLVER CONFIGURATION
        # ==============================
        options = {"print_time": False, "ipopt": {"max_iter": 100000}, "ipopt.print_level": 0}

        opti.solver("ipopt",options) # set numerical backend
        start_time = timeit.default_timer()

        # ==============================
        # SOLVE THE OPTIMIZATION PROBLEM
        # ==============================
        self.dt = dt
        self.sol = None
        try:
            self.sol = opti.solve()   # actual solve
            converged = True
        except:
            print("Not converged")
            converged = False

        # measure elapsed time
        elapsed = timeit.default_timer() - start_time
        print("To solve took {} seconds".format(elapsed))

        # ==============================
        # RETRIEVE FINAL COST VALUE
        # ==============================
        obj_vals =  opti.debug.stats()['iterations']['obj']

        self.cost = obj_vals[-1]

        return converged

    def update_state(self, start):
        """
        Extracts the optimized state values at a given time step.
        """
        state = {}
        state['x'] = self.sol.value(self.pos_x[start])
        state['y'] = self.sol.value(self.pos_y[start])
        state['theta'] = self.sol.value(self.pos_theta[start])

        state['vt'] = self.sol.value(self.vel_t[start])
        state['vr'] = self.sol.value(self.vel_r[start])
        return state

    def get_control(self, i=0):
        """
        Extracts the optimal control inputs from the solver.
        """
        control = {}
        control['A'] = self.sol.value(self.A[i])
        control['Phi'] = self.sol.value(self.Phi[i])
        return control
