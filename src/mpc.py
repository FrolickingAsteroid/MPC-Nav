"""
Filename: mpc.py
Description:
    MPC implementation for a people-following robot
"""

import casadi as ca

from casadi import *
import numpy as np
import math
import timeit

from StaticConstraint import StaticObstacle

class MPC(object):
    def __init__(self, T, N, state, map):
        # ==============================
        #  CONFIGURATION
        # ==============================
        self.T = T                  # The total prediction time horizon.
        self.N = N                  # The number of discrete time steps used for prediction.

        # Number of state variables (x, y, theta, vt, vr)
        # Number of control inputs (Acceleration A, Steering Phi)
        self.nr_states = 5; self.nr_controls = 2

        # ==============================
        # SYSTEM CONSTRAINT PARAMETERS
        # ==============================
        self.configs = {'max_A': 50.0,  'max_Phi': 1.5, 'max_vt': 50.0,  'max_vr':1.5,
                        'min_A': -50.0, 'min_Phi': -1.5, 'min_vt':-50.0, 'min_vr':-1.5}

        # Distance constraints
        self.min_distance = 50      # 1 m
        self.max_distance = 100     # 2 m

        # ==============================
        # CONSTANTS
        # ==============================
        # State cost gain
        # (add when certain...)

        # ==============================
        # ROBOT CHARACTERISTICS
        # ==============================
        # (Section for robot dynamics?)

        # ==============================
        # AUX VARIABLES
        # ==============================
        # Time related variables for printing
        self.status = ""
        self.solve_time = ""
        self.elapsed = 0

        # Optimization problem solution
        self.sol = None

        # distance information for the objective
        self.current_distance = 0;
        self.prev_target_x = 0
        self.prev_target_y = 0

        # ==============================
        # STATIC OBSTACLE AVOIDANCE
        # ==============================
        self.stat_obj = StaticObstacle(map)
        self.prev_trajectory = None
        self.constraints = []
        self.corners = []

        self.static_constraint = 0

    def mpc_opt(self, state, gp):
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
        U = opti.variable(self.nr_controls, self.N)     # control trajectory

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
        f = lambda x,u: vertcat(x[3]*cos(x[2]),     # dx/self.dt = vt*cos(theta)
                                x[3]*sin(x[2]),     # dy/self.dt = vt*sin(theta)
                                x[4],               # self.dtheta/self.dt = vr
                                u[0],               # dvt/self.dt = A
                                u[1])               # dvr/self.dt = phi

        self.dt = self.T/self.N          # Time interval: length of a control interval
        for k in range(self.N):          # loop over control intervals
            # Runge-Kutta 4 integration
            k1 = f(X[0:self.nr_states,k],         U[0:self.nr_controls,k])
            k2 = f(X[0:self.nr_states,k]+self.dt/2*k1, U[0:self.nr_controls,k])
            k3 = f(X[0:self.nr_states,k]+self.dt/2*k2, U[0:self.nr_controls,k])
            k4 = f(X[0:self.nr_states,k]+self.dt*k3,   U[0:self.nr_controls,k])
            x_next = X[0:self.nr_states,k] + self.dt/6*(k1+2*k2+2*k3+k4)
            opti.subject_to(X[0:self.nr_states,k+1]==x_next) # close the gaps

        # ==============================
        # INIT X Y AND THETA
        # ==============================
        if self.prev_trajectory:
            # Shift previous trajectory forward
            warm_start = self.stat_obj.shift_previous_trajectory(self.prev_trajectory, self.N)
            for k in range(0, self.N):
                opti.set_initial(self.pos_x[k], warm_start[k][0])
                opti.set_initial(self.pos_y[k], warm_start[k][1])
                opti.set_initial(self.pos_theta[k], warm_start[k][2])

            # Compute collision free fregions
            self.compute_static_constraints(warm_start)
        else:
            # If no previous trajectory, initialize using current robot state
            for k in range(0, self.N):
                opti.set_initial(self.pos_x[k], self.init_x)
                opti.set_initial(self.pos_y[k], self.init_y)

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

        # ==============================
        # DEFINE COST FUNCTION
        # ==============================
        # CHANGE TO FUZZY
        # Check if the target is moving
        target_moving = (abs(gp[0] - self.prev_target_x) > 1e-3 or abs(gp[1] - self.prev_target_y) > 1e-3)

        # Compute estimated target velocity
        vt_x = (gp[0] - self.prev_target_x) / self.dt
        vt_y = (gp[1] - self.prev_target_y) / self.dt
        target_speed = sqrt(vt_x**2 + vt_y**2)

        # Check heading angle for sharp turns
        target_heading = arctan2(gp[1] - self.prev_target_y, gp[0] - self.prev_target_x)

        # Use angle wrapping [-pi, pi] to prevent discontinuity and large angle errors
        # (Easier than changing frame atm)
        angle_error = atan2(sin(target_heading - state['theta']), cos(target_heading - state['theta']))
        large_angle_error = abs(angle_error) > 1

        # Define the cost function
        objective = 0
        objective = self.static_constraint
        for k in range(0, self.N + 1):
            # !NOTE Fov Soft constraint (for all states), recheck
            angle_to_target = arctan2(gp[1] - self.pos_y[k], gp[0] - self.pos_x[k])
            angle_error = atan2(sin(angle_to_target - self.pos_theta[k]), cos(angle_to_target - self.pos_theta[k]))
            objective += 1 * angle_error**2

            if self.min_distance < sqrt(self.current_distance) < self.max_distance:
                if not target_moving:

                    self.status = "Stopped"
                    objective += 10.0 * (self.vel_t[k] ** 2 + self.vel_r[k] ** 2)
                elif large_angle_error:

                    self.status = "Turn-In-Place"
                    objective += 5.0 * angle_error**2
                    objective += 10.0 * self.vel_t[k]**2
                else:

                    self.status = "Track target"
                    velocity_difference = self.vel_t[k] - target_speed
                    objective += 0.1 * (velocity_difference)**2
            elif  sqrt(self.current_distance) < self.min_distance:

                self.status = "Retreat"
                objective -= 1 * (self.pos_x[k] - gp[0])**2
                objective -= 1 * (self.pos_y[k] - gp[1])**2
            else:

                self.status = "Pursuit"
                objective += 1 * (self.pos_x[k] - (gp[0])) ** 2
                objective += 1 * (self.pos_y[k] - (gp[1])) ** 2

        # Update previous target position
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
        converged = self.solve_problem(opti)

        # measure elapsed time
        self.elapsed = timeit.default_timer() - start_time

        self.solve_time = "To solve took {:.4f} seconds".format(self.elapsed)
        print(self.solve_time)

        if converged:
            obj_vals =  opti.debug.stats()['iterations']['obj']
            self.cost = obj_vals[-1]

        return converged


    def compute_static_constraints(self, warm_start):
        """
        Compute and apply bounding box to robot
        """
        self.static_constraint = 0
        self.constraints, self.corners = self.stat_obj.compute_collision_free_area(warm_start)

        # Iterate over each prediction step
        for k in range(len(self.constraints)):
            constraint_set = self.constraints[k]

            # Sclae importance for each step
            weight = 1 * (k / self.N)

            for A, B, C in constraint_set:
                # penalize violation outside the constraint
                # If the constraint is satisfied, violation = 0
                # else violatio > 0
                lhs = A * self.pos_x[k] + B * self.pos_y[k]
                violation = lhs - C
                self.static_constraint += 1 * violation**2

    def solve_problem(self, opti):
        """
        Attempt to solve optimization problem
        """
        self.dt = self.dt
        self.sol = None
        try:
            self.sol = opti.solve()   # actual solve
            converged = True

            # Store the optimized trajectory for the next iteration
            self.prev_trajectory = [(self.sol.value(self.pos_x[k]),
                self.sol.value(self.pos_y[k]),
                self.sol.value(self.pos_theta[k])) for k in range(self.N+1)]

        except:
            print("Not converged")
            converged = False

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
