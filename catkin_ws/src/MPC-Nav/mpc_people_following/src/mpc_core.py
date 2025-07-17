"""
Filename: mpc_core.py
Description:
    MPC implementation for a people-following robot
"""

import casadi as ca

from casadi import *
import numpy as np
import math
import timeit

class MPC(object):
    def __init__(self, T, N):
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
        self.configs = {'max_A': 1.0,  'max_Phi': 1.0, 'max_vt': 0.5,  'max_vr':1.5,
                        'min_A': -1.0, 'min_Phi': -1.0, 'min_vt':-0.5, 'min_vr':-1.5}

        # ==============================
        # AUX VARIABLES
        # ==============================
        self.prev_trajectory = None

        # Time related variables for printing
        self.status = ""
        self.solve_time = ""
        self.elapsed = 0

        # Optimization problem solution
        self.sol = None


    def mpc_opt(self, state, gp, weights, polyhedron_planes, plane_weights):
        """
        Solves the mpc optimization problem
        """
        # ==============================
        # INITIAL CONDITIONS
        # ==============================
        # The optimization problem starts from the latest state measurements
        self.init_x = [state['x']]; self.init_y = [state['y']]; self.init_theta = [state['theta']]
        self.init_vt = [state['vt']]; self.init_vr = [state['vr']]

        opti = Opti() # Optimization problem

        # ==============================
        # DECISION VARIABLES
        # ==============================
        # X[1]
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
            warm_start = self.shift_previous_trajectory(self.prev_trajectory, self.N)
            for k in range(0, self.N):
                opti.set_initial(self.pos_x[k], warm_start[k][0])
                opti.set_initial(self.pos_y[k], warm_start[k][1])
                opti.set_initial(self.pos_theta[k], warm_start[k][2])
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
        objective = 0

        # global target position
        x = self.pos_x[0] + gp[0]
        y = self.pos_y[0] + gp[1]

        for k in range(self.N + 1):
            if abs(gp[2]) > np.radians(10):
                angle_error = ca.atan2(ca.sin(gp[2] - self.pos_theta[k]), ca.cos(gp[2] - self.pos_theta[k])) + np.pi / 2
                objective += weights["fov"] * angle_error**2

            if len(polyhedron_planes) > 0:
                objective += 1 * self.compute_static_constraints(polyhedron_planes, k, plane_weights)

            objective += weights["pursuit"] * ((x - self.pos_x[k])**2 + (y - self.pos_y[k])**2)
            objective += weights["turn"] * self.vel_t[k]**2
            objective += weights["pursuit"] * self.vel_r[k]**2

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

    def compute_static_constraints(self, polyhedron_planes, k, plane_weights=None):
        """
        Compute soft constraints for the k-th step using k-th polyhedron.
        """
        static_constraint = 0.0
        if k >= len(polyhedron_planes):
            return static_constraint  # No constraint for this step

        planes = polyhedron_planes[k]

        for i, (n, c) in enumerate(planes):
            A, B = n[0], n[1]
            lhs = A * self.pos_x[k] + B * self.pos_y[k]
            violation = lhs - (c + 1)
            static_constraint += 1 * violation**2

        return static_constraint


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

    def shift_previous_trajectory(self, p_prev, N):
        """
        Shift the optimal trajectory computed at time t − 1, namely,
        q_{0:N} = [p^∗_{1:N} |t−1 , q_N ], where q_N is an extrapolation of
        the last two points, that is, q_N = 2p^∗_N |t−1 − p^∗_{N−1}|t−1
        """
        q_new = []
        # Shift each step forward
        for k in range(N):
            q_new.append(p_prev[k + 1])

        # Extrapolate the last point
        last_point = (2 * np.array(p_prev[-1]) - np.array(p_prev[-2]))
        q_new.append(tuple(last_point))

        return q_new
