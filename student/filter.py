# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Kalman filter class
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params 

class Filter:
    '''Kalman filter class'''
    def __init__(self):
        pass

    def F(self):
        ############
        # TODO Step 1: implement and return system matrix F
        ############
        dt = params.dt
        # State transition matrix for 3D constant velocity model
        F_matrix = np.matrix([[1, 0, 0, dt, 0,  0],
                              [0, 1, 0, 0,  dt, 0],
                              [0, 0, 1, 0,  0,  dt],
                              [0, 0, 0, 1,  0,  0],
                              [0, 0, 0, 0,  1,  0],
                              [0, 0, 0, 0,  0,  1]])

        return F_matrix
        
        ############
        # END student code
        ############ 

    def Q(self):
        ############
        # TODO Step 1: implement and return process noise covariance Q
        ############
        dt = params.dt
        q = params.q
        
        
        # g = [((dt**2)/2)  dt]
        # Q = g * q * g^T
        
        # Calculate individual elements of the Q matrix
        q1 = ((dt**4) / 4) * q
        q2 = ((dt**3) / 2) * q
        q3 = q2
        q4 = (dt**2)*q
        
        # Process noise covariance matrix
        Q_matrix = np.matrix([[q1, 0,  0,  q2, 0,  0],
                              [0,  q1, 0,  0,  q2, 0],
                              [0,  0,  q1, 0,  0,  q2],
                              [q3, 0,  0,  q4, 0,  0],
                              [0,  q3, 0,  0,  q4, 0],
                              [0,  0,  q3, 0,  0,  q4]])

        return Q_matrix
        
        ############
        # END student code
        ############ 

    def predict(self, track):
        ############
        # TODO Step 1: predict state x and estimation error covariance P to next timestep, save x and P in track
        ############

        F = self.F()
        Q = self.Q()
        
        # Predict state
        x_pred = F * track.x
        
        # Predict covariance
        P_pred = F * track.P * F.transpose() + Q
        
        
        # Save to track
        track.set_x(x_pred)
        track.set_P(P_pred)
        
        ############
        # END student code
        ############ 

    def gamma(self, track, meas):
        ############
        # TODO Step 1: calculate and return residual gamma
        ############
        # Get expected measurement h(x) from the sensor model
        hx = meas.sensor.get_hx(track.x)
        
        # Calculate residual: actual measurement - expected measurement
        residual = meas.z - hx
        return residual
       
        
        ############
        # END student code
        ############ 

    def S(self, track, meas, H):
        ############
        # TODO Step 1: calculate and return covariance of residual S
        ############
        S_matrix = H * track.P * H.transpose() + meas.R
        return S_matrix
   
        
        ############
        # END student code
        ############ 
    
    def update(self, track, meas):
        ############
        # TODO Step 1: update state x and covariance P with associated measurement, save x and P in track
        ############
        # Get matrices
        H = meas.sensor.get_H(track.x)
        gamma = self.gamma(track, meas)
        S = self.S(track, meas, H)
        
        # Calculate Kalman Gain
        K = track.P * H.transpose() * np.linalg.inv(S)
        
     
        
        # Update state
        x_upd = track.x + K * gamma
        
        # Update covariance using Identity matrix
        I = np.matrix(np.identity(params.dim_state))
        #P_upd = (I - K * H) * track.P
        P_upd = (I - K * H) * track.P * (I - K * H).T + K * meas.R * K.T   # using Joseph more stable form
        
        # Save updated state and covariance
        track.set_x(x_upd)
        track.set_P(P_upd)
        
        # Update other track attributes (like dimensions and yaw)
        track.update_attributes(meas)
        
        ############
        # END student code
        ############ 
    
    