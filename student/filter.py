import numpy as np
import os
import sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params


class Filter:
    """Extended Kalman Filter for 3D multi-object tracking.
    
    State vector: [x, y, z, vx, vy, vz] — 3D position + velocity (constant velocity model)
    Supports both linear (LiDAR) and nonlinear (camera) measurement models.
    """

    def F(self):
        """State transition matrix for constant velocity model."""
        dt = params.dt
        return np.matrix([
            [1, 0, 0, dt, 0,  0 ],
            [0, 1, 0, 0,  dt, 0 ],
            [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1,  0,  0 ],
            [0, 0, 0, 0,  1,  0 ],
            [0, 0, 0, 0,  0,  1 ]
        ])

    def Q(self):
        """Process noise covariance matrix.
        
        Derived from Q = G * q * G^T where G = [dt²/2, dt] per axis.
        Accounts for unmodeled accelerations in the constant velocity assumption.
        """
        dt = params.dt
        q = params.q

        q1 = ((dt**4) / 4) * q   # position-position covariance
        q2 = ((dt**3) / 2) * q   # position-velocity covariance
        q4 = (dt**2) * q          # velocity-velocity covariance

        return np.matrix([
            [q1, 0,  0,  q2, 0,  0 ],
            [0,  q1, 0,  0,  q2, 0 ],
            [0,  0,  q1, 0,  0,  q2],
            [q2, 0,  0,  q4, 0,  0 ],
            [0,  q2, 0,  0,  q4, 0 ],
            [0,  0,  q2, 0,  0,  q4]
        ])

    def predict(self, track):
        """Propagate state and covariance forward one timestep."""
        F = self.F()
        track.set_x(F * track.x)
        track.set_P(F * track.P * F.T + self.Q())

    def gamma(self, track, meas):
        """Compute measurement residual: z - h(x)."""
        return meas.z - meas.sensor.get_hx(track.x)

    def S(self, track, meas, H):
        """Compute residual covariance: S = H*P*H^T + R."""
        return H * track.P * H.T + meas.R

    def update(self, track, meas):
        """Update state and covariance with a new measurement.
        
        Uses the Joseph form for covariance update to maintain 
        numerical stability and positive-definiteness:
        P = (I-KH)*P*(I-KH)^T + K*R*K^T
        """
        H = meas.sensor.get_H(track.x)
        gamma = self.gamma(track, meas)
        S = self.S(track, meas, H)

        K = track.P * H.T * np.linalg.inv(S)
        I = np.matrix(np.identity(params.dim_state))

        track.set_x(track.x + K * gamma)
        track.set_P((I - K * H) * track.P * (I - K * H).T + K * meas.R * K.T)
        track.update_attributes(meas)