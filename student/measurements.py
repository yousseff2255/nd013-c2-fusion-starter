import numpy as np
import os
import sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params


class Sensor:
    """Sensor model supporting LiDAR (linear) and camera (nonlinear) measurement functions.
    
    Handles coordinate transformations between vehicle space and sensor space,
    FOV checks, and Jacobian computation for the EKF camera update step.
    """

    def __init__(self, name, calib):
        self.name = name

        if name == 'lidar':
            self.dim_meas = 3
            # LiDAR detections already arrive in vehicle coordinates
            self.sens_to_veh = np.matrix(np.identity(4))
            self.fov = [-np.pi/2, np.pi/2]

        elif name == 'camera':
            self.dim_meas = 2
            # Extrinsic matrix: camera pose in vehicle coordinate frame
            self.sens_to_veh = np.matrix(calib.extrinsic.transform).reshape(4, 4)
            self.f_i = calib.intrinsic[0]  # focal length (u-axis)
            self.f_j = calib.intrinsic[1]  # focal length (v-axis)
            self.c_i = calib.intrinsic[2]  # principal point u
            self.c_j = calib.intrinsic[3]  # principal point v
            self.fov = [-0.35, 0.35]       # trimmed FOV to remove inaccurate boundary detections

        self.veh_to_sens = np.linalg.inv(self.sens_to_veh)

    def in_fov(self, x):
        """Check if a state vector x is within this sensor's field of view."""
        x_veh = np.ones((4, 1))
        x_veh[0:3] = x[0:3]
        x_sens = self.veh_to_sens * x_veh

        if x_sens[0] > 0:  # object must be in front of sensor
            alpha = np.arctan2(x_sens[1].item(), x_sens[0].item())
            if self.fov[0] <= alpha <= self.fov[1]:
                return True
        return False

    def get_hx(self, x):
        """Compute expected measurement h(x) for the current state estimate."""
        pos_veh = np.ones((4, 1))
        pos_veh[0:3] = x[0:3]
        pos_sens = self.veh_to_sens * pos_veh

        if self.name == 'lidar':
            return pos_sens[0:3]

        elif self.name == 'camera':
            if pos_sens[0] == 0:
                raise ValueError("Division by zero in camera projection — object on image plane")
            z = np.matrix(np.zeros((2, 1)))
            z[0] = self.f_i * pos_sens[1] / pos_sens[0] + self.c_i
            z[1] = self.f_j * pos_sens[2] / pos_sens[0] + self.c_j
            return z

    def get_H(self, x):
        """Compute measurement Jacobian H at current state x.
        
        For LiDAR: H is linear (just rotation matrix extracting position).
        For camera: H is the analytical Jacobian of the perspective projection,
        linearizing h(x) around the current estimate for the EKF update.
        """
        H = np.matrix(np.zeros((self.dim_meas, params.dim_state)))
        R = self.veh_to_sens[0:3, 0:3]
        T = self.veh_to_sens[0:3, 3]

        if self.name == 'lidar':
            H[0:3, 0:3] = R

        elif self.name == 'camera':
            px = R[0,0]*x[0] + R[0,1]*x[1] + R[0,2]*x[2] + T[0]
            if px == 0:
                raise NameError('Jacobian undefined — object projects onto camera center')

            H[0,0] = self.f_i * (-R[1,0]/px + R[0,0]*(R[1,0]*x[0] + R[1,1]*x[1] + R[1,2]*x[2] + T[1]) / px**2)
            H[1,0] = self.f_j * (-R[2,0]/px + R[0,0]*(R[2,0]*x[0] + R[2,1]*x[1] + R[2,2]*x[2] + T[2]) / px**2)
            H[0,1] = self.f_i * (-R[1,1]/px + R[0,1]*(R[1,0]*x[0] + R[1,1]*x[1] + R[1,2]*x[2] + T[1]) / px**2)
            H[1,1] = self.f_j * (-R[2,1]/px + R[0,1]*(R[2,0]*x[0] + R[2,1]*x[1] + R[2,2]*x[2] + T[2]) / px**2)
            H[0,2] = self.f_i * (-R[1,2]/px + R[0,2]*(R[1,0]*x[0] + R[1,1]*x[1] + R[1,2]*x[2] + T[1]) / px**2)
            H[1,2] = self.f_j * (-R[2,2]/px + R[0,2]*(R[2,0]*x[0] + R[2,1]*x[1] + R[2,2]*x[2] + T[2]) / px**2)

        return H

    def generate_measurement(self, num_frame, z, meas_list):
        """Create a new measurement object and append to the measurement list."""
        meas_list.append(Measurement(num_frame, z, self))
        return meas_list


class Measurement:
    """Single sensor measurement with value vector z and noise covariance R."""

    def __init__(self, num_frame, z, sensor):
        self.t = (num_frame - 1) * params.dt
        self.sensor = sensor

        if sensor.name == 'lidar':
            self.z = np.matrix(np.zeros((sensor.dim_meas, 1)))
            self.z[0] = float(z[0])
            self.z[1] = float(z[1])
            self.z[2] = float(z[2])
            self.R = np.matrix([
                [params.sigma_lidar_x**2, 0, 0],
                [0, params.sigma_lidar_y**2, 0],
                [0, 0, params.sigma_lidar_z**2]
            ])
            self.width  = float(z[4])
            self.length = float(z[5])
            self.height = float(z[3])
            self.yaw    = float(z[6])

        elif sensor.name == 'camera':
            self.z = np.matrix(np.zeros((sensor.dim_meas, 1)))
            self.z[0] = float(z[0])
            self.z[1] = float(z[1])
            self.R = np.matrix([
                [params.sigma_cam_i**2, 0],
                [0, params.sigma_cam_j**2]
            ])