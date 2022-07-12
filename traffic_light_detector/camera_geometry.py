import numpy as np
from math import cos, sin, pi, tan
from cutils import CUtils


def rotate_x(angle):
    R = np.mat([[ 1,         0,           0],
                 [ 0, cos(angle), -sin(angle) ],
                 [ 0, sin(angle),  cos(angle) ]])
    return R


def rotate_y(angle):
    R = np.mat([[ cos(angle), 0,  sin(angle) ],
                 [ 0,         1,          0 ],
                 [-sin(angle), 0,  cos(angle) ]])
    return R


def rotate_z(angle):
    R = np.mat([[ cos(angle), -sin(angle), 0 ],
                 [ sin(angle),  cos(angle), 0 ],
                 [         0,          0, 1 ]])
    return R


def to_rot(r):
    Rx = np.mat([[ 1,         0,           0],
                 [ 0, cos(r[0]), -sin(r[0]) ],
                 [ 0, sin(r[0]),  cos(r[0]) ]])

    Ry = np.mat([[ cos(r[1]), 0,  sin(r[1]) ],
                 [ 0,         1,          0 ],
                 [-sin(r[1]), 0,  cos(r[1]) ]])

    Rz = np.mat([[ cos(r[2]), -sin(r[2]), 0 ],
                 [ sin(r[2]),  cos(r[2]), 0 ],
                 [         0,          0, 1 ]])

    return Rz*Ry*Rx


class CameraGeometry:
    def __init__(self, camera_parameters):
        # Camera parameters
        self.cam_height = camera_parameters['z']
        self.cam_x_pos = camera_parameters['x']
        self.cam_y_pos = camera_parameters['y']

        self.camera_width = camera_parameters['width']
        self.camera_height = camera_parameters['height']

        self.camera_fov = camera_parameters['fov']

        # Calculate Intrinsic Matrix
        f = self.camera_width  * 0.5 / (2 * tan(self.camera_fov * pi / 360))
        center_x = self.camera_width / 2.0
        center_y = self.camera_height / 2.0

        intrinsic_matrix = np.array([[f, 0, center_x],
                                     [0, f, center_y],
                                     [0, 0, 1]])

        self.inv_intrinsic_matrix = np.linalg.inv(intrinsic_matrix)

        # Rotation matrix to align image frame to camera frame
        rotation_image_camera_frame = np.dot(rotate_z(-90 * pi / 180), rotate_x(-90 * pi / 180))

        image_camera_frame = np.zeros((4, 4))
        image_camera_frame[:3, :3] = rotation_image_camera_frame
        image_camera_frame[:, -1] = [0, 0, 0, 1]

        # Lambda Function for transformation of image frame in camera frame
        self.image_to_camera_frame = lambda object_camera_frame: np.dot(image_camera_frame, object_camera_frame)

    def get_traffic_light_fences(self, depth_data, current_x, current_y, current_yaw, x, y):
        """Obtain traffic light fences using data from depth camera

        Args:
            depth_data: data of depth camera
            current_x (float): x position of the vehicle in meters
            current_y (float): y position of the vehicle in meters
            current_yaw (float): yaw of the vehicle
            x (int): row index of first pixel that contains traffic light
            y (int): column index of first pixel that contains traffic light

        Returns:
            traffic_light_fences([x0, y0, x1, y1]): fence calculated that indicates the stop line
        """

        current_x -= self.cam_x_pos
        current_y -= self.cam_y_pos

        # From pixel to waypoint
        pixel = [x, y, 1]
        pixel = np.reshape(pixel, (3, 1))

        # Projection Pixel to Image Frame
        depth = depth_data[y][x] * 1000  # Consider depth in meters

        image_frame_vect = np.dot(self.inv_intrinsic_matrix, pixel) * depth

        # Create extended vector
        image_frame_vect_extended = np.zeros((4, 1))
        image_frame_vect_extended[:3] = image_frame_vect
        image_frame_vect_extended[-1] = 1

        # Projection Camera to Vehicle Frame
        camera_frame = self.image_to_camera_frame(image_frame_vect_extended)
        camera_frame = camera_frame[:3]
        camera_frame = np.asarray(np.reshape(camera_frame, (1, 3)))

        camera_frame_extended = np.zeros((4, 1))
        camera_frame_extended[:3] = camera_frame.T
        camera_frame_extended[-1] = 1

        camera_to_vehicle_frame = np.zeros((4, 4))
        camera_to_vehicle_frame[:3, :3] = to_rot([0, 0, 0])
        camera_to_vehicle_frame[:, -1] = [self.cam_x_pos, self.cam_y_pos, self.cam_height, 1]

        vehicle_frame = np.dot(camera_to_vehicle_frame, camera_frame_extended)
        vehicle_frame = vehicle_frame[:3]
        vehicle_frame = np.asarray(np.reshape(vehicle_frame, (1, 3)))

        stopsign_data = CUtils()
        if (int(round(abs(cos(current_yaw))))):
            stopsign_data.create_var('x', vehicle_frame[0][0] - self.cam_x_pos)
            stopsign_data.create_var('y', vehicle_frame[0][1])
        else:
            stopsign_data.create_var('x', vehicle_frame[0][1])
            stopsign_data.create_var('y', vehicle_frame[0][0] - self.cam_x_pos)
        stopsign_data.create_var('z', vehicle_frame[0][2])

        # obtain traffic light fence points for LP
        x = stopsign_data.x
        y = stopsign_data.y

        spos = np.array([
            [current_x - 5 * int(round(abs(sin(current_yaw)))), current_x + 5 * int(round(abs(sin(current_yaw))))],
            [current_y - 5 * int(round(abs(cos(current_yaw)))), current_y + 5 * int(round(abs(cos(current_yaw))))]])
        spos_shift = np.array([
            [x, x],
            [y, y]])

        if np.sign(round(np.cos(current_yaw))) > 0:  # the car is moving on the positive x
            spos = np.add(spos, spos_shift)

        elif np.sign(round(np.cos(current_yaw))) < 0:  # the car is moving on the negative x
            spos = np.subtract(spos, spos_shift)

        else:
            if np.sign(round(np.sin(current_yaw))) > 0:  # the car is moving on the positive y
                spos = np.add(spos, spos_shift)
            else:  # the car is moving on the positive y
                spos = np.subtract(spos, spos_shift)

        return [spos[0, 0], spos[1, 0], spos[0, 1], spos[1, 1]]
