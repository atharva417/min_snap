
import rospy
from math import sin, cos, tan, sqrt
import numpy as np
from math import factorial as f
from scipy.linalg import block_diag
from qpsolvers import solve_qp
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import warnings
import tensorflow as tf
from scipy.optimize import Bounds,minimize

from constrained_time_opt import min_snap

from std_msgs.msg import String, Float64
from sensor_msgs.msg import NavSatFix, Image,Imu
from mavros_msgs.srv import CommandTOL, SetMode, CommandBool
from mavros_msgs.msg import AttitudeTarget
from geometry_msgs.msg import PoseStamped, Pose, Point, Twist, TwistStamped
import math
from time import sleep

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore")

class DroneIn3D:
    
    def __init__(self):
        self.X=np.array([
            # x0, y1, z2, phi3, theta4, psi5, 
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            # x_dot6, y_dot7, z_dot8
            0.0, 0.0, 0.0])       
        self.g = 9.81

        self.gps_lat=0
        self.gps_long=0

        rospy.init_node('iris_drone', anonymous = True)

        #SUBSCRIBERS
        self.gps_subscriber =  rospy.Subscriber('/mavros/global_position/global', NavSatFix, self.gps_callback)
        self.get_pose_subscriber = rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self.get_pose)
        self.get_linear_vel=rospy.Subscriber('/mavros/local_position/velocity_local', TwistStamped, self.get_vel,)
        self.get_imu_data=rospy.Subscriber('/mavros/imu/data',Imu,self.get_euler_angles)

        #PUBLISHERS
        self.publish_pose = rospy.Publisher('/mavros/setpoint_position/local', PoseStamped,queue_size=1)
        self.publish_attitude_thrust=rospy.Publisher('/mavros/setpoint_raw/attitude', AttitudeTarget,queue_size=1)

        #SERVICES
        self.arm_service = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
        self.takeoff_service = rospy.ServiceProxy('/mavros/cmd/takeoff', CommandTOL)
        self.land_service = rospy.ServiceProxy('/mavros/cmd/land', CommandTOL)
        self.flight_mode_service = rospy.ServiceProxy('/mavros/set_mode', SetMode)
        rospy.loginfo('INIT')
        self.toggle_arm(True)
        self.takeoff(1.0)
        self.set_offboard_mode()
        #sleep(5)

        #rospy.loginfo('INIT')

    def gps_callback(self, data):
        self.gps_lat = data.latitude
        self.gps_long = data.longitude


    def get_pose(self, location_data):
        self.X[0] = location_data.pose.position.x
        self.X[1] = location_data.pose.position.y
        self.X[2] = location_data.pose.position.z


    def get_vel(self,vel_data):
        self.X[6]=	vel_data.twist.linear.x
        self.X[7]=	vel_data.twist.linear.y
        self.X[8]=	vel_data.twist.linear.z
        
        
    


    def get_euler_angles(self,orientaion_data):
        x=orientaion_data.orientation.x
        y=orientaion_data.orientation.y
        z=orientaion_data.orientation.z
        w=orientaion_data.orientation.w

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        self.X[3] = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        self.X[4] = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        self.X[5]= math.atan2(t3, t4)


    def toggle_arm(self, arm_bool):
        rospy.wait_for_service('/mavros/cmd/arming')
        try:
            self.arm_service(arm_bool)
        
        except rospy.ServiceException as e:
            rospy.loginfo("Service call failed: " %e)

    def takeoff(self, t_alt):
        self.gps_subscriber

        t_lat = self.gps_lat
        t_long = self.gps_long

        rospy.wait_for_service('/mavros/cmd/takeoff')
        try:
            self.takeoff_service(0.0,0,47.3977421,8.5455945,t_alt)
            rospy.loginfo('TAKEOFF')
        except rospy.ServiceException as e:
            rospy.loginfo("Service call failed: " %e)

    def set_offboard_mode(self):
        
        rate=rospy.Rate(20)
        #print('OFF')

        rospy.wait_for_service('/mavros/set_mode')
        PS = PoseStamped()

        PS.pose.position.x = 0
        PS.pose.position.y = 0
        PS.pose.position.z = 1
        for i in range(50):
            self.publish_pose.publish(PS)
            rate.sleep()
        try:
            self.flight_mode_service(0, 'OFFBOARD')
            rospy.loginfo('OFFBOARD')
            
        except rospy.ServiceException as e:
            rospy.loginfo("OFFBOARD Mode could not be set: " %e)


    def R(self):
    
        r_x = np.array([[1, 0, 0],
                        [0, cos(self.X[3]), -sin(self.X[3])],
                        [0, sin(self.X[3]), cos(self.X[3])]])

        r_y = np.array([[cos(self.X[4]), 0, sin(self.X[4])],
                        [0, 1, 0],
                        [-sin(self.X[4]), 0, cos(self.X[4])]])

        r_z = np.array([[cos(self.X[5]), -sin(self.X[5]), 0],
                        [sin(self.X[5]), cos(self.X[5]), 0],
                        [0,0,1]])

        r_yx = np.matmul(r_y, r_x)
        return np.matmul(r_z, r_yx)


class Controller:
    
    def __init__(self,
                z_k_p=1.0,
                z_k_d=1.0,
                x_k_p=1.0,
                x_k_d=1.0,
                y_k_p=1.0,
                y_k_d=1.0,
                k_p_roll=1.0,
                k_p_pitch=1.0,
                k_p_yaw=1.0):
        
        
        self.z_k_p = z_k_p
        self.z_k_d = z_k_d
        self.x_k_p = x_k_p
        self.x_k_d = x_k_d
        self.y_k_p = y_k_p
        self.y_k_d = y_k_d
        self.k_p_roll = k_p_roll
        self.k_p_pitch = k_p_pitch
        self.k_p_yaw = k_p_yaw
        self.k_p_p = k_p_p
        self.k_p_q = k_p_q
        self.k_p_r = k_p_r
       
        self.g = 9.81
    
    def altitude_controller(self,
                       z_target,
                       z_dot_target,
                       z_dot_dot_target,
                       z_actual,
                       z_dot_actual,
                       rot_mat):
    
        def pd(kp, kd, error, error_dot, target):
            p_term = kp * error
            d_term = kd * error_dot
            return p_term + d_term + target
        
        u_1_bar = pd(self.z_k_p, self.z_k_d, 
                    error     = z_target - z_actual, 
                    error_dot = z_dot_target - z_dot_actual,
                    target    = z_dot_dot_target)

        b_z = rot_mat[2,2]
        c=(u_1_bar - self.g)/b_z
        return c

    def lateral_controller(self,
                      x_target,
                      x_dot_target,
                      x_dot_dot_target,
                      x_actual,
                      x_dot_actual,
                      y_target,
                      y_dot_target,
                      y_dot_dot_target,
                      y_actual,
                      y_dot_actual,
                      c):
    
        def pd(kp, kd, error, error_dot, target):
            # Proportional and differential control terms
            p_term = kp * error
            d_term = kd * error_dot
            
            # Control command (with feed-forward term)
            return p_term + d_term + target
        
        # Determine errors
        x_err = x_target - x_actual
        y_err = y_target - y_actual
        x_err_dot = x_dot_target - x_dot_actual
        y_err_dot = y_dot_target - y_dot_actual
        
        # Apply the PD controller
        x_dot_dot_command = pd(self.x_k_p, self.x_k_d, x_err, x_err_dot, x_dot_dot_target)
        y_dot_dot_command = pd(self.y_k_p, self.y_k_d, y_err, y_err_dot, y_dot_dot_target)

        # Determine controlled values by normalizing with the collective thrust
        b_x_c = x_dot_dot_command / c
        b_y_c = y_dot_dot_command / c
        return b_x_c, b_y_c

    def roll_pitch_controller(self,
                          b_x_c_target,
                          b_y_c_target,
                          rot_mat):
    
        def p(kp, error):
            return kp * error
        
        b_x = rot_mat[0,2]
        b_y = rot_mat[1,2]
        
        b_x_commanded_dot = p(self.k_p_roll, error=b_x_c_target - b_x)
        b_y_commanded_dot = p(self.k_p_pitch, error=b_y_c_target - b_y)

        rot_mat1 = np.array([[rot_mat[1,0], -rot_mat[0,0]], 
                            [rot_mat[1,1], -rot_mat[0,1]]]) / rot_mat[2,2]

        rot_rate = np.matmul(rot_mat1, np.array([b_x_commanded_dot, b_y_commanded_dot]).T)
        p_c = rot_rate[0]
        q_c = rot_rate[1]

        return p_c, q_c

    
    def yaw_controller(self,
                   psi_target,
                   psi_actual):
    
        def p(kp, error):
            return kp * error

        return p(self.k_p_yaw, error=psi_target - psi_actual)

def actuate(x,y,z,v_test,v_min,v_max):

    #plt.figure(figsize=(10,5))
    #ax = plt.axes(projection ='3d')
    ms = min_snap(x,y,z,v_test,v_min,v_max)
    #ms.plot_test_case('r','Test Case Trajectory')
    ms.optimize()
    x_path,x_dot_path,x_dot_dot_path,y_path,y_dot_path,y_dot_dot_path,z_path,z_dot_path,z_dot_dot_path,psi_path=ms.get_trajectory_var()
    #ms.plot('g','Time Optimized Trajectory')
    #plt.legend()
    #plt.show()

    drone = DroneIn3D() 
    #sleep(2)   
    control_system = Controller(z_k_p=2.0,z_k_d=1.0,x_k_p=6.0,x_k_d=4.0,y_k_p=6.0,y_k_d=4.0,k_p_roll=8.0,k_p_pitch=8.0,
    k_p_yaw=8.0)
    iter=0 
    rate=rospy.Rate(100) 
    print(np.shape(z_path)[0]) 
    #for i in range(0,z_path.shape[0]):
    while (iter<np.shape(z_path)[0]):
        #print(np.shape(z_path)[0])
        rot_mat = drone.R()

        c = control_system.altitude_controller(z_path[iter],
                                            z_dot_path[iter],
                                            z_dot_dot_path[iter],
                                            drone.X[2],
                                            drone.X[8],
                                            rot_mat)

        b_x_c, b_y_c = control_system.lateral_controller(x_path[iter],
                                                        x_dot_path[iter],
                                                        x_dot_dot_path[iter],
                                                        drone.X[0],
                                                        drone.X[6],
                                                        y_path[iter],
                                                        y_dot_path[iter],
                                                        y_dot_dot_path[iter],
                                                        drone.X[1],
                                                        drone.X[7],
                                                        c) 

        #for j in range(5):
            
        rot_mat = drone.R()
        p_c, q_c = control_system.roll_pitch_controller(b_x_c,
                                            b_y_c,
                                            rot_mat)
        
        r_c = control_system.yaw_controller(psi_path[iter], 
                                drone.X[5])
        print(c,p_c,q_c,r_c)
        ATT=AttitudeTarget()
        ATT.body_rate.x=p_c
        ATT.body_rate.y=q_c
        ATT.body_rate.z=r_c
        ATT.thrust=c
        drone.publish_attitude_thrust.publish(ATT)    
        
        iter+=1
        rate.sleep()
            







#if __name__ == '__main__':
x = [0,0,-2,0]
y = [0,2,0,-2]
z = [1,1,2,0]
'''x=[2,-2,-2,1,4,6]
y=[4,0,-2,-4,-3,-5]
z=[10,7,5,3,0,1]'''
v_test = 2
v_min=0.1
v_max=15

actuate(x,y,z,v_test,v_min,v_max)