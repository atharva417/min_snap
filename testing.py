#!/usr/bin/env python2

import rospy
from std_msgs.msg import String, Float64
from sensor_msgs.msg import NavSatFix, Image,Imu
from mavros_msgs.srv import CommandTOL, SetMode, CommandBool
from mavros_msgs.msg import AttitudeTarget
from geometry_msgs.msg import PoseStamped, Pose, Point, Twist, TwistStamped
import math
from time import sleep


class FLIGHT_CONTROLLER:

	def __init__(self):

		#DATA
		self.gps_lat = 0
		self.gps_long = 0

		self.curr_x = 0
		self.curr_y = 0
		self.curr_z = 0

		self.x_vel=10
		self.y_vel=4
		self.z_vel=5

		self.set_x = 0
		self.set_y = 0
		self.set_z = 0

		self.pitch=0
		self.roll=0
		self.yaw=0
		

		self.delta = 0.05
		
		self.waypoint_number = 0


		#NODE
		rospy.init_node('iris_drone', anonymous = True)

		#SUBSCRIBERS
		self.gps_subscriber =  rospy.Subscriber('/mavros/global_position/global', NavSatFix, self.gps_callback)
		self.get_pose_subscriber = rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self.get_pose)
		self.get_linear_vel=rospy.Subscriber('/mavros/local_position/velocity_local', TwistStamped, self.get_vel,)
		self.get_imu_data=rospy.Subscriber('/mavros/imu/data',Imu,self.get_euler_angles)

		#PUBLISHERS
		self.publish_pose = rospy.Publisher('/mavros/setpoint_position/local', PoseStamped,queue_size=10)
		self.publish_attitude_thrust=rospy.Publisher('/mavros/setpoint_raw/attitude', AttitudeTarget,queue_size=0)

		#SERVICES
		self.arm_service = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
		self.takeoff_service = rospy.ServiceProxy('/mavros/cmd/takeoff', CommandTOL)
		self.land_service = rospy.ServiceProxy('/mavros/cmd/land', CommandTOL)
		self.flight_mode_service = rospy.ServiceProxy('/mavros/set_mode', SetMode)

		rospy.loginfo('INIT')

	#MODE SETUP

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



	def land(self, l_alt):

		self.gps_subscriber

		l_lat = self.gps_lat
		l_long = self.gps_long

		rospy.wait_for_service('/mavros/cmd/land')
		try:
			self.land_service(0.0, 0.0, l_lat, l_long, l_alt)
			rospy.loginfo("LANDING")

		except rospy.ServiceException as e:
			rospy.loginfo("Service call failed: " %e)


	def set_offboard_mode(self):
		
		rate=rospy.Rate(20)
		#print('OFF')

		rospy.wait_for_service('/mavros/set_mode')
		PS = PoseStamped()

		PS.pose.position.x = 0
		PS.pose.position.y = 0
		PS.pose.position.z = 0
		
		for i in range(50):
			self.publish_pose.publish(PS)
			
			rate.sleep()
		print('done')
		try:
			self.flight_mode_service(0, 'OFFBOARD')
			rospy.loginfo('OFFBOARD')
			
		except rospy.ServiceException as e:
			rospy.loginfo("OFFBOARD Mode could not be set: " %e)





	#CALLBACKS

	def gps_callback(self, data):
		self.gps_lat = data.latitude
		self.gps_long = data.longitude


	def get_pose(self, location_data):
		self.curr_x = location_data.pose.position.x
		self.curr_y = location_data.pose.position.y
		self.curr_z = location_data.pose.position.z


	def get_vel(self,vel_data):
		self.x_vel=	vel_data.twist.linear.x
		self.y_vel=	vel_data.twist.linear.y
		self.z_vel=	vel_data.twist.linear.z
		
		
	


	def get_euler_angles(self,orientaion_data):
		x=orientaion_data.orientation.x
		y=orientaion_data.orientation.y
		z=orientaion_data.orientation.z
		w=orientaion_data.orientation.w

		t0 = +2.0 * (w * x + y * z)
		t1 = +1.0 - 2.0 * (x * x + y * y)
		self.roll = math.atan2(t0, t1)

		t2 = +2.0 * (w * y - z * x)
		t2 = +1.0 if t2 > +1.0 else t2
		t2 = -1.0 if t2 < -1.0 else t2
		self.pitch = math.asin(t2)

		t3 = +2.0 * (w * z + x * y)
		t4 = +1.0 - 2.0 * (y * y + z * z)
		self.yaw= math.atan2(t3, t4)

		




	#PUBLISHERS

	def set_pose(self):

		update_rate = rospy.Rate(20)
		PS = PoseStamped()

		PS.pose.position.x = self.set_x
		PS.pose.position.y = self.set_y
		PS.pose.position.z = self.set_z

		PS.pose.orientation.x = 0
		PS.pose.orientation.y = 0
		PS.pose.orientation.z = 0.707
		PS.pose.orientation.w = 0.707

		distance =math.sqrt((self.set_x - self.curr_x)**2 + (self.set_y - self.curr_y)**2 + (self.set_z - self.curr_z)**2)

		while (distance > self.delta): #and (abs(self.set_z - self.curr_z) > self.delta_z):

			self.publish_pose.publish(PS)
			self.get_pose_subscriber
			distance =math.sqrt((self.set_x - self.curr_x)**2 + (self.set_y - self.curr_y)**2 + (self.set_z - self.curr_z)**2)
			self.rgb_flag = 0
			self.depth_flag = 0
			update_rate.sleep()

		self.waypoint_number = self.waypoint_number + 1
		#self.depth_flag = 1
		#rospy.loginfo('WAYPOINT REACHED: ' + str(self.waypoint_number))
		
	

		
	
	def move_to(self,x2,y2,z2):
	
		
		a=self.curr_x
		b=self.curr_y
		c=self.curr_z
		self.set_waypoints((x2+a)/2,(y2+b)/2,(z2+c)/2)
		self.set_pose()
		self.set_waypoints(x2,y2,z2)
		self.set_pose()

		#print('\n',self.roll, self.pitch, self.yaw)
		#print(self.x_vel,self.y_vel,self.z_vel)
		
		sleep(2)
		self.set_waypoints(x2,y2,z2)
		self.set_pose()

		#print('\n',self.roll, self.pitch, self.yaw)
		#print(self.x_vel,self.y_vel,self.z_vel)



	#MISSION CONTself.roll
	def set_waypoints(self, temp_x, temp_y, temp_z):

		self.set_x = temp_x
		self.set_y = temp_y
		self.set_z = temp_z



	def test_control(self):

		self.toggle_arm(True)
		#self.takeoff(1.0)
		self.set_offboard_mode()

		self.move_to(0,0,3)
		#sleep(2)
		'''
		self.move_to(-2,2,3.3)
		sleep(2)
		self.move_to(0,4,3.3)
		sleep(2)
		self.move_to(2,2,3.3)
		sleep(2)'''
        ############
		print('Starting')
		rate = rospy.Rate(20)
		sr = AttitudeTarget()
		sr.type_mask = 135
		sr.thrust = 0.9
		for i in range(20):
			print('stg 1')
			self.publish_attitude_thrust.publish(sr)
			rate.sleep()
		print('Stage 1 done')
		sr.type_mask = 134
		sr.body_rate.x = 1500.0
		sr.body_rate.y = 0.0
		sr.body_rate.z = 0.0
		sr.thrust = 0.5
		for i in range(20):
			print('stg 2')
			self.publish_attitude_thrust.publish(sr)
			rate.sleep()
		print('Stage 2 done')
		sr.type_mask = 134
		sr.body_rate.x = -1500.0
		sr.body_rate.y = 0.0
		sr.body_rate.z = 0.0
		sr.thrust = 0.5
		for i in range(5):
			print('final')
			self.publish_attitude_thrust.publish(sr)
			rate.sleep()
		print('Roll Complete!!')		
		



		

	



if __name__ == '__main__':

	try:
		iris_controller = FLIGHT_CONTROLLER()
		
		iris_controller.test_control()
		rospy.spin()


	except rospy.ROSInterruptException:
		pass