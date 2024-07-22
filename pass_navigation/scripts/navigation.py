#!/usr/bin/env python3

import rclpy, logging, sys
import numpy as np

from rclpy.qos import QoSProfile, qos_profile_sensor_data
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import Header
from sensor_msgs.msg import Imu
from navigationTool import *
from pass_navigation.msg import NavigationType

class PoseGenerator(Node):
    def __init__(self):
        super().__init__('pose_generator')
        qos = QoSProfile(depth = 10)
        
        ''' SLAM 데이터에 대한 축변환 관련 각도 설정 파라미터(축 맞추기)
        *** (self.angle_x, self_angle_y, self.angle_z)
        *** default : (0, 0, 0) --> 축 그대로 사용
        (1) x축 기준 회전 각, y축 기준 회전 각, z축 기준 회전 각 설정
        (2) 각도 설정 시, 시계방향이 양수        
        '''
        self.declare_parameter('slamangle_x', None)
        self.declare_parameter('slamangle_y', None)
        self.declare_parameter('slamangle_z', None)
        
        ''' IMU 센서 데이터에 대한 축변환 관련 각도 설정 파라미터(축 맞추기)
        *** (self.angle_x, self_angle_y, self.angle_z)
        *** default : (0, 0, 0) --> 축 그대로 사용
        (1) x축 기준 회전 각, y축 기준 회전 각, z축 기준 회전 각 설정
        (2) 각도 설정 시, 시계방향이 양수
        '''
        self.declare_parameter('imuangle_x', None)
        self.declare_parameter('imuangle_y', None)
        self.declare_parameter('imuangle_z', None)
        
        ''' EKF 보정 후 그래프 형태 관련 파라미터
        Q 강화 : 칼만이득, 측정값 반영 비율 증가    / Q 약화 : 칼만 이득, 측정값 반영 비율 감소
        R 강화 : 측정값 반영비율 감소, 부드러운 궤적 / R 약화 : 측정값 반영 비율 증가
        '''
        self.declare_parameter('q_matrix', None)
        self.declare_parameter('r_matrix', None)
        
        ''' 중력가속도 벡터 생성을 위한 쿼터니언 행렬 선정
        (1) /imu/data 데이터의 쿼터니언을 통해 중력가속도 벡터 생성 시 True
        (2) /boat_odometry 데이터의 쿼터니언을 통해 중력가속도 벡터 생성 시 False
        '''
        self.declare_parameter('imu_usable', False)
        
        self.slamangle_x = self.get_parameter('slamangle_x').value
        self.slamangle_y = self.get_parameter('slamangle_y').value
        self.slamangle_z = self.get_parameter('slamangle_z').value
        self.imuangle_x  = self.get_parameter('imuangle_x').value
        self.imuangle_y  = self.get_parameter('imuangle_y').value
        self.imuangle_z  = self.get_parameter('imuangle_z').value
        self.q_matrix    = self.get_parameter('q_matrix').value
        self.r_matrix    = self.get_parameter('r_matrix').value
        self.imu_usable  = self.get_parameter('imu_usable').value
        
        self.subscribe_imu_data          = self.create_subscription(Imu, '/imu/data', self.imu_callbcak, qos)
        self.subscriber_ouster_imu       = self.create_subscription(Imu, '/ouster/imu', self.ouster_callback, qos_profile = qos_profile_sensor_data)
        self.subscriber_ouster_pose      = self.create_subscription(Odometry, '/boat_odometry', self.slam_callback, qos)

        self.publisher_ekf_odometry      = self.create_publisher(Odometry, '/ekf/odometry', qos)
        self.publihser_pass_navigation   = self.create_publisher(NavigationType, '/pass/navigation', qos)
       
       ## 변수 초기화
        self.first     = True
        self.prev_time = None
        self.KF        = False
        self.dt        = 0
        self.imu_yaw   = 0
        self.slam_yaw  = 0
        self.imu_q     = np.zeros(4)
        self.slam_q    = np.zeros(4)
        self.imu_g     = np.zeros(3)
        self.slam_g    = np.zeros(3)
        self.pose      = np.zeros(3)
        self.vel       = np.zeros(2)
        self.v_esti    = np.zeros(4)
        self.x_esti    = np.zeros(4)
        self.velocity  = np.zeros(2)
        self.position  = np.zeros(2)
 
        self.a         = np.array([0, 0, 0])
        self.r         = np.array([0, 0, 0])
        self.P         = 0.01*np.eye(4)
        self.v_P       = 0.01*np.eye(4)
        self.x_P       = 0.01*np.eye(4)
        
        self.Q         = self.q_matrix*np.eye(4)
        self.R         = self.r_matrix*np.eye(2)
        
    def imu_callbcak(self, data):
        ## IMU 쿼터니언(자세) 계측을 통해 선박의 중력가속도 벡터 생성해주는 옵션
        imu_quaternion = data.orientation
        _,self.imu_yaw = Q2E(imu_quaternion)
        self.imu_q     = np.array([imu_quaternion.x, imu_quaternion.y, imu_quaternion.z, imu_quaternion.w])
        
        self.imu_g     = gravity(self.imu_q)
    
    def ouster_callback(self, data):
        current = self.get_clock().now().nanoseconds
        
        if self.imu_usable == True:
            g_rotate = self.imu_g
            
        else:
            g_rotate = self.slam_g
            
        if self.prev_time is not None:
            self.dt  = (current - self.prev_time) / 1e9
            
            # 선형 가속도
            ouster_a = data.linear_acceleration
            a        = np.array([ouster_a.x, ouster_a.y, ouster_a.z])
            
            # 각속도
            ouster_r = data.angular_velocity
            r   = np.array([ouster_r.x, ouster_r.y, ouster_r.z])

            # 선형 가속도 벡터 - 중력 가속도 벡터 = 실 가속도 벡터
            self.a   = rotate_vectorData(self.imuangle_x, self.imuangle_y, self.imuangle_z, a - g_rotate)
            # 각속도 벡터
            self.r   = rotate_vectorData(self.imuangle_x, self.imuangle_y, self.imuangle_z, r)
            
        self.prev_time = current
        
        self.velocity_Estimation()
        self.position_Estimation()
        self.publisher()
    
    def slam_callback(self, data):
        slam_position   = data.pose.pose.position
        slam_velocity   = data.twist.twist.linear
        slam_quaternion = data.pose.pose.orientation
        _,self.slam_yaw = Q2E(slam_quaternion)
        # 중력 가속도 벡터 --> IMU 쿼터니언을 통해 회전 변환
        self.slam_q = [slam_quaternion.x, slam_quaternion.y, slam_quaternion.z, slam_quaternion.w]
        self.slam_g = gravity(self.slam_q)
            
        # Ouster Localization 좌표계 회전 변환 행렬
        velocity    = np.array([slam_velocity.x, slam_velocity.y, slam_velocity.z])
        position    = np.array([slam_position.x, slam_position.y, slam_position.z])

        self.vel    = rotate_vectorData(self.slamangle_x, self.slamangle_y, self.slamangle_z, velocity)   
        self.pose   = rotate_vectorData(self.slamangle_x, self.slamangle_y, self.slamangle_z, position)
        
    ''' [확장 칼만 필터 함수]
    measured : 측정 데이터 행렬
    x_esti   : 추정값
    x_pred   : 예측값
    P        : 오차 공분산
    dt       : 측정 데이터 초기화 시간
    '''
    def EKF(self, measured, esti, P, dt):
        # 시스템 행렬 이산화
        A, H   = System_matrix(dt)
        
        # 1. 예측값 산출 (속도/위치 & 오차 공분산)
        if not self.KF:
            # 속도 추정
            x_pred = A @ esti 
        else:
            # 위치 추정
            x_pred = self.x_esti + np.array([self.velocity[0], 0, self.velocity[2], 0])*self.dt 
        
        P_pred = A @ P @ A.T + self.Q 
        
        # 2. 칼만 이득 산출
        S      = H @ P_pred @ H.T + self.R
        K      = P_pred @ H.T @ np.linalg.inv(S)
        
        # 3. 추정값 산출 (속도/위치 & 오차 공분산)
        y      = measured - H @ x_pred
        esti   = x_pred + K @ y
        P      = P_pred - K @ H @ P_pred
        
        return esti, P

    ## 속도 추정
    def velocity_Estimation(self):
        z_k     = np.array([self.vel[0], self.vel[1]])
        if z_k is None:
            self.v_esti, self.v_P = np.zeros(4), 0.01*np.eye(4)
        else:
            self.v_esti, self.v_P = self.EKF(z_k, self.v_esti, self.v_P, self.dt)
        
        self.velocity = np.array([self.v_esti[0], self.v_esti[2]])
        
    ## 위치 추정
    def position_Estimation(self):
        z_k     = np.array([self.pose[0], self.pose[1]])
        if z_k is None:
            self.x_esti, self.x_P = np.zeros(4), 0.01*np.eye(4)
        else:
            self.x_esti, self.x_P = self.EKF(z_k, self.x_esti, self.x_P, self.dt)
            
        self.position = np.array([self.x_esti[0], self.x_esti[2]])
        
    ## 데이터 Publish
    def publisher(self):
        ekf_odometry                             = Odometry()
        ekf_odometry.header                      = Header()
        ekf_odometry.header.stamp                = self.get_clock().now().to_msg()
        ekf_odometry.header.frame_id             = 'ekf_odom'
        ekf_odometry.child_frame_id              = 'base_link'
        ekf_position                             = Point(x=float(self.position[0]), y=float(self.position[1]), z=float(0))
        ekf_velocity                             = Vector3(x=float(self.velocity[0]), y=float(self.velocity[1]), z=float(0))
        ekf_odometry.pose.pose.position          = ekf_position
        ekf_odometry.twist.twist.linear          = ekf_velocity
        ekf_odometry.twist.twist.angular.z       = float(self.r[2])
        ekf_quaternion = self.imu_q if self.imu_usable else self.slam_q
        ekf_odometry.pose.pose.orientation.x = float(ekf_quaternion[0])
        ekf_odometry.pose.pose.orientation.y = float(ekf_quaternion[1])
        ekf_odometry.pose.pose.orientation.z = float(ekf_quaternion[2])
        ekf_odometry.pose.pose.orientation.w = float(ekf_quaternion[3])            
        self.publisher_ekf_odometry.publish(ekf_odometry)
        
        pass_navigation                          = NavigationType()
        pass_navigation.x                        = float(self.position[0])
        pass_navigation.y                        = float(self.position[1])
        psi = self.imu_yaw if self.imu_usable else self.slam_yaw
        pass_navigation.psi                      = float(rad2degree(psi))
        pass_navigation.u                        = float(self.velocity[0])
        pass_navigation.v                        = float(self.velocity[1])
        pass_navigation.r                        = float(self.r[2])
        self.publihser_pass_navigation.publish(pass_navigation)
        
def main(args=None):
    rclpy.init(args=args)
    ekf_node = PoseGenerator()
    rclpy.spin(ekf_node)
    ekf_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()