#!/usr/bin/env python3

import math
from math import cos, sin

import numpy as np
from numpy.linalg import inv

from scipy.spatial.transform import Rotation

def E2Q(a, b, c):
    '''
    [ 오일러 - 쿼터니언 변환 ]
    a : yaw [psi]
    b : pitch [theta]
    c : roll [phi]
    '''
    qx = np.sin(a/2) * np.cos(b/2) * np.cos(c/2) - np.cos(a/2) * np.sin(b/2) * np.sin(c/2)
    qy = np.cos(a/2) * np.sin(b/2) * np.cos(c/2) + np.sin(a/2) * np.cos(b/2) * np.sin(c/2)
    qz = np.cos(a/2) * np.cos(b/2) * np.sin(c/2) - np.sin(a/2) * np.sin(b/2) * np.cos(c/2)
    qw = np.cos(a/2) * np.cos(b/2) * np.cos(c/2) + np.sin(a/2) * np.sin(b/2) * np.sin(c/2)
    
    quaternion = [qx, qy, qz, qw]
    
    return quaternion

def Q2E(quaternion):
    '''
    [ 쿼터니언 - 오일러 변환 ]
    a : qx
    b : qy
    c : qz
    d : qw
    '''
    a = quaternion.x 
    b = quaternion.y 
    c = quaternion.z 
    d = quaternion.w 
    
    t0    = +2.0 * (d*a + b*c)
    t1    = +1.0 - 2.0*(a*a + b*b)
    roll  = math.atan2(t0, t1)
    
    t2    = +2.0 * (d*b - c*a)
    t2    = +1.0 if t2 > +1.0 else t2
    t2    = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    
    t3    = +2.0 * (d*c + a*b)
    t4    = +1.0 - 2.0*(b*b + c*c)
    yaw   = math.atan2(t3, t4)
    
    euler = [roll, pitch, yaw]
    
    return euler, yaw

def gravity(a):
    '''
    [ 가속도 데이터의 축 기준 중력 가속도 벡터 생성 (중력 가속도 회전 변환) ]
    a : 쿼터니언 데이터
    '''
    g_vector = np.array([0, 0, 9.81])
    rotation = Rotation.from_quat(a)
    g        = rotation.apply(g_vector)
    
    return g

def rotate_vectorData(angle_x, angle_y, angle_z, a):
    '''
    [ 가속도 벡터 생성 (회전 변환) ]
    angle_x : x축 기준 회전
    angle_y : y축 기준 회전
    angle_z : z축 기준 회전
    Rt      : 회전 변환 행렬
    a       : 가속도 벡터
    '''
    Rt = axis_coordinate(angle_x, angle_y, angle_z)
    a = np.dot(Rt, a)
    
    return a

def rad2degree(a):
    '''
    [ 라디안 - 디그리 변환 ]
    a : [degree]
    '''
    radian = a * (180 / math.pi)
        
    return radian

def deg2radian(a):
    '''
    [ 디그리 - 라디안 변환 ]
    a : [radian]
    '''
    degree = a * (math.pi / 180)
    
    return degree
    
def degreeLimit(angle):
    '''
    [ 각도 제한 (자세 추정 EKF 활용) ]
    angle : rotation angle [float]
    '''
    if 180 < angle:
        angle = angle - 360
            
    elif angle < -180:
        angle = angle + 360
            
    return angle

def x_rotation(angular):
    '''
    [ X축 회전 변환 행렬 ]
    angular : x축 기준 회전 각도
    '''
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angular), -np.sin(angular)],
                   [0, np.sin(angular), np.cos(angular)]])
    
    return Rx

def y_rotation(angular):
    '''
    [ Y축 회전 변환 행렬 ]
    angular : y축 기준 회전 각도
    '''
    Ry = np.array([[np.cos(angular), 0, np.sin(angular)],
                   [0, 1, 0],
                   [-np.sin(angular), 0, np.cos(angular)]])
    
    return Ry

def z_rotation(angular):
    '''
    [ Z축 회전 변환 행렬 ]
    angular : z축 기준 회전 각도
    '''
    Rz = np.array([[np.cos(angular), -np.sin(angular), 0],
                   [np.sin(angular), np.cos(angular), 0],
                   [0, 0, 1]])
    
    return Rz

def axis_coordinate(x_angle, y_angle, z_angle):
    '''
    [ 3차원 회전 변환 행렬 계산 함수 ]
    x_angle : x축 기준 회전 각도
    y_angle : y축 기준 회전 각도
    z_angle : z축 기준 회전 각도
    '''
    Rx = x_rotation(x_angle)
    Ry = y_rotation(y_angle)
    Rz = z_rotation(z_angle)
    
    Rt = np.dot(Rz, np.dot(Ry, Rx))
        
    return Rt

def System_matrix(dt):
    '''
    [ 시스템 행렬 및 이산화 함수 ]
    '''
    A_ = np.array([[1, dt, 0, 0],
                  [0,  1, 0, 0],
                  [0, 0, 1, dt],
                  [0, 0, 0, 1]]) # 적분을 위한 시스템 행렬
        
    H = np.array([[1, 0, 0, 0],
                  [0, 0, 1, 0]])  # 측정 데이터 : 가속도 및 속도
        
    A = np.eye(4) + A_*dt # 시스템 행렬 이산화
        
    return A, H