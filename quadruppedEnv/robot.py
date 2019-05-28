

import numpy as np
import gym
from gym import wrappers
import argparse
import pprint as pp
import roboschool
import math
from gym.envs.registration import register
from roboschool.scene_abstract import cpp_household
#from roboschool.scene_stadium import SinglePlayerStadiumScene
from roboschool.gym_forward_walker import RoboschoolForwardWalker
from roboschool.gym_mujoco_xml_env import RoboschoolMujocoXmlEnv
from roboschool.gym_urdf_robot_env import RoboschoolUrdfEnv
from roboschool.scene_stadium import SinglePlayerStadiumScene
import gym, gym.spaces, gym.utils, gym.utils.seeding
import numpy as np
import os, sys
from collections import deque
from numpy import arccos, array, dot, pi, cross
from numpy.linalg import det, norm
from importlib import import_module

from quadruppedEnv import settings

try:
    settings.robotLibAvailable = 1
    robotlib = os.path.realpath("/Users/pavlogryb/Dropbox/Projects/Robot/Robot/pyRobotLib")
    sys.path.insert(0, robotlib)
    import pyRobotLib
    #print (pyRobotLib.greet())
except ImportError:
    settings.robotLibAvailable = 0
    settings.robotNN = 1
    print("robot lib not fouund")
    pass

class RoboschoolForwardWalkerMujocoXML(RoboschoolForwardWalker, RoboschoolUrdfEnv):
    def __init__(self, fn, robot_name, action_dim, power):
        self.velObservations = 0
        sensorsObservations = 4 #4 legs from 0 or 1
        rotationObservation = 3 #roll pith yaw
        sinCosToTarget = 2 # sin cos from angle to target
        servoObservation = 12 # rel angles
        numObservation = sensorsObservations+rotationObservation+sinCosToTarget+servoObservation+self.velObservations

        if hasattr(settings,"history1Len")==False:
            settings.history1Len = 0.0
        if hasattr(settings,"history2Len")==False:
            settings.history2Len = 0.0
        if hasattr(settings,"history3Len")==False:
            settings.history3Len = 0.0


        baseObservations = numObservation
        if settings.history1Len>0.0:
            numObservation += baseObservations
        if settings.history2Len>0.0:
            numObservation += baseObservations
        if settings.history3Len>0.0:
            numObservation += baseObservations
        RoboschoolUrdfEnv.__init__(self, fn, robot_name, action_dim, numObservation,
            fixed_base=False,
            self_collision=False)
        RoboschoolForwardWalker.__init__(self, power)
        self.servo_kp = 0.3
        self.servo_kd = 0.9
        self.servo_maxTorque = 2.0
        self.time = 0.0
        self.max_servo_speed = math.radians(180.0)
        self.physics_time_step = 1.0/240.0
        self.walk_target_x = 1000.0 # 1km  meters
        self.walk_target_z = 0.15
        self.debugStats = 0
        self.movement_dir = [1,0,0]
        self.robotLibOn = 0
        if settings.robotLibAvailable and hasattr(settings,"robotNN") and settings.robotNN==0:
            self.robotLibOn = 1
            self.robotLib = pyRobotLib.Robot()
            self.robotLib.load("/Users/pavlogryb/Dropbox/Projects/Robot/Robot/Robot/")
            print(os.getpid())


    def create_single_player_scene(self):
        return SinglePlayerStadiumScene(gravity=9.8, timestep=self.physics_time_step, frame_skip=1)

    def calc_state(self):
        robot_state = self.calc_state_single()
        if settings.history1Len>0.0:
            self.history1.append(robot_state)
        if settings.history2Len>0.0:
            self.history2.append(robot_state)
        if settings.history3Len>0.0:
            self.history3.append(robot_state);
        self.last_state = robot_state
        if settings.history1Len>0.0:
            robot_state= np.append(robot_state,self.history1[0])
        if settings.history2Len>0.0:
            robot_state= np.append(robot_state,self.history2[0])
        if settings.history3Len>0.0:
            robot_state= np.append(robot_state,self.history3[0])
        return robot_state

 

    def calc_state_single(self):
        body_pose = self.robot_body.pose()
        servo_angles =  np.array([], dtype=np.float32);
        for j in self.ordered_joints:
            #print(j.current_relative_position(),j.name)
            servo_angles = np.append(servo_angles,j.servo_target)

        jointsData = np.array([j.current_relative_position() for j in self.ordered_joints], dtype=np.float32).flatten()
        j  = servo_angles;
        #print(j)
        # even elements [0::2] position, scaled to -1..+1 between limits
        # odd elements  [1::2] angular speed, scaled to show -1..+1
        self.joint_speeds = jointsData[1::2]
        self.joints_at_limit = np.count_nonzero(np.abs(j[0::2]) > 0.99)

        #parts_xyz = np.array( [p.pose().xyz() for p in self.parts.values()] ).flatten()
        curBodyXYZ = (body_pose.xyz()[0], body_pose.xyz()[1], body_pose.xyz()[2])
        self.prev_body_xyz = self.body_xyz;
        self.body_xyz = curBodyXYZ
        self.body_rpy = body_pose.rpy()
        z = self.body_xyz[2]
        r, p, yaw = self.body_rpy
        if self.initial_z is None:
            self.initial_z = z
        self.movement_per_frame = [self.body_xyz[0] - self.prev_body_xyz[0],self.body_xyz[1] - self.prev_body_xyz[1], self.body_xyz[2] - self.prev_body_xyz[2]]
        self.walk_target_theta = np.arctan2( self.walk_target_y - self.body_xyz[1], self.walk_target_x - self.body_xyz[0] )
        self.vecToTarget = [self.walk_target_x - self.body_xyz[0],self.walk_target_y - self.body_xyz[1], self.walk_target_z - self.body_xyz[2]]
        self.walk_target_dist  = np.linalg.norm( self.vecToTarget )
        self.angle_to_target = self.walk_target_theta - yaw

        self.rot_minus_yaw = np.array(
            [[np.cos(-yaw), -np.sin(-yaw), 0],
             [np.sin(-yaw),  np.cos(-yaw), 0],
             [           0,             0, 1]]
            )
        #vx, vy, vz = np.dot(self.rot_minus_yaw, self.robot_body.speed())  # rotate speed back to body point of view

        vx, vy, vz = self.robot_body.speed()

        observations = np.array([
            r, p, yaw,
            np.sin(self.angle_to_target), np.cos(self.angle_to_target)
            ], dtype=np.float32)
        observations = np.append(observations,self.feet_contact)
        #if( self.feet_contact[0]>0 ):
        #    print("")
        # 0.3 is just scaling typical speed into -1..+1, no physical sense here
        body_vel = np.array([0.3*vx, 0.3*vy, 0.3*vz], dtype=np.float32)
        if self.velObservations>0:
            observations = np.append(observations,body_vel)

        observations = np.append(observations,j)

        robot_state = np.clip( observations, -5, +5)
        #print(robot_state)
        return robot_state

    def calc_potential(self):
        # progress in potential field is speed*dt, typical speed is about 2-3 meter per second, this potential will change 2-3 per frame (not per second),
        # all rewards have rew/frame units and close to 1.0
        #return - self.walk_target_dist
        return - self.walk_target_dist / self.physics_time_step

    def step(self, a):

        if self.robotLibOn:
            #print(self.last_state)
            if self.frame==50:
                self.robotLib.executeCommand("cmd#initIdle")
            if self.frame==200:
                self.robotLib.executeCommand("cmd#testAction")
            actions = self.robotLib.getActions(self.last_state.tolist(),self.physics_time_step)
            a = np.asarray(actions)


        if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then step() for all robots with the same actions
            self.apply_action(a)
            self.scene.global_step()

        state = self.calc_state()  # also calculates self.joints_at_limit

        globalMultiplier = 10.0

        alive_multiplier = 0.1/1.0
        alive = float(self.alive_bonus(state))
        if alive<0 and self.frame==0:
            print("bad transition")

        alive*=alive_multiplier
        alive *=globalMultiplier

        if alive<0:
            alive = -100.0

        self.reward_alive.append(alive)
        done = alive < 0
        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True

        potential_old = self.potential
        self.potential = self.calc_potential()
        progress = float(self.potential - potential_old)
        '''
        if progress<0:
            progress = -1.0
        else:
            progress = 1.0
        '''
        
        if self.walk_target_dist>0.1 and self.prev_body_xyz is not None:
            vecStep = [self.prev_body_xyz[0] - self.body_xyz[0],self.prev_body_xyz[1] - self.body_xyz[1]] #, self.prev_body_xyz[2] - self.body_xyz[2]]
            vecMovement = np.linalg.norm( vecStep )
            minSpeedToPunishPerSec = 0.01 # 1 cm
            minMovePerFrame = minSpeedToPunishPerSec/self.numStepsPerSecond
            #if vecMovement<minMovePerFrame:
            #    progress = -1
            
        feet_collision_cost = 0.0
        for i,f in enumerate(self.feet):
            contact_names = set(x.name for x in f.contact_list())
            #print("CONTACT OF '%s' WITH %s" % (f.name, ",".join(contact_names)) )
            self.feet_contact[i] = 1.0 if (self.foot_ground_object_names & contact_names) else 0.0
            if contact_names - self.foot_ground_object_names:
                feet_collision_cost += self.foot_collision_cost

        #electricity_cost  = self.electricity_cost  * float(np.abs(a*self.joint_speeds).mean())  # let's assume we have DC motor with controller, and reverse current braking
        #electricity_cost += self.stall_torque_cost * float(np.square(a).mean())
        electricity_cost = 0.0
        # bigger delta angle mean more jumpy movement
        
        angleDiffSpeed_cost = 0.0
        for n,j in enumerate(self.ordered_joints):
            ratio = min(abs(j.targetAngleDelta),np.pi)/np.pi
            angleDiffSpeed_cost = max(ratio,angleDiffSpeed_cost)
        #angleDiffSpeed_cost = angleDiffSpeed_cost/len(self.ordered_joints)
        angleDiffSpeed_cost_multiplier = -0.02
        angleDiffSpeed_cost = angleDiffSpeed_cost*angleDiffSpeed_cost_multiplier*globalMultiplier
        
        #energy cost for non contacted chain is better
        # use joint_to_foot
        
        
        self.reward_angleDiff.append(angleDiffSpeed_cost)
        
        #joints_at_limit_cost = float(self.joints_at_limit_cost * self.joints_at_limit)
        #joints_at_limit_cost = 0.0
        
        progressMultiplier = 5.0
        progressNegativeMultiplier = 10.0
        progressDir = np.dot(self.movement_per_frame,self.movement_dir)
        if progressDir<0.0:
            progressDir*=progressNegativeMultiplier
        else:
            progressDir*=progressMultiplier
        progressDir*=globalMultiplier
        progress*=globalMultiplier
        progress+=progressDir
        self.rewards_progress.append(progress)


        self.rewards = [
            alive,
            progress,
            angleDiffSpeed_cost,
            #electricity_cost,
            #joints_at_limit_cost,
            #feet_collision_cost
            ]

        self.frame  += 1
        rewardSummary = {}

        if (done and not self.done) or self.frame==self.spec.max_episode_steps:
            self.episode_over(self.frame)

            reward_progress = np.sum(self.rewards_progress)
            reward_angleDiff = np.sum(self.reward_angleDiff)
            reward_alive = np.sum(self.reward_alive)
            rewardSummary = { "alive":reward_alive, "progress":reward_progress, "servo":reward_angleDiff, "distToTarget":self.walk_target_dist }

            if self.debugStats:
                print("Episode stats::")
                print("Reward_progress: ", reward_progress)
                print("Reward_angleDiff: ", reward_angleDiff)
                print("Reward_alive: ", reward_alive)
                print("Reward_total: ", reward_alive+reward_progress+reward_angleDiff)
            self.reward_alive.clear()
            self.rewards_progress.clear()
            self.reward_angleDiff.clear()
            #find projection to target vector to figureout dist walked
            #distWalked = np.linalg.norm( [self.body_xyz[1],self.body_xyz[0]])
            #print("Dist Walked: ",distWalked)
        self.done   += done   # 2 == 1+True
        if bool(done) and self.frame==1:
            print("First frame done - something bad happended")
        self.reward += sum(self.rewards)
        self.HUD(state, a, done)
        return state, sum(self.rewards), bool(done), rewardSummary

    def robot_specific_reset(self):
        self.rewards_progress = []
        self.reward_angleDiff = []
        self.reward_alive = []
        self.ordered_joints = []
        self.jdict = {}
        jointToUse = ["fl1","fl2","fl3","fr1","fr2","fr3","bl1","bl2","bl3","br1","br2","br3"]
        for j in self.urdf.joints:
            if j.name in jointToUse:
                #print("\tJoint '%s' limits = %+0.2f..%+0.2f effort=%0.3f speed=%0.3f" % ((j.name,) + j.limits()) )
                j.servo_target = 0.0
                j.set_servo_target(j.servo_target,self.servo_kp,self.servo_kd,self.servo_maxTorque)
                j.power_coef, j.max_velocity = j.limits()[2:4]
                self.ordered_joints.append(j)
                self.jdict[j.name] = j
                continue
        RoboschoolForwardWalker.robot_specific_reset(self)
        self.body_xyz = [-0.3,0,0.18]
        RoboschoolForwardWalker.move_robot(self,self.body_xyz[0],self.body_xyz[1],self.body_xyz[2])
        self.numStepsPerSecond = 1.0/self.physics_time_step
        self.numSecondsToTrack = 1.0
        self.walked_distance = deque(maxlen=int(self.numSecondsToTrack*self.numStepsPerSecond))
        if settings.history1Len>0.0:
            self.history1 = deque(maxlen=int(settings.history1Len*self.numStepsPerSecond))
        if settings.history2Len>0.0:
            self.history2 = deque(maxlen=int(settings.history2Len*self.numStepsPerSecond))
        if settings.history3Len>0.0:
            self.history3 = deque(maxlen=int(settings.history3Len*self.numStepsPerSecond))
        if self.robotLibOn:
            self.robotLib.executeCommand("cmd#zero")
            #self.robotLib.executeCommand("cmd#testAction")
            #self.robotLib.executeCommand("cmd#initIdle")

    def reset(self):
        if self.scene is None:
            self.scene = self.create_single_player_scene()
        if not self.scene.multiplayer:
            self.scene.episode_restart()

        pose = cpp_household.Pose()
        #import time
        #t1 = time.time()
        self.urdf = self.scene.cpp_world.load_urdf(
            os.path.join(os.path.dirname(__file__), "models_robot", self.model_urdf),
            pose,
            self.fixed_base,
            self.self_collision)
        #t2 = time.time()
        #print("URDF load %0.2fms" % ((t2-t1)*1000))

        self.ordered_joints = []
        self.jdict = {}
        self.parts = {}
        self.frame = 0
        self.done = 0
        self.reward = 0
        dump = 0
        r = self.urdf
        self.cpp_robot = r
        if dump: print("ROBOT '%s'" % r.root_part.name)
        if r.root_part.name==self.robot_name:
            self.robot_body = r.root_part
        for part in r.parts:
            if dump: print("\tPART '%s'" % part.name)
            self.parts[part.name] = part
            if part.name==self.robot_name:
                self.robot_body = part
        for j in r.joints:
            if dump: print("\tALL JOINTS '%s' limits = %+0.2f..%+0.2f effort=%0.3f speed=%0.3f" % ((j.name,) + j.limits()) )
            if j.name[:6]=="ignore":
                j.set_motor_torque(0)
                continue
            j.power_coef, j.max_velocity = j.limits()[2:4]
            self.ordered_joints.append(j)
            self.jdict[j.name] = j
        #print("ordered_joints", len(self.ordered_joints))
        self.robot_specific_reset()
        self.cpp_robot.query_position()
        s = self.calc_state()
        self.potential = self.calc_potential()
        self.camera = self.scene.cpp_world.new_camera_free_float(self.VIDEO_W, self.VIDEO_H, "video_camera")
        #self.camera.move_and_look_at(0,1,0,0,0,0)
        return s

    def apply_action(self, a):
        assert( np.isfinite(a).all() )
        for n,j in enumerate(self.ordered_joints):
            # -1 to 1 converted to -pi,+pi
            #j.set_servo_target(float(np.clip(a[n], -1, +1))*np.pi,self.servo_kp,self.servo_kd,self.servo_maxTorque)
            #math.radians(10)
            #targetAngle = float(np.clip(a[n], -1, +1))*np.pi
            #self.time+=0.001
            curTarget = float(np.clip(a[n], -1, +1))*np.pi

            ''' Test pose 
            if(j.name=="fl1"):
                curTarget = math.radians(0);
            if(j.name=="fr1"):
                curTarget = math.radians(0);
            if(j.name=="bl1"):
                curTarget = math.radians(0);
            if(j.name=="br1"):
                curTarget = math.radians(0);
            if(j.name=="fl2"):
                curTarget = math.radians(45);
            if(j.name=="fr2"):
                curTarget = math.radians(45);
            if(j.name=="fl3"):
                curTarget = math.radians(-45);
            if(j.name=="fr3"):
                curTarget = math.radians(-45);
            if(j.name=="bl2"):
                curTarget = math.radians(45);
            if(j.name=="br2"):
                curTarget = math.radians(45);
            if(j.name=="bl3"):
                curTarget = math.radians(-45);
            if(j.name=="br3"):
                curTarget = math.radians(-45);
            '''

            j.targetAngle = curTarget
            j.targetAngleDelta = curTarget-j.servo_target
            #curTarget = math.radians(90)
            prevTarget = j.servo_target
            deltaAngle = curTarget-prevTarget
            deltaAngleAbs = abs(deltaAngle)
            maxServoAnglePerFrame = self.max_servo_speed*self.physics_time_step
            if deltaAngleAbs>maxServoAnglePerFrame:
                deltaAngleAbs = maxServoAnglePerFrame
            
            if curTarget>prevTarget:
                curTarget = prevTarget+deltaAngleAbs
            else:
                curTarget = prevTarget-deltaAngleAbs

            #clamp angles to limits
            limits = j.limits()
            if curTarget<limits[0]:
                curTarget = limits[0]
            if curTarget>limits[1]:
                curTarget = limits[1]
            
            j.servo_target = curTarget


            #j.servo_target =  math.sin(self.time)*np.pi*0.3
            j.set_servo_target(j.servo_target,self.servo_kp,self.servo_kd,self.servo_maxTorque)
            #j.set_motor_torque( self.power*j.power_coef*float(np.clip(a[n], -1, +1)) )


class QuadruppedWalker(RoboschoolForwardWalkerMujocoXML):
    '''
    3-D Quadruped walker similar to roboschool. 
    The task is to make the creature walk as fast as possible
    '''
    foot_list = ['fl4_link', 'fr4_link', 'bl4_link', 'br4_link']
    joint_to_foot  = { 
                    "fl1" : "fl4_link",
                    "fl2" : "fl4_link",
                    "fl3" : "fl4_link",

                    "fr1" : "fr4_link",
                    "fr2" : "fr4_link",
                    "fr3" : "fr4_link",
                    
                    "bl1" : "bl4_link",
                    "bl2" : "bl4_link",
                    "bl3" : "bl4_link",
                    
                    "br1" : "br4_link",
                    "br2" : "br4_link",
                    "br3" : "br4_link"
                }
    def __init__(self):
        RoboschoolForwardWalkerMujocoXML.__init__(self, "cat.urdf", "chassis", action_dim=12, power=2.5)

    def distance_numpy(A, B, P):
        """ segment line AB, point P, where each one is an array([x, y]) """
        if all(A == P) or all(B == P):
            return 0
        if arccos(dot((P - A) / norm(P - A), (B - A) / norm(B - A))) > pi / 2:
            return norm(P - A)
        if arccos(dot((P - B) / norm(P - B), (A - B) / norm(A - B))) > pi / 2:
            return norm(P - B)
        return norm(cross(A-B, A-P))/norm(B-A)


    def alive_bonus(self,state):
        vecStep = [0,0,0]
        vecMovement = 0.0
        if self.prev_body_xyz is not None:
            vecStep = [self.prev_body_xyz[0] - self.body_xyz[0],self.prev_body_xyz[1] - self.body_xyz[1]] #, self.prev_body_xyz[2] - self.body_xyz[2]]
            vecMovement = np.linalg.norm( vecStep )
            self.walked_distance.append(vecMovement)
        if len(self.walked_distance)==self.walked_distance.maxlen:
            distLastInterval = np.mean(self.walked_distance)
            minDistPerSecToWalk = 0.01 # cm per sec
            minAvgDist = 0.0003 # minDistPerSecToWalk/self.numStepsPerSecond
            #        '''
            if distLastInterval<minAvgDist:
                if self.debugStats:
                    print("Long wait without moving:",distLastInterval," ",minAvgDist)
                self.walked_distance.clear();
                return -1
        #'''
        #distToLine = norm(np.cross(p2-p1, p1-p3))/norm(p2-p1)
        #if np.abs(self.body_xyz[1])>0.5:
        #    print("Far away from central line ",self.body_xyz[1] )
        #    return -1
        maxRoll = 20.0
        maxPitch = 25.0
        r, p, yaw = self.body_rpy
        # roll pitch angles check
        if r>math.radians(maxRoll) or p>math.radians(maxPitch) or r<math.radians(-maxRoll) or p<math.radians(-maxPitch):
            return -1
        # body height check
        avgZ = self.parts["fl4_link"].pose().xyz()[2]+self.parts["fr4_link"].pose().xyz()[2]+self.parts["bl4_link"].pose().xyz()[2]+self.parts["br4_link"].pose().xyz()[2]
        avgZ =avgZ/4.0
        if self.body_xyz[2]<avgZ+0.07:
            return -1
        return +1