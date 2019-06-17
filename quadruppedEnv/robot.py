
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
import time
from quadruppedEnv import settings
#from pyrr import Quaternion, Matrix44, Vector3
#import cProfile
#cp = cProfile.Profile()
import glm
import xml.etree.ElementTree as ET

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

        urdfFilename = os.path.join(os.path.dirname(__file__), "models_robot", fn)


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
        RoboschoolUrdfEnv.__init__(self, urdfFilename,robot_name, action_dim, numObservation,
            fixed_base=False,
            self_collision=False)
        RoboschoolForwardWalker.__init__(self, power)
        self.servo_kp = 0.3
        self.servo_kd = 0.9
        self.servo_maxTorque = 2.0
        self.time = 0.0
        self.max_servo_speed = math.radians(230.0)
        self.physics_time_step = 1.0/240.0
        self.walk_target_x = 1000.0 # 1km  meters
        self.walk_target_z = 0.15
        self.debugStats = 0
        self.movement_dir = [1,0,0]
        self.robotLibOn = 0

        self.jointsLocalPos = {}
        tree = ET.parse(urdfFilename)
        root = tree.getroot()
        for robot in root.iter('robot'):
            for joint in robot.iter('joint'):
                origin = joint.find("origin")
                posStr = origin.attrib["xyz"].split(" ")
                pos = [float(posStr[0]),float(posStr[1]),float(posStr[2])]
                self.jointsLocalPos[joint.attrib["name"]] = pos
                
        self.link1 = self.jointsLocalPos["fl3"]
        self.link2 = self.jointsLocalPos["fl4"]

        self.extLenX = 0.15 # 15 cm
        self.extLenY = 0.15 # 15 cm
        self.minZLeg = -0.2
        self.maxZLeg = -0.02

        self.minArea = [-self.extLenX,-self.extLenY,self.minZLeg] # 20 cm away from chassis
        self.maxArea = [+self.extLenX,+self.extLenY,self.maxZLeg] # 2 cm from chassis 

        self.flMinArea = glm.vec3(self.minArea)
        self.flMaxArea = glm.vec3(self.maxArea)
        self.frMinArea = glm.vec3(self.minArea)
        self.frMaxArea = glm.vec3(self.maxArea)

        self.blMinArea = glm.vec3(self.minArea)
        self.blMaxArea = glm.vec3(self.maxArea)
        self.brMinArea = glm.vec3(self.minArea)
        self.brMaxArea = glm.vec3(self.maxArea)

        self.areaSize = [self.maxArea[0]-self.minArea[0],self.maxArea[1]-self.minArea[1],self.maxArea[2]-self.minArea[2]]
        self.len1 = abs( self.link1[2])
        self.len2 = abs( self.link2[2])
        self.ActionIsAngles = False
        self.ActionsIsAdditive = False
        self.actionsMult = 1.0
        self.chassisSpaceX = 0.30
        self.chassisSpaceY = 0.20
        self.chassisSpaceMin = [-self.chassisSpaceX,-self.chassisSpaceY,-0.2]
        self.chassisSpaceMax = [+self.chassisSpaceX,+self.chassisSpaceY,-0.02]
        self.inputsInChassisSpace = False
        self.actionsInChassisSpace = False
        self.inputsIsIKTargets = True

        self.perLegSpace = True
        if self.perLegSpace:
            self.extLenYOut = 0.1 # 10  cm
            self.extLenYIn = 0.03 # 2 cm
            self.flMinArea = glm.vec3(-self.extLenX,-self.extLenYIn,self.minZLeg)
            self.flMaxArea = glm.vec3(self.extLenX,self.extLenYOut,-self.maxZLeg)
            self.blMinArea = glm.vec3(-self.extLenX,-self.extLenYIn,self.minZLeg)
            self.blMaxArea = glm.vec3(self.extLenX,self.extLenYOut,-self.maxZLeg)

            self.frMinArea = glm.vec3(-self.extLenX,-self.extLenYOut,self.minZLeg)
            self.frMaxArea = glm.vec3(self.extLenX,self.extLenYIn,-self.maxZLeg)
            self.brMinArea = glm.vec3(-self.extLenX,-self.extLenYOut,self.minZLeg)
            self.brMaxArea = glm.vec3(self.extLenX,self.extLenYIn,-self.maxZLeg)


        if settings.robotLibAvailable and hasattr(settings,"robotNN"):
            if settings.robotNN==0:
                self.robotLibOn = 1
            else:
                self.robotLibOn = 0
            self.robotLib = pyRobotLib.Robot()
            self.robotLib.load("/Users/pavlogryb/Dropbox/Projects/Robot/Robot/Robot/")
            print(os.getpid())

    def create_single_player_scene(self):
        return SinglePlayerStadiumScene(gravity=9.8, timestep=self.physics_time_step, frame_skip=1)

    def calc_state(self):
        #cp.enable()
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
        #cp.disable()
        #cp.print_stats()
        return robot_state

    def getLastSingleState(self):
        return self.last_single_state


    def rotation(self,orig,dest):
        identityQuat = glm.quat(1.0,0.0,0.0,0.0)
        epsilon = 0.00001

        cosTheta = glm.dot(orig, dest);

        if cosTheta >= 1.0 - epsilon:
            #// orig and dest point in the same direction
            return identityQuat;

        if cosTheta < -1.0 + epsilon:
            '''
            // special case when vectors in opposite directions :
            // there is no "ideal" rotation axis
            // So guess one; any will do as long as it's perpendicular to start
            // This implementation favors a rotation around the Up axis (Y),
            // since it's often what you want to do.
            '''
            rotationAxis = glm.cross(glm.vec3(0.0, 0.0, 1.0), orig);
            if glm.length(rotationAxis) < epsilon: # // bad luck, they were parallel, try again!
                rotationAxis = glm.cross(glm.vec3(1.0, 0.0, 0.0), orig)

            rotationAxis = glm.normalize(rotationAxis)
            return glm.angleAxis(glm.pi(), rotationAxis)

        #// Implementation from Stan Melax's Game Programming Gems 1 article
        rotationAxis = glm.cross(orig, dest);

        s = math.sqrt((1.0 + cosTheta) * 2.0);
        invs = 1.0 / s;

        return glm.quat(s * 0.5,rotationAxis.x * invs,rotationAxis.y * invs,rotationAxis.z * invs)



    def solve_square_equation(self,a,b,c):
        d = b * b - 4 * a * c; #// disc
        if d < 0:
            return 0,0.0,0.0

        q = 0.0
        if b >= 0:
            q = (-b - math.sqrt(d)) / 2.0
        else:
            q = (-b + math.sqrt(d)) / 2.0

        x1 = q / a
        x2 = c / q

        if d == 0.0:
            return 1,x1,x2
        return 2,x1,x2

    def angleBetween(self,a3,b3):
        a = glm.vec2(a3)
        b = glm.vec2(b3)
        p = glm.vec2(-b[1], b[0])
        b_coord = glm.dot(a, b)
        p_coord = glm.dot(a, p)
        return math.atan2(p_coord, b_coord)

    def normAngle(self,angle):
        #// 180-270
        if angle >= glm.pi() and angle <= glm.three_over_two_pi():
            return angle - glm.two_pi()
        #// 270-360
        if angle >= glm.three_over_two_pi() and angle <= glm.two_pi():
            return angle - glm.two_pi()
        return angle

    def solveIK(self,coord, Linkage_1, Linkage_2):
        l2 = Linkage_1 * Linkage_1;
        L2 = Linkage_2 * Linkage_2;

        l = Linkage_1
        L = Linkage_2

        x = coord[0];
        y = coord[1];

        SCARA_C2 = ((x * x) + (y * y) - l2 - L2) / (2.0 * l * L)

        bb = 1.0 - (SCARA_C2 * SCARA_C2)
        if bb < 0.0:
            norm = glm.normalize(glm.vec3(coord[0], coord[1], 0.0))
            q1 = math.atan2(norm[1], norm[0])
            return q1, 0.0, 0.0
  
        SCARA_S2 = math.sqrt(bb)

        SCARA_K1 = l + L * SCARA_C2
        SCARA_K2 = L * SCARA_S2

        SCARA_theta = (math.atan2(x, y) - math.atan2(SCARA_K1, SCARA_K2)) * -1.0
        SCARA_psi = math.atan2(SCARA_S2, SCARA_C2)

        q11 = SCARA_theta
        q22a = SCARA_psi

        q11 = self.normAngle(q11)
        q22a = self.normAngle(q22a)
        return q11, q22a, 1.0

    def angleDiff(self,a,b):
        dif = math.fmod(b - a + glm.pi(), 2.0 * glm.pi())
        if dif < 0:
            dif += 2.0 *glm.pi()
        return dif - glm.pi()

    def solveLeg(self,upperLegOffset,l1,l2,target):
        upperLeg = glm.vec3(upperLegOffset[1], upperLegOffset[2], 0.0);
        upperLegNorm = glm.normalize(upperLeg);
        '''
        //
        //                 0
        //           a    /
        //              /
        //         ulo/  cosa
        //            |
        //            |  L
        //         b  |
        //            |
        //            t
        //
        // c = |0-t|
        // b = |ulo-t| , is unknown
        '''
        cosa = upperLegNorm[1]
        upperLegLen = glm.length(upperLeg)

        targetYZ = glm.vec3(target[1], target[2], 0.0);
        targetYZLen = glm.length(targetYZ);
        '''
        //
        // c^2 = a^2 + b^2 - 2*a*b*cosa
        //
        //  b^2 - b*(2*a*cosa) + (a^2-c^2)
        //
        '''
        ka = 1.0;
        kb = -2.0 * upperLegLen * cosa;
        kc = upperLegLen * upperLegLen - targetYZLen * targetYZLen;
        numRoots, x1, x2 = self.solve_square_equation(ka, kb, kc)

        root = max(x1, x2)

        targetYZOrg = glm.vec3(upperLegOffset[1], upperLegOffset[2] - root, 0.0)
        targetYZOrgNorm = glm.normalize(targetYZOrg)
        targetYZCurNorm = glm.normalize(targetYZ)

        alpha = self.angleBetween(targetYZCurNorm, targetYZOrgNorm)
        targetYZCurNorm3D = glm.vec3(0.0, targetYZCurNorm[0], targetYZCurNorm[1])
        targetYZOrgNorm3D = glm.vec3(0.0, targetYZOrgNorm[0], targetYZOrgNorm[1])
        rot = self.rotation(targetYZOrgNorm3D, targetYZCurNorm3D)

        newTarget = glm.mat3_cast(rot)*glm.vec3(target[0], target[2], target[1])

        target2d = newTarget - glm.vec3(upperLegOffset[0], upperLegOffset[2], upperLegOffset[1])
        rotLeg = self.solveIK(target2d, l1, l2)

        angleOffset = glm.vec3(0.0, glm.radians(-90.0), 0.0)

        jangles = glm.vec4(self.angleDiff(angleOffset[0], alpha), -1.0*self.angleDiff(angleOffset[1], rotLeg[0]),
                    -1.0*self.angleDiff(angleOffset[2], rotLeg[1]), rotLeg[2])

        return jangles

    def getJ1TM(self,jointName):
        resultTm = glm.mat4x4()
        #resultTm = glm.rotate(resultTm,angles[0], glm.vec3([1.0,0.0,0.0]))
        resultTm = glm.translate(resultTm,self.jointsLocalPos[jointName])
        return resultTm

    def getLocalXYZFromAngles(self,angles,len1,len2,joint2Offs):
        #state_start = time.time()
        resultTm = glm.mat4x4()
        resultTm = glm.rotate(resultTm,angles[0], glm.vec3([1.0,0.0,0.0]))
        resultTm = glm.translate(resultTm,joint2Offs)
        resultTm = glm.rotate(resultTm,angles[1], glm.vec3([0.0,1.0,0.0]))
        resultTm = glm.translate(resultTm,len1)
        resultTm = glm.rotate(resultTm,angles[2], glm.vec3([0.0,1.0,0.0]))
        resultTm = glm.translate(resultTm,len2)
        point = glm.vec3(glm.column(resultTm,3))
        #print("tm",(time.time()-state_start)*1000.0)
        return point
        '''
        j1Rot = Matrix44.from_x_rotation(angles[0])
        print("tm",(time.time()-state_start)*1000.0)

        j2Offs = Matrix44.from_translation(joint2Offs)
        j2Rot = Matrix44.from_y_rotation(angles[1])

        j3Offs = Matrix44.from_translation(len1)
        j3Rot = Matrix44.from_y_rotation(angles[2])

        j4Offs = Matrix44.from_translation(len2)

        resMatrix = j1Rot * j2Offs * j2Rot * j3Offs * j3Rot * j4Offs
        # * j2Rot * j2Offs * j3Rot * j3Offs  * j4Offs
        point = resMatrix * Vector3([0.,0.,0.])
        return point
        '''

    def rangeNormalize(self,value,minArea,maxArea):
        areaSize = maxArea-minArea
        clipped = np.clip(value,minArea,maxArea)
        x = (clipped[0]-minArea[0])/(areaSize[0])*2.0-1.0
        y = (clipped[1]-minArea[1])/(areaSize[1])*2.0-1.0
        z = (clipped[2]-minArea[2])/(areaSize[2])*2.0-1.0
        return [x,y,z]

    def chassisNormalize(self,value):
        clipped = np.clip(value,self.chassisSpaceMin,self.chassisSpaceMax)
        x = (clipped[0]-self.chassisSpaceMin[0])/(self.chassisSpaceMax[0]-self.chassisSpaceMin[0])*2.0-1.0
        y = (clipped[1]-self.chassisSpaceMin[1])/(self.chassisSpaceMax[1]-self.chassisSpaceMin[1])*2.0-1.0
        z = (clipped[2]-self.chassisSpaceMin[2])/(self.chassisSpaceMax[2]-self.chassisSpaceMin[2])*2.0-1.0
        return [x,y,z]

    def syncLocalFromXYZ(self,fl,fr,bl,br, chassisSpace=False):
        self.flPos = self.getLocalXYZFromAngles(fl,self.link1,self.link2,self.jointsLocalPos["fl2"])
        if chassisSpace:
            self.flPos = self.flPos+self.jointsLocalPos["fl1"]
            self.flPosN = np.float32(self.chassisNormalize(self.flPos))
        else:
            self.flPosN = np.float32(self.rangeNormalize(self.flPos,self.flMinArea,self.flMaxArea))

        self.frPos = self.getLocalXYZFromAngles(fr,self.link1,self.link2,self.jointsLocalPos["fr2"])
        if chassisSpace:
            self.frPos = self.frPos+self.jointsLocalPos["fr1"]
            self.frPosN = np.float32(self.chassisNormalize(self.frPos))
        else:
            self.frPosN = np.float32(self.rangeNormalize(self.frPos,self.frMinArea,self.frMaxArea))

        self.blPos = self.getLocalXYZFromAngles(bl,self.link1,self.link2,self.jointsLocalPos["bl2"])
        if chassisSpace:
            self.blPos = self.blPos+self.jointsLocalPos["bl1"]
            self.blPosN = np.float32(self.chassisNormalize(self.blPos))
        else:
            self.blPosN = np.float32(self.rangeNormalize(self.blPos,self.blMinArea,self.blMaxArea))

        self.brPos = self.getLocalXYZFromAngles(br,self.link1,self.link2,self.jointsLocalPos["br2"])
        if chassisSpace:
            self.brPos = self.brPos+self.jointsLocalPos["br1"]
            self.brPosN = np.float32(self.chassisNormalize(self.brPos))
        else:
            self.brPosN = np.float32(self.rangeNormalize(self.brPos,self.brMinArea,self.brMaxArea))


    def calc_state_single(self):
        #state_start = time.time()
        body_pose = self.robot_body.pose()
        
        self.body_rpy = body_pose.rpy()
        r, p, yaw = self.body_rpy

        j  = []

        #self.bodyRot = glm.rotate(yaw, glm.vec3(0.0, 0.0, 1.0)) *
        #           glm.rotate(p, glm.vec3(0.0, 1.0, 0.0)) *
        #           glm::rotate(r, glm::vec3(1.0, 0.0, 0.0))

        if self.inputsIsIKTargets:
            
            fl = [self.jdict["fl1"].servo_target,self.jdict["fl2"].servo_target,self.jdict["fl3"].servo_target]
            fr = [self.jdict["fr1"].servo_target,self.jdict["fr2"].servo_target,self.jdict["fr3"].servo_target]
            bl = [self.jdict["bl1"].servo_target,self.jdict["bl2"].servo_target,self.jdict["bl3"].servo_target]
            br = [self.jdict["br1"].servo_target,self.jdict["br2"].servo_target,self.jdict["br3"].servo_target]

            self.syncLocalFromXYZ(fl,fr,bl,br,self.inputsInChassisSpace)
            j = np.concatenate([self.flPosN]+[self.frPosN]+[self.blPosN]+[self.brPosN])
            self.lastJointsStates = j
        else:
            servo_angles =  []
            for j in self.ordered_joints:
                # scaled beetween -1..1 between limits
                limits = j.limits()
                limitsMiddle = (limits[0]+limits[1])*0.5
                jointObservation = 2 * (j.servo_target - limitsMiddle) / (limits[1]-limits[0])
                servo_angles.append(jointObservation)
            j  = servo_angles

        #print("Joints3",(time.time()-state_start)*1000.0)
        #print("Joints",time.time()-state_start)

        #print(j)
        #parts_xyz = np.array( [p.pose().xyz() for p in self.parts.values()] ).flatten()
        curBodyXYZ = (body_pose.xyz()[0], body_pose.xyz()[1], body_pose.xyz()[2])
        self.prev_body_xyz = self.body_xyz;
        self.body_xyz = curBodyXYZ
        self.movement_per_frame = [self.body_xyz[0] - self.prev_body_xyz[0],self.body_xyz[1] - self.prev_body_xyz[1], self.body_xyz[2] - self.prev_body_xyz[2]]
        self.walk_target_theta = np.arctan2( self.walk_target_y - self.body_xyz[1], self.walk_target_x - self.body_xyz[0] )
        self.vecToTarget = [self.walk_target_x - self.body_xyz[0],self.walk_target_y - self.body_xyz[1], self.walk_target_z - self.body_xyz[2]]
        self.walk_target_dist  = np.linalg.norm( self.vecToTarget )
        self.angle_to_target = self.walk_target_theta - yaw
        frontRot = glm.rotate(glm.mat4x4(),yaw,glm.vec3(0.0,0.0,1.0))
        self.body_frontVec = glm.mat3(frontRot)* glm.vec3(1.0,0.0,0.0);
        '''
        self.rot_minus_yaw = np.array(
            [[np.cos(-yaw), -np.sin(-yaw), 0],
             [np.sin(-yaw),  np.cos(-yaw), 0],
             [           0,             0, 1]]
            )
        '''

        observationsBase = np.array([
            r, p, yaw,
            np.sin(self.angle_to_target), np.cos(self.angle_to_target)
            ], dtype=np.float32)
        observations = np.concatenate([observationsBase]+[self.feet_contact]+[np.float32(j)])

        # 0.3 is just scaling typical speed into -1..+1, no physical sense here
        if self.velObservations>0:
            vx, vy, vz = self.robot_body.speed()
            body_vel = np.array([0.3*vx, 0.3*vy, 0.3*vz], dtype=np.float32)
            observations = np.append(observations,body_vel)

        self.last_single_state = observations

        #print("all",time.time()-state_start)
        return observations

    def calc_potential(self):
        # progress in potential field is speed*dt, typical speed is about 2-3 meter per second, this potential will change 2-3 per frame (not per second),
        # all rewards have rew/frame units and close to 1.0
        #return - self.walk_target_dist
        return - self.walk_target_dist / self.physics_time_step

    def calculateAnalyticalActions(self,state):
        actions = self.robotLib.getActions(state.tolist(),self.physics_time_step)
        a = np.asarray(actions)
        if self.ActionIsAngles==False:
            if self.ActionsIsAdditive:
                fl = [a[0]*glm.pi(),a[1]*glm.pi(),a[2]*glm.pi()]
                fr = [a[3]*glm.pi(),a[4]*glm.pi(),a[5]*glm.pi()]
                bl = [a[6]*glm.pi(),a[7]*glm.pi(),a[8]*glm.pi()]
                br = [a[9]*glm.pi(),a[10]*glm.pi(),a[11]*glm.pi()]
                self.syncLocalFromXYZ(fl,fr,bl,br,self.inputsInChassisSpace)
                a = np.concatenate([self.flPosN]+[self.frPosN]+[self.blPosN]+[self.brPosN])
                a = np.subtract(a,self.lastJointsStates)
            else:
                fl = [a[0]*glm.pi(),a[1]*glm.pi(),a[2]*glm.pi()]
                fr = [a[3]*glm.pi(),a[4]*glm.pi(),a[5]*glm.pi()]
                bl = [a[6]*glm.pi(),a[7]*glm.pi(),a[8]*glm.pi()]
                br = [a[9]*glm.pi(),a[10]*glm.pi(),a[11]*glm.pi()]
                self.syncLocalFromXYZ(fl,fr,bl,br,self.inputsInChassisSpace)
                a = np.concatenate([self.flPosN]+[self.frPosN]+[self.blPosN]+[self.brPosN])
        return a

    def stepPG(self, a):
        #cp.enable()
        if self.robotLibOn:
            #print(self.last_state)
            if self.frame==50:
                self.robotLib.executeCommand("cmd#initIdle")
            if self.frame==199:
                forDebug = 0
            if self.frame==200:
                forDebug = 0
                self.robotLib.executeCommand("cmd#testAction")
            a = self.calculateAnalyticalActions(self.last_state)/self.actionsMult

        if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then step() for all robots with the same actions
            self.apply_action(a)
            self.scene.global_step()

        state = self.calc_state()  # also calculates self.joints_at_limit

        alive_multiplier = 1.0
        alive = float(self.alive_bonus(state))
        if alive<0 and self.frame==0:
            print("bad transition")
        alive*=alive_multiplier

        #if alive<0:
        #    alive = -100.0
 
        done = alive < 0
        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True

        potential_old = self.potential
        self.potential = self.calc_potential()
        progress = float(self.potential - potential_old)
        self.progressHistory.append(progress)
        progressMultiplier = 2.0
        progress*=progressMultiplier

        #'''
        progressDirMultiplier = 1.0 #10.0
        self.moveDirNormalizer = glm.normalize(self.movement_per_frame)
        progressDir = glm.dot(self.moveDirNormalizer,self.movement_dir)
        progressDir*=progressDirMultiplier
        progress += progressDir

        progressDirMultiplier = 0.2 #10.0
        progressDir = glm.dot(self.body_frontVec,self.movement_dir)
        progressDir*=progressDirMultiplier
        progress += progressDir


        #'''
        '''
        if progress<0:
            progress = -1.0
        else:
            progress = 1.0
        '''
        '''
        if self.walk_target_dist>0.1 and self.prev_body_xyz is not None:
            vecStep = [self.prev_body_xyz[0] - self.body_xyz[0],self.prev_body_xyz[1] - self.body_xyz[1]] #, self.prev_body_xyz[2] - self.body_xyz[2]]
            vecMovement = np.linalg.norm( vecStep )
            minSpeedToPunishPerSec = 0.01 # 1 cm
            minMovePerFrame = minSpeedToPunishPerSec/self.numStepsPerSecond
            if vecMovement<minMovePerFrame:
                progress = -1
        '''
            
        for i,f in enumerate(self.feet):
            contact_names = set(x.name for x in f.contact_list())
            #print("CONTACT OF '%s' WITH %s" % (f.name, ",".join(contact_names)) )
            self.feet_contact[i] = 1.0 if (self.foot_ground_object_names & contact_names) else 0.0

        maxServoAnglePerFrame = self.max_servo_speed*self.physics_time_step
        angleDiffSpeed_cost_max = 0.0
        angleDiffSpeed_cost_min = 1.0
        angleDiffSpeed_cost_multiplier = -0.08
        for n,j in enumerate(self.ordered_joints):
            ratio = min(abs(j.targetAngleDelta),np.pi)/np.pi
            #ratio = min(abs(j.targetAngleDeltaFrame),maxServoAnglePerFrame)/maxServoAnglePerFrame
            ratio*=j.energy_cost
            ratio+=(1.0-j.newCurTargetReachable)*0.6
            angleDiffSpeed_cost_min = min(ratio,angleDiffSpeed_cost_min)
            angleDiffSpeed_cost_max = max(ratio,angleDiffSpeed_cost_max)
        lerpFactor = 0.7
        angleDiffSpeed_cost = angleDiffSpeed_cost_min+lerpFactor*(angleDiffSpeed_cost_max-angleDiffSpeed_cost_min)
        angleDiffSpeed_cost *= angleDiffSpeed_cost_multiplier
        
        #energy cost for non contacted chain is better
        # use joint_to_foot
        
        
        self.rewards_progress.append(progress)
        self.reward_alive.append(alive)
        self.reward_angleDiff.append(angleDiffSpeed_cost)

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
            done = True

            reward_progress = np.sum(self.rewards_progress)
            reward_angleDiff = np.sum(self.reward_angleDiff)
            reward_alive = np.sum(self.reward_alive)
            rewardTotal = reward_progress+reward_angleDiff+reward_alive
            rewardSummary = { "alive":reward_alive, "progress":reward_progress, "servo":reward_angleDiff, "distToTarget":self.walk_target_dist, "episode_steps":self.frame, "episode_reward":rewardTotal  }

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
        #cp.disable()
        #cp.print_stats()
        if done:
            debugBrek = 0
        return state, sum(self.rewards), bool(done), rewardSummary

    def step(self, a):
        return self.stepPG(a)

    def setupEnergyCost(self,link):
        self.jdict[link+"1"].energy_cost = 1.7 # full link mass
        self.jdict[link+"2"].energy_cost = 2.0 # link mass
        self.jdict[link+"3"].energy_cost = 1.0


    def robot_specific_reset(self):
        self.rewards_progress = []
        self.reward_angleDiff = []
        self.reward_alive = []
        self.ordered_joints = []
        self.jdict = {}

        self.jointToUse = ["fl1","fl2","fl3","fr1","fr2","fr3","bl1","bl2","bl3","br1","br2","br3"]
        for j in self.urdf.joints:
            if j.name in self.jointToUse:
                #print("\tJoint '%s' limits = %+0.2f..%+0.2f effort=%0.3f speed=%0.3f" % ((j.name,) + j.limits()) )
                j.servo_target = 0.0
                j.set_servo_target(j.servo_target,self.servo_kp,self.servo_kd,self.servo_maxTorque)
                j.power_coef, j.max_velocity = j.limits()[2:4]
                self.ordered_joints.append(j)
                self.jdict[j.name] = j
                continue
        
        self.iklinks_fl = [self.jdict["fl1"],self.jdict["fl2"],self.jdict["fl3"]]
        self.iklinks_fr = [self.jdict["fr1"],self.jdict["fr2"],self.jdict["fr3"]]
        self.iklinks_bl = [self.jdict["bl1"],self.jdict["bl2"],self.jdict["bl3"]]
        self.iklinks_br = [self.jdict["br1"],self.jdict["br2"],self.jdict["br3"]]

        self.setupEnergyCost("fl")
        self.setupEnergyCost("fr")
        self.setupEnergyCost("bl")
        self.setupEnergyCost("br")
 
        RoboschoolForwardWalker.robot_specific_reset(self)
        self.body_xyz = [-0.3,0,0.18]
        RoboschoolForwardWalker.move_robot(self,self.body_xyz[0],self.body_xyz[1],self.body_xyz[2])
        self.numStepsPerSecond = 1.0/self.physics_time_step
        self.numSecondsToTrack = 1.0
        self.walked_distance = deque(maxlen=int(self.numSecondsToTrack*self.numStepsPerSecond))
        self.progressHistory = deque(maxlen=int(self.numSecondsToTrack*self.numStepsPerSecond))
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
            self.model_urdf,
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

    def syncAngles(self,linkName, target, j0, j1, j2, minArea,maxArea, chassisSpace=False):
        target = np.clip(target, -1.0, +1.0)
        j2LocalPos = self.jointsLocalPos[linkName+"2"]
        if chassisSpace:
            target[0] = glm.mix(self.chassisSpaceMin[0],self.chassisSpaceMax[0],target[0]*0.5+0.5)
            target[1] = glm.mix(self.chassisSpaceMin[1],self.chassisSpaceMax[1],target[1]*0.5+0.5)
            target[2] = glm.mix(self.chassisSpaceMin[2],self.chassisSpaceMax[2],target[2]*0.5+0.5)
            target = target-self.jointsLocalPos[linkName+"1"]
        else:
            target[0] = glm.mix(minArea[0],maxArea[0],target[0]*0.5+0.5)
            target[1] = glm.mix(minArea[1],maxArea[1],target[1]*0.5+0.5)
            target[2] = glm.mix(minArea[2],maxArea[2],target[2]*0.5+0.5)
        angles = self.solveLeg(j2LocalPos,self.len1,self.len2,target)
        j0.newCurTarget = angles[0]
        j0.newCurTargetReachable = angles[3]
        j1.newCurTarget = angles[1]
        j1.newCurTargetReachable = angles[3]
        j2.newCurTarget = angles[2]
        j2.newCurTargetReachable = angles[3]


    def apply_action(self, a):
        assert( np.isfinite(a).all() )
        a = a * self.actionsMult
        #a[0],a[1],a[2] = self.rangeNormalize([0.0,0.195,-0.15])
        #a[3],a[4],a[5] = self.rangeNormalize([0.0,-0.195,-0.15])
        #a[6],a[7],a[8] = self.rangeNormalize([0.0,0.195,-0.15])
        #a[9],a[10],a[11] = self.rangeNormalize([0.0,-0.195,-0.15])

        #angles = float(np.clip([self.ordered_joints[0],self.ordered_joints[1],self.ordered_joints[2]], -1.0, +1.0))
        #angles[0] = glm.mix(angles[0]*0.5+0.5,self.minArea[0],self.minArea[0])
        #angles[1] = glm.mix(angles[1]*0.5+0.5,self.minArea[1],self.minArea[1])
        #angles[2] = glm.mix(angles[2]*0.5+0.5,self.minArea[2],self.minArea[2])

        #angles = self.solveLeg(self.jointsLocalPos["fl2"],self.len1,self.len2,[0,0.035,-0.15-0.25])

        #angles = self.solveLeg(self.jointsLocalPos["fl2"],self.len1,self.len2,[0,0.-0.035,-0.15-0.25])

        #angles = self.solveLeg(self.jointsLocalPos["fl2"],self.len1,self.len2,[0.1,0.-0.035,-0.10])
        
        if self.ActionIsAngles:
            for n,j in enumerate(self.ordered_joints):
                j.newCurTarget = float(np.clip(a[n], -1, +1))*np.pi
        else:
            a = np.clip(a, -1.0, +1.0)

            #aProcessed = [a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9],a[10],a[11]]
            if self.ActionsIsAdditive:
                a = np.add(self.lastJointsStates,a)
            self.syncAngles("fl",[a[0],a[1],a[2]],self.jdict['fl1'],self.jdict['fl2'],self.jdict['fl3'],self.flMinArea,self.flMaxArea,self.actionsInChassisSpace)
            self.syncAngles("fr",[a[3],a[4],a[5]],self.jdict['fr1'],self.jdict['fr2'],self.jdict['fr3'],self.frMinArea,self.frMaxArea,self.actionsInChassisSpace)
            self.syncAngles("bl",[a[6],a[7],a[8]],self.jdict['bl1'],self.jdict['bl2'],self.jdict['bl3'],self.blMinArea,self.blMaxArea,self.actionsInChassisSpace)
            self.syncAngles("br",[a[9],a[10],a[11]],self.jdict['br1'],self.jdict['br2'],self.jdict['br3'],self.brMinArea,self.brMaxArea,self.actionsInChassisSpace)

        for n,j in enumerate(self.ordered_joints):
            curTarget = j.newCurTarget

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
            
            j.targetAngleDeltaFrame = curTarget-j.servo_target
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
                return 0.1
        #'''
        #distToLine = norm(np.cross(p2-p1, p1-p3))/norm(p2-p1)
        #if np.abs(self.body_xyz[1])>0.5:
        #    print("Far away from central line ",self.body_xyz[1] )
        #    return -1
        maxRoll = 35.0
        maxPitch = 35.0
        r, p, yaw = self.body_rpy
        # roll pitch angles check
        if r>math.radians(maxRoll) or p>math.radians(maxPitch) or r<math.radians(-maxRoll) or p<math.radians(-maxPitch):
            #print(r,p,yaw)
            return -1
        # body height check
        avgZ = self.parts["fl4_link"].pose().xyz()[2]+self.parts["fr4_link"].pose().xyz()[2]+self.parts["bl4_link"].pose().xyz()[2]+self.parts["br4_link"].pose().xyz()[2]
        avgZ =avgZ/4.0
        if self.body_xyz[2]<avgZ+0.05:
            return -1
        return +1
