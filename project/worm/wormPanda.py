#!/usr/bin/env python



from direct.showbase.ShowBase import ShowBase
# from panda3d.core import AmbientLight, DirectionalLight, LightAttrib
# from panda3d.core import NodePath
# from panda3d.core import LVector3
from panda3d.core import *
from direct.interval.IntervalGlobal import *  # Needed to use Intervals
from direct.gui.DirectGui import *
#import "/simple_worm/worm.py"
# import importlib
#
# moduleName=input('simple_worm/worm.py')
# importlib.import_module(moduleName)
import random
import numpy as np
from numpy import *
import worm
from worm import Worm as wm
from worm import Geometry
# Importing math constants and functions
from math import pi, sin, cos
# from pandac.PandaModules import Vec4
# from panda.coreimport LineSegs
# import direct.directbase.DirectStart
from panda3d.core import GeomNode
import math

import sys
import os

from panda3d.bullet import BulletSphereShape
from panda3d.bullet import BulletRigidBodyNode
from panda3d.bullet import BulletWorld

import time
from fenics import Expression
# def normalize_v3(arr):
#     ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
#     lens = numpy.sqrt( arr[:,0]**2 + arr[:,1]**2 + arr[:,2]**2 )
#     arr[:,0] /= lens
#     arr[:,1] /= lens
#     arr[:,2] /= lens
#     return arr
def addInstructions(pos, msg):
    return OnscreenText(text=msg, style=1, fg=(1, 1, 1, 1), scale=.05,
                        shadow=(0, 0, 0, 1), parent=base.a2dTopLeft,
                        pos=(0.08, -pos - 0.04), align=TextNode.ALeft)
def normalizeTriangle(x,y,z):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    # arr=np.array([0,0,0])
    arr=([0,0,0])
    A=z-y
    B=y-x
    # a1=(A[1]*B[2])-(A[2]*B[1])
    # a2=(A[2]*B[0])-(A[0]*B[2])
    # a3=(A[0]*B[1])-(A[1]*B[0])
    # print(a1)
    # print(a2)
    # print(a3)
    # # print(a1)
    # arr[0]=a1/a3
    # arr[1]=a2/a3
    # arr[2]=1.0
    vlength=sqrt(x*x+y*y+z*z)
    arr[0]=x/vlength
    arr[1]=y/vlength
    arr[2]=z/vlength
    # arr/=np.linalg.norm(arr)
    # if a1<0:
    #     arr[0]=-1
    # if a2<0:
    #     arr[1]=-1
    # if a3<0:
    #     arr[2]=-1


    return arr



class WormDemo(ShowBase):

    def __init__(self):
        # Initialize the ShowBase class from which we inherit, which will
        # create a window and set up everything we need for rendering into it.
        ShowBase.__init__(self)

        # This creates the on screen title that is in every tutorial
        self.title = OnscreenText(text="Panda3D:Worm",
                                  parent=base.a2dBottomCenter,
                                  fg=(1, 1, 1, 1), shadow=(0, 0, 0, .5),
                                  pos=(0, .1), scale=.1)
        # Post the instructions

        self.inst1 = addInstructions(0.06, "[ESC]: Quit")
        self.inst2 = addInstructions(0.12, "[Left Arrow]: Rotate Worm Left")
        self.inst3 = addInstructions(0.18, "[Right Arrow]: Rotate Worm Right")
        self.inst4 = addInstructions(0.24, "[Up Arrow]: Run Worm Forward")
        self.inst6 = addInstructions(0.30, "[A]: Rotate Camera Left")
        self.inst7 = addInstructions(0.36, "[S]: Rotate Camera Right")

          # Allow manual positioning of the camera
         # Set the cameras' position    # self.world.attachRigidBody(np.node())
                                                # and orientation
        #x+R(cos(theta)*e1+sin(theta(e2)))
        # Set the background color to black
        # self.win.setClearColor((0, 0, 0, 1))
        dt = globalClock.getDt()

        self.cnt=0
        # This is used to store which keys are currently pressed.
        self.keyMap = {
            "left": 0, "right": 0, "forward": 0, "cam-left": 0, "cam-right": 0, "cam1": 0, "cam2": 0, "cam3": 0, "cam4": 0, "cam5": 0, "unlock": 0,"wm1":0,"mup":0,"mright":0,"mleft":0,"mdown":0}


        self.accept("escape", sys.exit)
        self.accept("arrow_left", self.setKey, ["left", True])
        self.accept("arrow_right", self.setKey, ["right", True])
        self.accept("arrow_up", self.setKey, ["forward", True])
        self.accept("a", self.setKey, ["cam-left", True])
        self.accept("d", self.setKey, ["cam-right", True])
        self.accept("s", self.setKey, ["cam1", True])
        self.accept("w", self.setKey, ["cam2", True])
        self.accept("z", self.setKey, ["cam3", True])
        self.accept("x", self.setKey, ["cam4", True])
        self.accept("b", self.setKey, ["cam5", True])
        self.accept("h", self.setKey, ["wm1", True])
        self.accept("y", self.setKey, ["unlock", True])
        self.accept("j", self.setKey, ["mleft", True])
        self.accept("k", self.setKey, ["mdown", True])
        self.accept("l", self.setKey, ["mright", True])
        self.accept("i", self.setKey, ["mup", True])
        self.accept("arrow_left-up", self.setKey, ["left", False])
        self.accept("arrow_right-up", self.setKey, ["right", False])
        self.accept("arrow_up-up", self.setKey, ["forward", False])
        self.accept("a-up", self.setKey, ["cam-left", False])
        self.accept("s-up", self.setKey, ["cam1", False])
        self.accept("w-up", self.setKey, ["cam2", False])
        self.accept("d-up", self.setKey, ["cam-right", False])
        self.accept("z-up", self.setKey, ["cam3", False])
        self.accept("x-up", self.setKey, ["cam4", False])
        self.accept("h-up", self.setKey, ["wm1", False])
        self.accept("y-up", self.setKey, ["unlock", True])
        self.accept("j-up", self.setKey, ["mleft", False])
        self.accept("k-up", self.setKey, ["mdown", False])
        self.accept("l-up", self.setKey, ["mright", False])
        self.accept("i-up", self.setKey, ["mup", False])
        taskMgr.add(self.move, "moveTask")
        #
        self.disableMouse()
        camera.setPosHpr(0, -5, 0, 0, 10, 0)
        z=Point3(0,0,1)
        x=Point3(0,1,0)
        y=Point3(0,0,0)

        # b=normalizeTriangle(0,2,0)
        # print(b)
        # self.nWorm=wm(101,self.dt)
        # dlight=DirectionalLight('my dlight')
        # d1np=render.attachNewNode(dlight)
        # d1np.setPos(-5,-5,0)
        # d1np.lookAt(0,0,0)
        # camera.setPos( 0, 0, 50)
        # camera.lookAt( 0, 0, 0 )
# Create Ambient Light
        # ambientLight = AmbientLight('ambientLight')
        # ambientLight.setColor((0.1, 0.1, 0.1, 1))
        # ambientLightNP = render.attachNewNode(ambientLight)
        # render.setLight(ambientLightNP)

        # Directional light 01
        directionalLight = DirectionalLight('directionalLight')
        directionalLight.setColor((1, 1, 1, 1))
        directionalLightNP = render.attachNewNode(directionalLight)
        # This light is facing backwards, towards the camera.
        # directionalLightNP.setPos(30,30,30)
        directionalLightNP.setPos(0.,60.,0.)
        directionalLightNP.setHpr( 180, -90,0)
        render.setLight(directionalLightNP)

        # Directional light 02
        # directionalLight = DirectionalLight('directionalLight')
        # directionalLight.setColor((0.2, 0.2, 0.8, 1))
        # directionalLightNP = render.attachNewNode(directionalLight)
        # # This light is facing forwards, away from the camera.
        # directionalLightNP.setHpr(0, -20, 0)
        render.setLight(directionalLightNP)
        #
        # # Now attach a green light only to object x.
        ambient = AmbientLight('ambient')
        ambient.setColor((0.05,0.05, 0.2, 1))
        alnp=render.attachNewNode(ambient)
        render.setLight(alnp)
        # plight = PointLight('plight')
        # plight.setColor((0.2, 0.2, 0.2, 1))
        # plnp = render.attachNewNode(plight)
        # plnp.setPos(10, 20, 20)
        # render.setLight(plnp)

        # If we did not call setLightOff() first, the green light would add to
        # the total set of lights on this object. Since we do call
        # setLightOff(), we are turning off all the other lights on this
        # object first, and then turning on only the green light.

        self.dt = globalClock.getDt()
        # world=BulletWorld()
        # shape=BulletSphereShape(10.5)
        # node=BulletRigidBodyNode('Sphere')
        # node.addShape(shape)
        # np=render.attachNewNode(node)
        # np.setPos(1.,1.,1.)
        # world.attachRigidBody(node)
        # nWorm=wm(101,self.dt)
        #x+R(cos(theta)*e1+sin(theta(e2)))

        self.eachb=[]
        self.eachc=[]
        self.eachd=[]
        self.x=0.0
        self.y=0.0
        self.z=0.0
        self.cright=1
        self.cleft=1
        self.cup=1
        self.cdown=1
        self.sunrad=3
        self.moons=[]

        self.R=0.0
        # self.world=BulletWorld()
        # self.worldNP=render.attachNewNode('World')
        self.myMaterial=Material()

        self.myMaterial.setShininess(0.3*128)
        self.myMaterial.setAmbient((0.05375 	,0.05 ,	0.06625,1))
        self.myMaterial.setDiffuse((0.18275 ,	0.17 ,	0.22525  ,1))
        self.myMaterial.setSpecular(	(0.332741 	,0.328634 ,	0.346435 ,1))
        self.c=0
        self.vn='triangle'+str(self.c)
         	#
        # #
        # terrain = GeoMipTerrain("mySimpleTerrain")
        # terrain.setHeightfield("models/snow_field_aerial_height_2k.jpg")
        # #terrain.setBruteforce(True)
        # terrain.getRoot().reparentTo(render)
        # terrain.generate()
        g = loader.loadModel("models/sand/snd.bam")
        g.setPos(0,0.,-0.2)
        g.reparentTo(render)

        self.s = loader.loadModel("models/sun/sun.bam")
        self.s.setPos(1,1,0.5)
        self.s.reparentTo(render)

        self.m = loader.loadModel("models/moon/moon.bam")
        self.m.setPos(-1,-1,0.5)
        self.m.reparentTo(render)
        self.moons.append(self.m.getPos())
        sk = loader.loadModel("models/sky/sky.bam")
        sk.setPos(-1,-1,-80)
        sk.reparentTo(render)


    def create_triangle(self):



        _format = GeomVertexFormat.get_v3n3()
        vdata = GeomVertexData(self.vn, _format, Geom.UHStatic)


        # texcoord = GeomVertexWriter(vdata, 'texcoord')
        vertex = GeomVertexWriter(vdata, 'vertex')
        # vert = GeomVertexReader(vdata, 'vertex')
        normal = GeomVertexWriter(vdata, 'normal')
        # color = GeomVertexWriter(vdata, 'color')
        i=0
        tris = GeomTriangles(Geom.UHStatic)
        k0=0
        k2=2
        k1=1
        #929
        for i in range(929):
            p0 = Point3(self.eachb[i],self.eachc[i],self.eachd[i]+0.5)
            p1 = Point3(self.eachb[i+10], self.eachc[i+10],self.eachd[i+10]+0.5)
            p2 = Point3(self.eachb[i+1],self.eachc[i+1],self.eachd[i+1]+0.5)


            vertex.addData3(p0)
            # else:
            #     vertex.setData3(p0)
            vertex.addData3(p1)
            vertex.addData3(p2)

            normal.addData3(0,0,1)
            normal.addData3(0,0,1)
            normal.addData3(0,0,1)
   
            tris.addVertices(k0,k1,k2)


            k0=k0+3
            k1=k1+3
            k2=k2+3

            p0 = Point3(self.eachb[i+1],self.eachc[i+1],self.eachd[i+1]+0.5)
            p1 = Point3(self.eachb[i+10], self.eachc[i+10],self.eachd[i+10]+0.5)
            p2 = Point3(self.eachb[i+11],self.eachc[i+11],self.eachd[i+11]+0.5)


            # else:
            vertex.addData3(p0)
            vertex.addData3(p1)
            vertex.addData3(p2)
            #
            # x=[0,0,0]

            normal.addData3(0,0,1)
            normal.addData3(0,0,1)
            normal.addData3(0,0,1)
              			

            tris.addVertices(k0,k1,k2)


            k0=k0+3
            k1=k1+3
            k2=k2+3



        triangle = Geom(vdata)
        triangle.addPrimitive(tris)

        return triangle
    def newCircle(self,cn1,arr1,arr2,arr3):

        numSteps=10
        appendb=self.eachb.append
        appendc=self.eachc.append
        appendd=self.eachd.append
        # R=0.005
        if cn1<6:
            self.R=self.R+0.004
        # if cn1>91:
        #     self.R=0.001
        if cn1>88:
            self.R=0.001
        elif cn1>79:
            self.R=self.R-0.002
        # self.R=0.025

        i=0
        ls=LineSegs()
        for i in range(numSteps):
            a = 2 * math.pi * i / numSteps
            # a=360/101
            x = cos(a)
            y = sin(a)

            # b=newWorm.get_x()[cn1][0]+(newWorm.get_e1()[cn1][0]*x+newWorm.get_e2()[cn1][0]*y)*self.R
            b=arr1[cn1][0]+(arr2[cn1][0]*x+arr3[cn1][0]*y)*self.R

            # c=newWorm.get_x()[cn1][1]+(newWorm.get_e1()[cn1][1]*x+newWorm.get_e2()[cn1][1]*y)*self.R
            # d=newWorm.get_x()[cn1][2]+(newWorm.get_e1()[cn1][2]*x+newWorm.get_e2()[cn1][2]*y)*self.R
            c=arr1[cn1][1]+(arr2[cn1][1]*x+arr3[cn1][1]*y)*self.R
            d=arr1[cn1][2]+(arr2[cn1][2]*x+arr3[cn1][2]*y)*self.R
            appendb(b)
            appendc(c)
            appendd(d)
        #     if i==0:
        #         ls.moveTo(b,c,d)
        #     else:
        #         ls.drawTo(b,c,d)
        #         ls.moveTo(b,c,d)
        # node=ls.create()
        # self.render.attachNewNode(node)
                    # print(b)
        #     ls.moveTo(b,c,d)
        #     ls.drawTo(b,c,d)
        # node=ls.create()
        # self.render.attachNewNode(node)
        # self.R=0.0


    def updateWorm(self,control,control2):
        self.eachb.clear()
        self.eachc.clear()
        self.eachd.clear()
        # print(len(self.eachb))
        # if control==0:
        #     self.nWorm.update(beta_pref=Expression('v',t=self.nWorm.t,v=control2,degree=1))
        # else:
        #     self.nWorm.update(alpha_pref=Expression('10*sin(2.0*pi*x[0]-t)+v',t=self.nWorm.t,v=control,degree=1))

        self.nWorm.update(alpha_pref=Expression('10*sin(2.0*pi*(x[0]-t))+v',t=self.nWorm.t,v=control,degree=1)),
        # self.nWorm.update(alpha_pref=Expression('v',t=self.nWorm.t,v=control,degree=1)),
        # beta_pref=Expression('v',t=self.nWorm.t,v=control2,degree=1))

        # self.newx.clear()
        # self.newe1.clear()
        # self.newe2.clear()
        # self.newx=self.nWorm.get_x()
        # self.newe1=self.nWorm.get_e1()
        # self.newe2=self.nWorm.get_e2()
        newx=[]
        newe1=[]
        newe2=[]
        newx.clear()
        newe1.clear()
        newe2.clear()
        newx=self.nWorm.get_x()
        newe1=self.nWorm.get_e1()
        newe2=self.nWorm.get_e2()
        self.camerapos=(newx[40][0],newx[40][1],newx[40][2])
        print(newx[4][0],newx[4][1],newx[4][2])
        a1=0
        # print(newx[0])
        # print(newx[9])
        # newCircle(a1)            # if cn1>91:
        #     self.R=0.001
        
        start=time.time()
        for a1 in range(98):
            self.newCircle(a1,newx,newe1,newe2)
            if abs(newx[a1][1]-self.s.getY())<=0.13 and abs(newx[a1][0]-self.s.getX())<=0.13:
            	self.sunrad=self.sunrad+1
            	self.s.setPos(random.randint(-self.sunrad,self.sunrad),random.randint(-self.sunrad,self.sunrad),0.5)
            	m = loader.loadModel("models/moon/moon.bam")
            	m.setPos(random.randint(-self.sunrad,self.sunrad),random.randint(-self.sunrad,self.sunrad),0.5)
            	m.reparentTo(render)
            	self.moons.append(m.getPos())
            for j in range(self.sunrad-2):
            	if abs(newx[a1][1]-self.moons[j][0])<=0.13 and abs(newx[a1][0]-self.moons[j][1])<=0.13:
            		print('Game over')
            		
            	
            	
        self.R=0.0
        end=time.time()
        print(end-start)
        # print(eachb)
        start2=time.time()
        i=0

        # for i in range(929):
        #     p0 = Point3(self.eachb[i],self.eachc[i],self.eachd[i])2.0*pi*
        #     p1 = Point3(self.eachb[i+10], self.eachc[i+10],self.eachd[i+10])
        #     p2 = Point3(self.eachb[i+1],self.eachc[i+1],self.eachd[i+1])
        #     tr1 = create_triangle(p0,p1,p2, (0.7, 0.3, 0, 1))
        #     gnode = GeomNode('triangle')
        #     gnode.addGeom(tr1)
        #     ship = self.render.attachNewNode(gnode)
        #     # ambientNP = ship.attachNewNode(ambient)
        #     # ship.setLightOff()
        #     # ship.setLight(ambientNP)
        #     # ship.setMaterial(myMaterial)
        #     p0 = Point3(self.eachb[i+1],self.eachc[i+1],self.eachd[i+1])
        #     p1 = Point3(self.eachb[i+10], self.eachc[i+10],self.eachd[i+10])
        #     p2 = Point3(self.eachb[i+11],self.eachc[i+11],self.eachd[i+11])
        #     tr2 = create_triangle(p0,p1,p2, (0.7, 0.3, 0, 1))
        #     gnode2 = GeomNode('triangle')
        gnode2 = GeomNode(self.vn)

        gnode2.addGeom(self.create_triangle())
        self.ship=self.render.attachNewNode(gnode2)


        camera.lookAt(self.ship)
        self.ship.setMaterial(self.myMaterial)

        end2=time.time()
        print(end2-start2)

        # self.animWorm=updateWorm(self,1,1)
        # self.animWorm2=updateWorm(self,4,1)
    # Records the state of the arrow keys
    def setKey(self, key, value):
        self.keyMap[key] = value


    def move(self, task):

        # Get the time that elapsed since last frame.  We multiply this with
        # the desired speed in order to find out with which distance to move
        # in order to achieve that desired speed.
        self.dt=globalClock.getDt()
        dt = globalClock.getDt()
        self.startg=0
        # If the camera-left key is pressed, move camera left.
        # If the camera-right key is pressed, move camera right.
        if self.keyMap["wm1"]:
            self.nWorm=wm(101,self.dt)
            self.updateWorm(1,0.05)
            self.ship.removeNode()
            self.updateWorm(1,0.05)
            self.cnt=1
            self.unlockvalue=1
            self.camera.setZ(self.camera, +10 * dt)
            self.startg=1
        if self.keyMap["cam-left"]:
            if self.cnt>0:
                self.camera.setX(self.camera, -10 * dt)
                camera.lookAt(self.camerapos)
        if self.keyMap["cam-right"]:
            if self.cnt>0:
                self.camera.setX(self.camera, +10 * dt)
                camera.lookAt(self.camerapos)
        if self.keyMap["cam1"]:
            if self.cnt>0:
                print(self.camerapos[2])            
                if self.camera.getZ()>0.5:
                    self.camera.setY(self.camera, +10 * dt)
                camera.lookAt(self.camerapos)
        if self.keyMap["cam2"]:
            if self.cnt>0:
                self.camera.setY(self.camera, -10 * dt)
                camera.lookAt(self.camerapos)
        if self.keyMap["cam3"]:
            if self.cnt>0:
                if self.camera.getZ()<5:
                    self.camera.setZ(self.camera, +10*dt)
                camera.lookAt(self.camerapos)
        if self.keyMap["cam4"]:
            if self.cnt>0:
                if self.camera.getZ()>0.5:
                    self.camera.setZ(self.camera, -10*dt)
                    camera.lookAt(self.camerapos)
        if self.keyMap["unlock"]:
            if self.cnt>0:
                if self.unlockvalue==1:
                    self.unlockvalue=0
                else:
                    self.unlockvalue=1
        if self.keyMap["mright"]:
            self.cdown=0.05
            self.cleft=1
            self.cup=0.05
            if self.cnt>0:
                self.ship.removeNode()

            self.cnt=self.cnt+1
            self.c=self.c+1
            self.vn='triangle'+str(self.c)
            self.updateWorm(self.cright,self.cup)
            # self.x=self.x+0.1
            if self.cright<6:
                self.cright=self.cright+0.5
            # self.cright=self.cright+2
            # self.ship.setPos(self.x,self.y,self.z)
            # camera.lookAt(self.ship)
            # if self.unlockvalue==1:
            camera.lookAt(self.camerapos)
            # else.
        if self.keyMap["mleft"]:
            self.cup=0.05
            self.cdown=0.05
            # self.cright=1
            # self.cleft=7
            # self.cright=1
            # print(self.ship)
            if self.cnt>0:
                self.ship.removeNode()
            # print(self.ship)
            self.cnt=self.cnt+1
            self.c=self.c+1
            self.vn='triangle'+str(self.c)
            self.updateWorm(self.cright,self.cup)
            # self.cleft=self.cleft+2
            # print(self.ship)
            if self.cright>-6:
                self.cright=self.cright-0.5
            # self.x=self.x-0.1
            # self.ship.setPos(self.x,self.y,self.z)
            # if self.unlockvalue==1:
            camera.lookAt(self.camerapos)
        if self.keyMap["mdown"]:
            self.cup=0.05
            self.cright=1
            self.cleft=1

            if self.cnt>0:
                self.ship.removeNode()
            # print(self.ship)
            self.cnt=self.cnt+1
            self.c=self.c+1
            self.vn='triangle'+str(self.c)
            self.updateWorm(0,-self.cdown)
            if self.cdown<10:
                self.cdown=self.cdown+1
            if self.unlockvalue==1:
                camera.lookAt(self.camerapos)
        if self.keyMap["mup"]:
            self.cdown=0.05
            self.cright=1
            self.cleft=1
            if self.cnt>0:
                self.ship.removeNode()
            # print(self.ship)
            self.cnt=self.cnt+1
            self.c=self.c+1
            self.vn='triangle'+str(self.c)
            self.updateWorm(0,self.cup)
            if self.cup<10:
                self.cup=self.cup+1

            camera.lookAt(self.ship)
        return task.cont



demo = WormDemo()
demo.run()
