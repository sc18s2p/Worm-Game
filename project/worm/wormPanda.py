#!/usr/bin/env python



from direct.showbase.ShowBase import ShowBase

from panda3d.core import *
from direct.interval.IntervalGlobal import *
from direct.gui.DirectGui import *

import random
import numpy as np
from numpy import *
import worm
from worm import Worm as wm
from worm import Geometry

from math import pi, sin, cos

from panda3d.core import GeomNode
import math

import sys
import os



import time
from fenics import Expression
# Styling for instructions
def addInstructions(pos, msg):
    return OnscreenText(text=msg, style=1, fg=(1, 1, 1, 1), scale=.05,
                        shadow=(0, 0, 0, 1), parent=base.a2dTopLeft,
                        pos=(0.08, -pos - 0.04), align=TextNode.ALeft)




class WormDemo(ShowBase):

    def __init__(self):

        # Initialize the ShowBase class from which we inherit, which will
        # create a window and set up everything we need for rendering into it.
        ShowBase.__init__(self)



        # On screen title
        self.title = OnscreenText(text="Panda3D:Worm",
                                  parent=base.a2dBottomCenter,
                                  fg=(1, 1, 1, 1), shadow=(0, 0, 0, .5),
                                  pos=(0, .1), scale=.1)
        # Post Score
        self.s1 = OnscreenText(text="Score:",
                                  parent=base.a2dTopLeft,
                                  fg=(1, 1, 1, 1), shadow=(0, 0, 0, .5),
                                  pos=(0.1, 0.1), scale=.1,align=TextNode.ALeft)
        # Post the instructions
        self.inst1 = addInstructions(0.06, "[ESC]: Quit")
        self.inst2 = addInstructions(0.12, "[J]: Rotate Worm Left")
        self.inst3 = addInstructions(0.18, "[L]: Rotate Worm Right")
        self.inst4 = addInstructions(0.24, "[W]: Zoom Out")
        self.inst5 = addInstructions(0.30, "[S]: Zoom In")
        self.inst6 = addInstructions(0.36, "[A]: Rotate Camera Left")
        self.inst7 = addInstructions(0.42, "[D]: Rotate Camera Right")
        self.inst8 = addInstructions(0.48, "[Z]: Rotate Camera Upwards")
        self.inst9 = addInstructions(0.54, "[X]: Rotate Camera Downwards")

        # Count that is set to 1 when the worm is first animated
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

        #Disable the mouse to discard movement of camera with mouse
        self.disableMouse()

        # Set the Camera position and where it looks
        camera.setPosHpr(0, -5, 0, 0, 10, 0)

        # Directional light that looks towards the ground
        directionalLight = DirectionalLight('directionalLight')
        directionalLight.setColor((1, 1, 1, 1))
        directionalLightNP = render.attachNewNode(directionalLight)
        directionalLightNP.setPos(0.,60.,0.)
        directionalLightNP.setHpr( 180, -90,0)
        render.setLight(directionalLightNP)

        # Lighting for whole scene so that the sky can be visible
        ambient = AmbientLight('ambient')
        ambient.setColor((0.05,0.05, 0.2, 1))
        alnp=render.attachNewNode(ambient)
        render.setLight(alnp)


        # Global Time
        self.dt = globalClock.getDt()

        # Initialize score of player
        self.score=0
        # Arrays to store the points of the worm
        self.point1array=[]
        self.point2array=[]
        self.point3array=[]

        # value of worm's steering wheel
        self.steer=1

        # The radius in which the planets spawn in
        self.sunrad=3

        # Array to store the moons' positions
        self.moons=[]

        # Radius of the worm
        self.R=0.0

        # Material of the worm
        self.myMaterial=Material()
        # Properties for obsedian masterial
        self.myMaterial.setShininess(0.3*128)
        self.myMaterial.setAmbient((0.05375 	,0.05 ,	0.06625,1))
        self.myMaterial.setDiffuse((0.18275 ,	0.17 ,	0.22525  ,1))
        self.myMaterial.setSpecular(	(0.332741 	,0.328634 ,	0.346435 ,1))

        self.c=0
        self.vn='wormShape'+str(self.c)

        def ModelSet(name="set1"):
            return NodePath(name)
        self.set1 = ModelSet("set1")
        self.set1.reparentTo(self.render)
        # load the ground model
        g = loader.loadModel("models/sand/snd.bam")
        g.setPos(0,0.,-0.2)
        g.reparentTo(render)


        # Load the sun model
        self.s = loader.loadModel("models/sun/sun.bam")
        self.s.setPos(1,1,0.5)
        self.s.reparentTo(render)

        # loaf the moon model
        self.m = loader.loadModel("models/moon/moon.bam")
        self.m.setPos(-1,-1,0.5)
        self.m.reparentTo(render)
        # update this moon's position to the array
        self.moons.append(self.m.getPos())

        # load the sky model
        sk = loader.loadModel("models/sky/sky.bam")
        sk.setPos(-1,-1,-80)
        sk.reparentTo(render)


    def create_wormShape(self):
        # set the format for the shape
        # v3c3 asks for 3 vectors and 3 normals
        _format = GeomVertexFormat.get_v3n3()
        vdata = GeomVertexData(self.vn, _format, Geom.UHStatic)


        # texcoord = GeomVertexWriter(vdata, 'texcoord')
        vertex = GeomVertexWriter(vdata, 'vertex')
        # vert = GeomVertexReader(vdata, 'vertex')
        normal = GeomVertexWriter(vdata, 'normal')
        # color = GeomVertexWriter(vdata, 'color')

        tris = GeomTriangles(Geom.UHStatic)

        k0=0
        k2=2
        k1=1
        # add the points to the vertices
        for i in range(929):
            # First set of triangles
            p0 = Point3(self.point1array[i],self.point2array[i],self.point3array[i]+0.5)
            p1 = Point3(self.point1array[i+10], self.point2array[i+10],self.point3array[i+10]+0.5)
            p2 = Point3(self.point1array[i+1],self.point2array[i+1],self.point3array[i+1]+0.5)


            vertex.addData3(p0)
            vertex.addData3(p1)
            vertex.addData3(p2)

            # add the normals of the vertices
            normal.addData3(0,0,1)
            normal.addData3(0,0,1)
            normal.addData3(0,0,1)

            tris.addVertices(k0,k1,k2)


            k0=k0+3
            k1=k1+3
            k2=k2+3
            # Second set of triangles facing the opposite way
            p0 = Point3(self.point1array[i+1],self.point2array[i+1],self.point3array[i+1]+0.5)
            p1 = Point3(self.point1array[i+10], self.point2array[i+10],self.point3array[i+10]+0.5)
            p2 = Point3(self.point1array[i+11],self.point2array[i+11],self.point3array[i+11]+0.5)

            vertex.addData3(p0)
            vertex.addData3(p1)
            vertex.addData3(p2)

            normal.addData3(0,0,1)
            normal.addData3(0,0,1)
            normal.addData3(0,0,1)


            tris.addVertices(k0,k1,k2)


            k0=k0+3
            k1=k1+3
            k2=k2+3


        # Create the shape
        wormShape = Geom(vdata)
        wormShape.addPrimitive(tris)

        return wormShape

    def newCircle(self,cn1,arr1,arr2,arr3):
        # number of points for a circle
        numSteps=10
        appendb=self.point1array.append
        appendc=self.point2array.append
        appendd=self.point3array.append

        # Radius for the head of the worm
        if cn1<5:
            self.R=0.001
        elif cn1<15:
            self.R=self.R+0.002
        # Radius for the tail of the worm
        if cn1>90:
            self.R=self.R-0.004


        i=0

        for i in range(numSteps):
            # angle between two points
            a = 2 * math.pi * i / numSteps

            x = cos(a)
            y = sin(a)

            # equations to create points of the circle
            b=arr1[cn1][0]+(arr2[cn1][0]*x+arr3[cn1][0]*y)*self.R
            c=arr1[cn1][1]+(arr2[cn1][1]*x+arr3[cn1][1]*y)*self.R
            d=arr1[cn1][2]+(arr2[cn1][2]*x+arr3[cn1][2]*y)*self.R
            # the points are added to the arrays
            appendb(b)
            appendc(c)
            appendd(d)



    def updateWorm(self,control):
        # clear the arrays for the points of the new worm
        self.point1array.clear()
        self.point2array.clear()
        self.point3array.clear()

        # Update worm with new control to return new values
        self.nWorm.update(alpha_pref=Expression('10*sin(2.0*pi*(x[0]-t))+v',t=self.nWorm.t,v=control,degree=1)),

        newx=[]
        newe1=[]
        newe2=[]
        # newx.clear()
        # newe1.clear()
        # newe2.clear()

        # Add the outputs of the update function to the arrays
        newx=self.nWorm.get_x()
        newe1=self.nWorm.get_e1()
        newe2=self.nWorm.get_e2()
        # set the camera to look at the middle of the worm
        self.camerapos=(newx[50][0],newx[50][1],newx[50][2])

        #Value to determine whether the game is lost
        lose=0
        for a1 in range(98):
            self.newCircle(a1,newx,newe1,newe2)
            # compare coordinates of radius of sun with each point of the worm
            if abs(newx[a1][1]-self.s.getY())<=0.13 and abs(newx[a1][0]-self.s.getX())<=0.13:
                # increment the spawn radius of planets
            	self.sunrad=self.sunrad+1
                # set position of sun to a random location within the radius
            	self.s.setPos(random.randint(-self.sunrad,self.sunrad),random.randint(-self.sunrad,self.sunrad),0.5)
                # upgrade user's score
            	self.score=self.score+self.sunrad*10
            	print('Score: ',self.score)
                # spawn a new mone to a random location within the radius
            	m = loader.loadModel("models/moon/moon.bam")
            	m.setPos(random.randint(-self.sunrad,self.sunrad),random.randint(-self.sunrad,self.sunrad),0.5)
            	m.reparentTo(self.set1)
                # append its location to the moons array
            	self.moons.append(m.getPos())

            for j in range(len(self.moons)):
                # compare coordinates of radius of moon with each point of the worm
            	if abs(newx[a1][1]-self.moons[j][1])<=0.13 and abs(newx[a1][0]-self.moons[j][0])<=0.13:
            		print('Game over')
            		print('Your final score is:', self.score)

            		lose=1
                    # start counter set to 0,score and radius are reset and the moons array clears
            		self.cnt=0
            		self.score=0
            		self.sunrad=0
            		self.set1.removeNode()
            		self.moons.clear()
            	if lose==1:
            		break
            if lose==1:
                break


        self.R=0.0
        # if the game is lost remove the worm node for the worm to disappear
        if lose==1:
            self.wNode.removeNode()
        else:
            # create a new Geom node
            gnode2 = GeomNode(self.vn)
            # add the geom shape to the node and at the node to the main renderer
            gnode2.addGeom(self.create_wormShape())
            self.wNode=self.render.attachNewNode(gnode2)


            camera.lookAt(self.wNode)
            self.wNode.setMaterial(self.myMaterial)


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

        if self.keyMap["wm1"]:
            # call the worm class from simulator file
            self.nWorm=wm(101,self.dt)
            self.updateWorm(1)
            self.wNode.removeNode()
            self.updateWorm(0)
            # increment cnt to signal start of game
            self.cnt=1

            self.camera.setZ(self.camera, +10 * dt)
            self.startg=1
        # turn camera left
        if self.keyMap["cam-left"]:
            if self.cnt>0:
                self.camera.setX(self.camera, -10 * dt)
                camera.lookAt(self.camerapos)
        # turn camera right
        if self.keyMap["cam-right"]:
            if self.cnt>0:
                self.camera.setX(self.camera, +10 * dt)
                camera.lookAt(self.camerapos)
        # Zoom in
        if self.keyMap["cam1"]:
            if self.cnt>0:
                if self.camera.getZ()>0.5:
                    self.camera.setY(self.camera, +10 * dt)
                camera.lookAt(self.camerapos)
        # Zoom out
        if self.keyMap["cam2"]:
            if self.cnt>0:
                self.camera.setY(self.camera, -10 * dt)
                camera.lookAt(self.camerapos)
        # Move camera up
        if self.keyMap["cam3"]:
            if self.cnt>0:
                if self.camera.getZ()<5:
                    self.camera.setZ(self.camera, +10*dt)
                camera.lookAt(self.camerapos)
        # Move camera down
        if self.keyMap["cam4"]:
            if self.cnt>0:
                if self.camera.getZ()>0.5:
                    self.camera.setZ(self.camera, -10*dt)
                    camera.lookAt(self.camerapos)
        # worm turns right
        if self.keyMap["mright"]:

            if self.cnt>0:
                # remove previous worm
                self.wNode.removeNode()

                self.cnt=self.cnt+1
                self.c=self.c+1
                self.vn='wormShape'+str(self.c)
                self.updateWorm(self.steer)
                # increase right steering until maximum value
                if self.steer<6:
              	     self.steer=self.steer+0.5

                camera.lookAt(self.camerapos)
        # worm turns left
        if self.keyMap["mleft"]:


            if self.cnt>0:
                self.wNode.removeNode()

                self.cnt=self.cnt+1
                self.c=self.c+1
                self.vn='wormShape'+str(self.c)
                self.updateWorm(self.steer)
                # increase left steering until maximum value
                if self.steer>-6:
                    self.steer=self.steer-0.5


                camera.lookAt(self.camerapos)



        return task.cont



demo = WormDemo()
demo.run()
