
#from pandac.PandaModules import loadPrcFileData
#loadPrcFileData('', 'load-display tinydisplay')
#loadPrcFileData('', 'bullet-enable-contact-events true')

import sys

import math
# import worm.py
from math import *
# import direct.directbase.DirectStart
#
# from direct.showbase.DirectObject import DirectObject
from direct.showbase.InputStateGlobal import inputState

from direct.showbase.ShowBase import ShowBase
from panda3d.core import GeomNode
import worm
from worm import Worm as wm
from worm import Geometry
import math
from numpy.core.umath import deg2rad
from math import pi, sin, cos
from panda3d.core import RopeNode
from panda3d.core import NurbsCurveEvaluator
from panda3d.bullet import BulletSoftBodyNode
from panda3d.core import AmbientLight
from panda3d.core import DirectionalLight
from panda3d.core import Vec3
from panda3d.core import Vec4
from panda3d.core import Point3
from panda3d.core import TransformState
from panda3d.core import BitMask32
from panda3d.bullet import BulletHelper
from panda3d.bullet import BulletWorld
from panda3d.bullet import BulletPlaneShape
from panda3d.bullet import BulletBoxShape
from panda3d.bullet import BulletRigidBodyNode
from panda3d.bullet import BulletDebugNode
from panda3d.bullet import BulletSphereShape
from panda3d.bullet import BulletCapsuleShape
from panda3d.bullet import BulletCylinderShape
from panda3d.bullet import BulletConeShape
from panda3d.bullet import BulletConvexHullShape
from panda3d.bullet import BulletTriangleMesh
from panda3d.bullet import BulletTriangleMeshShape
from panda3d.bullet import BulletMultiSphereShape
from panda3d.bullet import XUp
from panda3d.bullet import YUp
from panda3d.bullet import ZUp

from panda3d.core import *
class WormDemo(ShowBase):

  def __init__(self):
    # sys.path.insert(0, "../../")
    # sys.path.insert(0, "../../RenderPipeline")
    #     # Import the main render pipeline class
    # from rpcore import RenderPipeline, SpotLight
    # world = BulletWorld()
    #     # Construct and create the pipeline
    # self.render_pipeline = RenderPipeline()
    # self.render_pipeline.create(self)
    # from rpcore.util.movement_controller import MovementController
    # self.render_pipeline.daytime_mgr.time = 0.769
    # self.controller = MovementController(self)
    # self.controller.set_initial_position(
    # Vec3(6.6, -18.8, 4.5), Vec3(4.7, -16.7, 3.4))
    # self.controller.setup()

    # base.setBackgroundColor(0.1, 0.1, 0.8, 1)
    # sys.path.insert(0, "../../")
    # sys.path.insert(0, "../../RenderPipeline")
    #
    # from rpcore import RenderPipeline, SpotLight
    # self.render_pipeline = RenderPipeline()
    # self.render_pipeline.create(self)
    # base.setFrameRateMeter(True)

    # base.cam.setPos(0, -20, 4)
    # base.cam.lookAt(0, 0, 0)
    ShowBase.__init__(self)
    self.keyMap = {
        "left": 0, "right": 0, "forward": 0, "cam-left": 0, "cam-right": 0, "cam1": 0, "cam2": 0, "cam3": 0, "cam4": 0, "cam5": 0}


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
    self.accept("arrow_left-up", self.setKey, ["left", False])
    self.accept("arrow_right-up", self.setKey, ["right", False])
    self.accept("arrow_up-up", self.setKey, ["forward", False])
    self.accept("a-up", self.setKey, ["cam-left", False])
    self.accept("s-up", self.setKey, ["cam1", False])
    self.accept("w-up", self.setKey, ["cam2", False])
    self.accept("d-up", self.setKey, ["cam-right", False])
    self.accept("z-up", self.setKey, ["cam3", False])
    self.accept("x-up", self.setKey, ["cam4", False])
    taskMgr.add(self.move, "moveTask")

    self.disableMouse()
    camera.setPosHpr(0, -8, 2.5, 0, -9, 0)
    # Light
    alight = AmbientLight('ambientLight')
    alight.setColor(Vec4(0.5, 0.5, 0.5, 1))
    alightNP = render.attachNewNode(alight)

    dlight = DirectionalLight('directionalLight')
    dlight.setDirection(Vec3(1, 1, -1))
    dlight.setColor(Vec4(0.7, 0.7, 0.7, 1))
    dlightNP = render.attachNewNode(dlight)

    render.clearLight()
    render.setLight(alightNP)
    render.setLight(dlightNP)

    # Input
    # self.accept('escape', self.doExit)
    # self.accept('r', self.doReset)
    # self.accept('f1', self.toggleWireframe)
    # self.accept('f2', self.toggleTexture)
    # self.accept('f3', self.toggleDebug)
    # self.accept('f5', self.doScreenshot)
    #
    # inputState.watchWithModifiers('forward', 'w')
    # inputState.watchWithModifiers('left', 'a')
    # inputState.watchWithModifiers('reverse', 's')
    # inputState.watchWithModifiers('right', 'd')
    # inputState.watchWithModifiers('turnLeft', 'q')
    # inputState.watchWithModifiers('turnRight', 'e')

    # Task
    # taskMgr.add(self.update, 'updateWorld')

    # Physics
    self.setup()

  # _____HANDLER_____
    #
    # def doExit(self):
    #     self.cleanup()
    #     sys.exit(1)
    #
    # def doReset(self):
    #     self.cleanup()
    #     self.setup()
    #
    # def toggleWireframe(self):
    #     base.toggleWireframe()
    #
    # def toggleTexture(self):
    #     base.toggleTexture()
    #
    # def toggleDebug(self):
    #     if self.debugNP.isHidden():
    #       self.debugNP.show()
    #     else:
    #       self.debugNP.hide()
    #
    # def doScreenshot(self):
    #     base.screenshot('Bullet')
    #
    # # ____TASK___
    #
    # def processInput(self, dt):
    #     force = Vec3(0, 0, 0)
    #     torque = Vec3(0, 0, 0)
    #
    #     if inputState.isSet('forward'): force.setY( 1.0)
    #     if inputState.isSet('reverse'): force.setY(-1.0)
    #     if inputState.isSet('left'):    force.setX(-1.0)
    #     if inputState.isSet('right'):   force.setX( 1.0)
    #     if inputState.isSet('turnLeft'):  torque.setZ( 1.0)
    #     if inputState.isSet('turnRight'): torque.setZ(-1.0)
    #
    #     force *= 30.0
    #     torque *= 10.0

    # self.boxNP.node().setActive(True)
    # self.boxNP.node().applyCentralForce(force)
    # self.boxNP.node().applyTorque(torque)


  #def doAdded(self, node1, node2):
  #  print 'added:', node1.getName(), node2.getName()

  #def doDestroyed(self, node1, node2):
  #  print 'destroyed:', node1.getName(), node2.getName()

  def setup(self):
    self.worldNP = render.attachNewNode('World')

    # World
    self.debugNP = self.worldNP.attachNewNode(BulletDebugNode('Debug'))
    self.debugNP.show()
    self.debugNP.node().showWireframe(True)
    self.debugNP.node().showConstraints(True)
    self.debugNP.node().showBoundingBoxes(False)
    self.debugNP.node().showNormals(True)

    self.world = BulletWorld()
    self.world.setGravity(Vec3(0, 0, -9.81))
    self.world.setDebugNode(self.debugNP.node())

    # Plane (static)
    shape = BulletPlaneShape(Vec3(0, 0, 1), 0)

    np = self.worldNP.attachNewNode(BulletRigidBodyNode('Ground'))
    np.node().addShape(shape)
    np.setPos(0, 0, -1)
    np.setCollideMask(BitMask32.allOn())

    self.world.attachRigidBody(np.node())


    newWorm=wm(101,1e-3)
    #x+R(cos(theta)*e1+sin(theta(e2)))

    eachb=[]
    eachc=[]
    eachd=[]
    z=0
    self.world=BulletWorld()
    self.worldNP=render.attachNewNode('World')
    def newCircle(cn1):
        angleDegrees=360
        numSteps=10
        ls = LineSegs()
        # R=1/40
        angleRadians = deg2rad(angleDegrees)
        for i in range(numSteps):
            a = angleRadians * i / numSteps
            # a=360/101
            x = math.cos(a)
            y = math.sin(a)

            b=newWorm.get_x()[cn1][0]+(newWorm.get_e1()[cn1][0]*x+newWorm.get_e2()[cn1][0]*y)/40
            c=newWorm.get_x()[cn1][1]+(newWorm.get_e1()[cn1][1]*x+newWorm.get_e2()[cn1][1]*y)/40
            d=newWorm.get_x()[cn1][2]+(newWorm.get_e1()[cn1][2]*x+newWorm.get_e2()[cn1][2]*y)/40
            eachb.append(b)
            eachc.append(c)
            eachd.append(d)

            if i!=0:
                ls.drawTo(b, c, d)
            ls.moveTo(b,c,d)



        node = ls.create()
        render.attachNewNode( node )

    a1=0
    i=0
    b=0

    # newCircle(a1)
    for a1 in range(9):
        newCircle(a1)
    # print(eachb)
    z=40
    for b in range(z):
        print(eachd[b])
    p0 = Point3(eachb[i],eachc[i],eachd[i])
    p1 = Point3(eachb[i+10], eachc[i+10],eachd[i+10])
    p2 = Point3(eachb[i+1],eachc[i+1],eachd[i+1])

    p3 = Point3(10, 10, 0)
    mesh = BulletTriangleMesh()
    mesh.addTriangle(p0, p1, p2)
    # mesh.addGeom(geom)
    # mesh.addTriangle(p1, p2, p3)
    shape = BulletTriangleMeshShape(mesh, dynamic=False)

    np = self.worldNP.attachNewNode(BulletRigidBodyNode('Mesh'))
    np.node().addShape(shape)
    np.setPos(0, 0, 0)
    np.setCollideMask(BitMask32.allOn())

    self.world.attachRigidBody(np.node())


    info = self.world.getWorldInfo()
    info.setAirDensity(1.2)
    info.setWaterDensity(0)
    info.setWaterOffset(0)
    info.setWaterNormal(Vec3(0, 0, 0))

    # Softbody
    center = Point3(0, 0, 0)
    radius = Vec3(1, 1, 1) * 1.5

    bodyNode = BulletSoftBodyNode.makeEllipsoid(info, center, radius, 128)
    bodyNode.setName('Ellipsoid')
    bodyNode.getMaterial(0).setLinearStiffness(0.1)
    bodyNode.getCfg().setDynamicFrictionCoefficient(1)
    bodyNode.getCfg().setDampingCoefficient(0.001)
    bodyNode.getCfg().setPressureCoefficient(1500)
    bodyNode.setTotalMass(30, True)
    bodyNode.setPose(True, False)

    bodyNP = render.attachNewNode(bodyNode)
    bodyNP.setPos(-20, -10, -20)
    bodyNP.setH(90.0)
    self.world.attachSoftBody(bodyNP.node())
    fmt = GeomVertexFormat.getV3n3t2()

    geom = BulletHelper.makeGeomFromFaces(bodyNode, fmt)
    bodyNode.linkGeom(geom)
    visNode = GeomNode('EllipsoidVisual')
    visNode.addGeom(geom)
    visNP = bodyNP.attachNewNode(visNode)

  def setKey(self, key, value):
    self.keyMap[key] = value
  def camLoop(self,task):
    # check if the mouse is available
    if not base.mouseWatcherNode.hasMouse():
        return Task.cont

    # get the relative mouse position,
    # its always between 1 and -1
    mpos = base.mouseWatcherNode.getMouse()

    if mpos.getX() > 0.9:
        self.cameraTurn(1)

    elif mpos.getX() < -0.9:
        self.cameraTurn(-1)

    return Task.cont
  # def update(self, task):
  #   dt = globalClock.getDt()
  #
  #   self.processInput(dt)
  #   self.world.doPhysics(dt)

  def move(self, task):

    # Get the time that elapsed since last frame.  We multiply this with
    # the desired speed in order to find out with which distance to move
    # in order to achieve that desired speed.
    dt = globalClock.getDt()

    # If the camera-left key is pressed, move camera left.
    # If the camera-right key is pressed, move camera right.

    if self.keyMap["cam-left"]:
        self.camera.setX(self.camera, -10 * dt)
    if self.keyMap["cam-right"]:
        self.camera.setX(self.camera, +10 * dt)
    if self.keyMap["cam1"]:
        self.camera.setY(self.camera, +10 * dt)
    if self.keyMap["cam2"]:
        self.camera.setY(self.camera, -10 * dt)
    if self.keyMap["cam3"]:
        self.camera.setZ(self.camera, +10*dt)
    if self.keyMap["cam4"]:
        self.camera.setZ(self.camera, -10*dt)
    return task.cont



    #pairs = [(mf.getNode0().getName(),
    #          mf.getNode1().getName())
    #  for mf in self.world.getManifolds() if mf.getNumManifoldPoints() > 0]
    #print pairs


  def cleanup(self):
    self.world = None
    self.worldNP.removeNode()
demo = WormDemo()
demo.run()
