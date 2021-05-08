"""

Simplest possible application using the render pipeline.

This sample will not show any fancy rendering output, but you can base your own
applications on this skeleton.

This is the preferred way of initializing the pipeline, however you can find
alternative ways in the other included files.

"""

import sys


from panda3d.core import *
from direct.gui.OnscreenText import OnscreenText
from direct.actor.Actor import Actor
from direct.showbase.ShowBase import ShowBase

from panda3d.bullet import BulletCylinderShape
from panda3d.bullet import BulletCapsuleShape
from panda3d.bullet import ZUp
from panda3d.bullet import BulletCharacterControllerNode
from panda3d.bullet import BulletWorld

def create_colored_rect(x, z, width, height, colors=None):
    _format = GeomVertexFormat.getV3c4()
    vdata = GeomVertexData('square', _format, Geom.UHDynamic)

    vertex = GeomVertexWriter(vdata, 'vertex')
    color = GeomVertexWriter(vdata, 'color')

    vertex.addData3(x, 0, z)
    vertex.addData3(x + width, 0, z)
    vertex.addData3(x + width, 0, z + height)
    vertex.addData3(x, 0, z + height)

    if colors:
        if len(colors) < 4:
            colors = (1.0, 1.0, 1.0, 1.0)
        color.addData4f(colors)
        color.addData4f(colors)
        color.addData4f(colors)
        color.addData4f(colors)
    else:
        color.addData4f(1.0, 0.0, 0.0, 1.0)
        color.addData4f(0.0, 1.0, 0.0, 1.0)
        color.addData4f(0.0, 0.0, 1.0, 1.0)
        color.addData4f(1.0, 1.0, 1.0, 1.0)

    tris = GeomTriangles(Geom.UHDynamic)
    tris.addVertices(0, 1, 2)
    tris.addVertices(2, 3, 0)

    square = Geom(vdata)
    square.addPrimitive(tris)
    return square

def create_triangle(x, z, y, x1,z1,y1,x2,z2,y2, colors=None):
    _format = GeomVertexFormat.getV3c4()
    vdata = GeomVertexData('triangle', _format, Geom.UHDynamic)

    vertex = GeomVertexWriter(vdata, 'vertex')
    color = GeomVertexWriter(vdata, 'color')


    vertex.addData3(x, z, y)
    vertex.addData3(x1, z1, y1)
    vertex.addData3(x2, z2, y2)

    # vertex.addData3(x, 0, z + height)

    if colors:
        if len(colors) < 4:
            colors = (1.0, 1.0, 1.0, 1.0)
        color.addData4f(colors)
        color.addData4f(colors)
        color.addData4f(colors)
        # color.addData4f(colors)
    else:
        color.addData4f(1.0, 0.0, 0.0, 1.0)
        color.addData4f(0.0, 1.0, 0.0, 1.0)
        color.addData4f(0.0, 0.0, 1.0, 1.0)
        # color.addData4f(1.0, 1.0, 1.0, 1.0)

    tris = GeomTriangles(Geom.UHDynamic)
    tris.addVertices(0, 1, 2)
    # tris.addVertices(2, 3, 0)

    triangle = Geom(vdata)
    triangle.addPrimitive(tris)
    return triangle

def create_textured_rect(x, z, width, height):
    _format = GeomVertexFormat.getV3t2()
    vdata = GeomVertexData('square', _format, Geom.UHDynamic)

    vertex = GeomVertexWriter(vdata, 'vertex')
    texcoord = GeomVertexWriter(vdata, 'texcoord')

    vertex.addData3(x, 0, z)
    vertex.addData3(x + width, 0, z)
    vertex.addData3(x + width, 0, z + height)
    vertex.addData3(x, 0, z + height)

    texcoord.addData2f(0.0, 0.0)
    texcoord.addData2f(1.0, 0.0)
    texcoord.addData2f(1.0, 1.0)
    texcoord.addData2f(0.0, 1.0)

    tris = GeomTriangles(Geom.UHDynamic)
    tris.addVertices(0, 1, 2)
    tris.addVertices(2, 3, 0)

    square = Geom(vdata)
    square.addPrimitive(tris)
    return square


class Application(ShowBase):

    def __init__(self):

        # Notice that you must not call ShowBase.__init__ (or super), the
        # render pipeline does that for you. If this is unconvenient for you,
        # have a look at the other initialization possibilities.

        # Insert the pipeline path to the system path, this is required to be
        # able to import the pipeline classes. In case you placed the render
        # pipeline in a subfolder of your project, you have to adjust this.
        sys.path.insert(0, "../../")
        sys.path.insert(0, "../../RenderPipeline")

        # Import the main render pipeline class
        from rpcore import RenderPipeline, SpotLight
        world = BulletWorld()
        # Construct and create the pipeline
        self.render_pipeline = RenderPipeline()
        self.render_pipeline.create(self)
        from rpcore.util.movement_controller import MovementController
        self.render_pipeline.daytime_mgr.time = 0.769
        self.controller = MovementController(self)
        self.controller.set_initial_position(
            Vec3(6.6, -18.8, 4.5), Vec3(4.7, -16.7, 3.4))
        self.controller.setup()


        # Done! You can start setting up your application stuff as regular now.
        self.keyMap = {"left":0, "right":0, "forward":0, "backward":0, "cam-left":0, "cam-right":0}
        self.speed = 1.0
        base.win.setClearColor(Vec4(0,0,0,1))

        #
        # vertex_format = GeomVertexFormat.get_v3n3()
        # vertex_data = GeomVertexData("triangle_data", vertex_format, Geom.UH_static)
        #
        # pos_writer = GeomVertexWriter(vertex_data,"vertex")
        # normal_writer = GeomVertexWriter(vertex_data,"normal")
        # normal = Vec3(0., -1., 0.)
        #
        # pos_writer.add_data3(-1., 0., -1.)
        # pos_writer.add_data3(1., 0., -1.)
        # pos_writer.add_data3(0., 0., 1.)
        #
        # for _ in range(3):
        #     normal_writer.add_data3(normal)
        #
        # prim = GeomTriangles(Geom.UH_static)
        # prim.add_vertices(0, 1, 2)
        #
        # geom = Geom(vertex_data)
        # geom.add_primitive(prim)
        # node = GeomNode("my_triangle")
        # node.add_geom(geom)
        # triangle = NodePath(node)
        # triangle.reparent_to(some_other_nodepath)
        # square1 = create_colored_rect(0, 0, 200, 200)
        # square2 = create_colored_rect(350, 100, 200, 200, (0, 0, 1, 1))
        # square3 = create_colored_rect(-640, -360, 200, 200, (0, 1, 0, 1))
        self.tr1 = create_triangle(0,0,0,0,200,0,0,0,200, (0, 1, 0, 1))
        self.tr2 = create_triangle(-500,0,0,-300,200,0,-300,0,200, (0, 1, 0, 1))
        radius = 60
        height = 40
        shape1 = BulletCylinderShape(radius, height, ZUp)
        shape2 = BulletCylinderShape(Vec3(radius, 0, 0.5 * height), ZUp)
        self.gnode = GeomNode('square')
        # gnode.addGeom(square1)
        # gnode.addGeom(square2)
        # gnode.addGeom(square3)
        height = 1.75
        radius = 0.4
        shape = BulletCapsuleShape(radius, height - 2*radius, ZUp)
        self.gnode.addGeom(self.tr1)
        self.gnode.addGeom(self.tr2)
        playerNode = BulletCharacterControllerNode(shape, 0.4, 'Player')
        playerNP = self.render.attachNewNode(playerNode)
        playerNP.setPos(0, 0, 14)
        playerNP.setH(45)
        playerNP.setCollideMask(BitMask32.allOn())

        world.attachCharacter(playerNP.node())
        # self.tr1.setPos(400,400, 0)
        # self.render.attachNewNode(self.gnode)

        gnode2 = GeomNode('square2')
        textured_rect = create_textured_rect(-320, 0, 200, 280)
        gnode2.addGeom(textured_rect)

        texture = self.loader.loadTexture("assets/playte.png")
        ship = self.render.attachNewNode(gnode2)
        ship.setTransparency(TransparencyAttrib.MAlpha)
        ship.setTexture(texture)

        # self.ralph = Actor(tr1)

        self.floater = NodePath(PandaNode("floater"))
        self.floater.reparentTo(render)

        self.accept("escape", sys.exit)
        self.accept("a", self.setKey, ["left",1])
        self.accept("d", self.setKey, ["right",1])
        self.accept("w", self.setKey, ["forward",1])
        self.accept("p", self.setKey, ["backward",1])
        self.accept("arrow_left", self.setKey, ["cam-left",1])
        self.accept("arrow_right", self.setKey, ["cam-right",1])
        self.accept("a-up", self.setKey, ["left",0])
        self.accept("d-up", self.setKey, ["right",0])
        self.accept("w-up", self.setKey, ["forward",0])
        self.accept("s-up", self.setKey, ["backward",0])
        self.accept("arrow_left-up", self.setKey, ["cam-left",0])
        self.accept("arrow_right-up", self.setKey, ["cam-right",0])
        self.accept("=", self.adjustSpeed, [0.25])
        self.accept("+", self.adjustSpeed, [0.25])
        self.accept("-", self.adjustSpeed, [-0.25])

        taskMgr.add(self.move,"moveTask")

        # Game state variables
        self.isMoving = False

        # Set up the camera

        base.disableMouse()
        base.camera.setPos(5,5, 0)
        base.camLens.setFov(80)

    def setKey(self, key, value):
        self.keyMap[key] = value

    # Adjust movement speed
    def adjustSpeed(self, delta):
        newSpeed = self.speed + delta
        if 0 <= newSpeed <= 3:
          self.speed = newSpeed

    def move(self, task):

        # If the camera-left key is pressed, move camera left.
        # If the camera-right key is pressed, move camera right.

        # base.camera.lookAt(self.tr1)
        if (self.keyMap["cam-left"]!=0):
            base.camera.setX(base.camera, +100 * globalClock.getDt())
        if (self.keyMap["cam-right"]!=0):
            base.camera.setX(base.camera, -100 * globalClock.getDt())
        if (self.keyMap["backward"]!=0):
            self.gnode.removeNode()
            self.tr9 = create_triangle(100,0,0,0,100,0,0,0,100, (0, 1, 0, 1))
            self.tr8 = create_triangle(0,0,0,100,200,0,200,0,200, (0, 1, 0, 1))
            self.gnode = GeomNode('square')
            # gnode.addGeom(square1)
            # gnode.addGeom(square2)
            # gnode.addGeom(square3)
            self.gnode.addGeom(self.tr9)
            self.gnode.addGeom(self.tr8)
            # self.tr1.setPos(400,400, 0)
            self.render.attachNewNode(self.gnode)


        # startpos = self.gnode.getPos()
        return task.cont
        # lens = OrthographicLens()
        # lens.setFilmSize(1280, 720)
        # lens.setNearFar(-50, 50)
        # self.cam.setPos(0, 0, 0)
        # self.cam.node().setLens(lens)

Application().run()
