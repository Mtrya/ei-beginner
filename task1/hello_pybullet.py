import pybullet as p  # pyright: ignore[reportMissingImports]
import pybullet_data  # pyright: ignore[reportMissingImports]
import time

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.setGravity(0, 0, -10)

planeId = p.loadURDF("plane.urdf")

startPos = [0, 0, 1]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])
boxId = p.loadURDF("cube.urdf", startPos, startOrientation)

for i in range(1000):
    p.stepSimulation()
    time.sleep(0.01)

p.disconnect()