from multiprocessing import JoinableQueue
import pybullet as p  #pyright: ignore[reportMissingImports]
import pybullet_data  #pyright: ignore[reportMissingImports]
import time

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
robotId = p.loadURDF("kuka_iiwa/model.urdf")
#robotId = p.loadURDF("franka_panda/panda.urdf")

numJoints = p.getNumJoints(robotId)
print(f"Number of joints: {numJoints}")

joint_types = {
    p.JOINT_REVOLUTE: "旋转关节",
    p.JOINT_PRISMATIC: "棱柱关节", 
    p.JOINT_FIXED: "固定关节",
    p.JOINT_POINT2POINT: "点对点关节",
    p.JOINT_SPHERICAL: "球关节"
}

for i in range(numJoints):
    jointInfo = p.getJointInfo(robotId, i)
    
    jointIndex = jointInfo[0]
    jointName = jointInfo[1]
    jointType = jointInfo[2]
    qIndex = jointInfo[3]
    uIndex = jointInfo[4]
    flags = jointInfo[5]
    jointDamping = jointInfo[6]
    jointFriction = jointInfo[7]
    jointLowerLimit = jointInfo[8]
    jointUpperLimit = jointInfo[9]
    jointMaxForce = jointInfo[10]
    jointMaxVelocity = jointInfo[11]
    linkName = jointInfo[12]
    jointAxis = jointInfo[13]
    parentFramePos = jointInfo[14]
    parentFrameOrn = jointInfo[15]
    parentIndex = jointInfo[16]
    
    print(f"关节 {jointIndex}: {jointName}")
    print(f"  类型: {joint_types.get(jointType, '未知')}")
    print(f"  连杆: {linkName}")
    print(f"  父关节索引: {parentIndex}")
    print(f"  运动范围: [{jointLowerLimit:.4f}, {jointUpperLimit:.4f}] {'rad' if jointType == p.JOINT_REVOLUTE else 'm'}")
    print(f"  最大力矩/力: {jointMaxForce:.2f}")
    print(f"  最大速度: {jointMaxVelocity:.2f}")
    print(f"  关节轴: {jointAxis}")
    print()

for i in range(10000):
    p.stepSimulation()
    time.sleep(0.01)

p.disconnect()