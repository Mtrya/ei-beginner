import pybullet as p
import pybullet_data
import time
import numpy as np

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
robotId = p.loadURDF("kuka_iiwa/model.urdf")

p.setGravity(0, 0, -9.81)

numJoints = p.getNumJoints(robotId)
print(f"Number of joints: {numJoints}")

movable_joints = []
for i in range(numJoints):
    jointInfo = p.getJointInfo(robotId, i)
    if jointInfo[2] == p.JOINT_REVOLUTE:
        movable_joints.append(i)

print(f"Movable joints: {movable_joints}")

# 方法1：位置控制
print(f"Method 1: Position Control")
print("-" * 50)
target_positions = [0.5, 0.3, -0.2, 0.8, -0.5, 0.0, 0.0]

for i, joint_idx in enumerate(movable_joints):
    if i < len(target_positions):
        p.setJointMotorControl2(
            bodyUniqueId=robotId,
            jointIndex=joint_idx,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target_positions[i],
            force=500,
            maxVelocity=1.0,
            positionGain=0.3, # kp
            velocityGain=1.0 # kd
        )
        # PyBullet内置了PD控制器：控制力矩 = kp * (target_position - current_position) - kd * current_velocity
        # But why not PID? Maybe there's issue with integration within a discrete-time physics engine.

for step in range(1000):
    p.stepSimulation()

    if step % 100 == 0:
        current_positions = []
        for joint_idx in movable_joints:
            joint_state = p.getJointState(robotId, joint_idx)
            current_positions.append(joint_state[0])

        print(f"Step {step}: Current positions: {current_positions}")
        print(f"Target positions: {target_positions}")
        print(f"Error: {np.abs(np.array(current_positions) - np.array(target_positions))}")
        print("-" * 50)

    time.sleep(0.01)

# 方法2：速度控制
print(f"Method 2: Velocity Control")
print("-" * 50)
joint_velocities = [0.1, -0.1, 0.05, -0.05, 0.08, -0.08, 0.03]
for step in range(1000):
    for i, joint_idx in enumerate(movable_joints):
        p.setJointMotorControl2(
            bodyUniqueId=robotId,
            jointIndex=joint_idx,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=joint_velocities[i],
        )

    if step % 100 == 0:
        current_positions = []
        for joint_idx in movable_joints:
            joint_state = p.getJointState(robotId, joint_idx)
            current_positions.append(joint_state[0])

        print(f"Step {step}: Current positions: {current_positions}")
        print(f"Target positions: {target_positions}")
        print(f"Error: {np.abs(np.array(current_positions) - np.array(target_positions))}")
        print("-" * 50)

    p.stepSimulation()
    time.sleep(0.01)

p.disconnect()