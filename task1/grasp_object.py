import pybullet as p
import pybullet_data
import time
import numpy as np

# ==================== 全局配置 ====================
# Joint indices
arm_joints = [0, 1, 2, 3, 4, 5, 6]
gripper_joints = [7, 8, 11]
gripper_fingers = [9, 10]

home_pos = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
end_effector_link = 11

# ==================== 辅助函数 ====================
def set_joint_positions(robot_id, joints, positions, max_force=100, max_velocity=5):
    """设置关节位置
    如果夹爪移动过快，会导致物体被甩飞，所以必须限制移动速度
    """
    for joint_idx, pos in zip(joints, positions):
        p.setJointMotorControl2(
            bodyIndex=robot_id,
            jointIndex=joint_idx,
            controlMode=p.POSITION_CONTROL,
            targetPosition=pos,
            force=max_force,
            maxVelocity=max_velocity,
        )

def wait_for_reach(robot_id, joints, target_positions, threshold=0.01, max_steps=1000):
    for step in range(max_steps):
        p.stepSimulation()
        time.sleep(0.01)

        current_positions = []
        for joint_idx in joints:
            joint_state = p.getJointState(robot_id, joint_idx)
            current_positions.append(joint_state[0])

        error = np.abs(np.array(current_positions) - np.array(target_positions))
        if np.all(error < threshold):
            print(f"Reached target positions in {step} steps")
            return True

    print(f"Failed to reach target positions in {max_steps} steps")
    return False

def wait_until_stable(body_id, threshold=0.0001, max_steps=1000):
    prev_pos, _ = p.getBasePositionAndOrientation(body_id)

    for step in range(max_steps):
        p.stepSimulation()
        time.sleep(0.01)

        curr_pos, _ = p.getBasePositionAndOrientation(body_id)
        movement = np.linalg.norm(np.array(curr_pos) - np.array(prev_pos))

        if movement < threshold:
            print(f"Body {body_id} is stable in {step} steps")
            return curr_pos

        prev_pos = curr_pos

    print(f"Body {body_id} is not stable in {max_steps} steps")
    return prev_pos

def get_gripper_center(robot_id):
    left_finger_state = p.getLinkState(robot_id, 9)
    right_finger_state = p.getLinkState(robot_id, 10)

    left_pos = np.array(left_finger_state[0])
    right_pos = np.array(right_finger_state[0])
    return (left_pos + right_pos) / 2

def calculate_ik_with_constraints(robot_id, target_pos, target_orn, rest_poses=None):
    """IK求解器返回的解不够精确，会导致末端执行器实际位置与期望位置出现偏移，即使关节到达了目标角度，末端执行器也没有达到期望位置。
    通过更多参数来获得准确解。"""
    if rest_poses is None:
        rest_poses = home_pos
    
    lower_limits = []
    upper_limits = []
    joint_ranges = []

    for joint_idx in arm_joints:
        joint_info = p.getJointInfo(robot_id, joint_idx)
        lower_limits.append(joint_info[8])
        upper_limits.append(joint_info[9])
        joint_ranges.append(joint_info[9] - joint_info[8])

    joint_poses = p.calculateInverseKinematics(
        bodyUniqueId=robot_id,
        endEffectorLinkIndex=end_effector_link,
        targetPosition=target_pos,
        targetOrientation=target_orn,
        residualThreshold=1e-5,
        lowerLimits=lower_limits,
        upperLimits=upper_limits,
        jointRanges=joint_ranges,
        restPoses=rest_poses,
        maxNumIterations=1000,
    )
    return joint_poses[:7]

def move_to_position(robot_id, target_pos, target_orn, rest_poses=None, verbose=True,
    max_velocity=5, use_trajectory=False, num_waypoints=5):
    """移动到指定位置
    最大速度与轨迹规划都是为了解决夹爪移动过快将物体甩飞的问题
    """
    if verbose:
        print(f"Moving to position: {target_pos}")

    if use_trajectory:
        current_ee_state = p.getLinkState(robot_id, end_effector_link)
        current_pos = np.array(current_ee_state[0])
        current_orn = np.array(current_ee_state[1])

        target_pos_array = np.array(target_pos)

        for i in range(1, num_waypoints + 1):
            alpha = i / (num_waypoints + 1)

            waypoint_pos = current_pos + alpha * (target_pos_array - current_pos)
            waypoint_orn = p.getQuaternionSlerp(current_orn, target_orn, alpha)

            joint_poses = calculate_ik_with_constraints(robot_id, waypoint_pos, waypoint_orn, rest_poses)
            set_joint_positions(robot_id, arm_joints, joint_poses, max_velocity=max_velocity)
            wait_for_reach(robot_id, arm_joints, joint_poses)

    joint_poses = calculate_ik_with_constraints(robot_id, target_pos, target_orn, rest_poses)
    set_joint_positions(robot_id, arm_joints, joint_poses, max_velocity=max_velocity)
    return wait_for_reach(robot_id, arm_joints, joint_poses)

# ==================== 主函数 ====================
def pick_and_place(
    cube_start_pos=[0.5, 0, 0.5],
    target_pos=[0.5, 0, 0.8],
    cube_scaling=0.05,
    pre_grasp_height=0.5,
    lift_height=0.5,
    use_gui=True,
    verbose=True,
    max_velocity=3,
    use_trajectory=True,
    num_waypoints=5
):
    """
    执行完整的抓取-放置任务
    
    参数:
        cube_start_pos: 立方体的起始位置 [x, y, z]
        target_pos: 目标放置位置 [x, y, z]
        cube_scaling: 立方体的缩放比例
        pre_grasp_height: 预抓取位置在物体上方的距离 (米)
        lift_height: 抬起物体后的高度 (米)
        use_gui: 是否使用图形界面
        verbose: 是否打印详细信息
    
    返回:
        success: 是否成功完成
    """
    # 初始化仿真环境
    if use_gui:
        physicsClient = p.connect(p.GUI)
    else:
        physicsClient = p.connect(p.DIRECT)
    
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # 加载地面
    planeId = p.loadURDF("plane.urdf")

    # 加载带夹爪的机械臂 (Franka Panda)
    robotId = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True)

    # 加载待抓取物体 (立方体)
    cube_start_orn = p.getQuaternionFromEuler([0, np.pi, 0])
    cubeId = p.loadURDF("cube.urdf", cube_start_pos, cube_start_orn, globalScaling=cube_scaling)

    # Step1: 等待物体稳定
    if verbose:
        print("=" * 50)
        print("Step 1: Waiting for cube to stabilize")
        print("=" * 50)
    stable_cube_pos = wait_until_stable(cubeId)
    if verbose:
        print(f"Stable cube position: {stable_cube_pos}")

    # Step2: 移动到初始位置
    if verbose:
        print("\n" + "=" * 50)
        print("Step 2: Moving arm to home position")
        print("=" * 50)
    set_joint_positions(robotId, arm_joints, home_pos)
    wait_for_reach(robotId, arm_joints, home_pos)

    # Step3: 打开夹爪
    if verbose:
        print("\n" + "=" * 50)
        print("Step 3: Opening gripper")
        print("=" * 50)
    gripper_open = [0.04, 0.04]
    set_joint_positions(robotId, gripper_fingers, gripper_open)
    wait_for_reach(robotId, gripper_fingers, gripper_open)

    # Step4: 移动到物体上方（预抓取位置）
    if verbose:
        print("\n" + "=" * 50)
        print("Step 4: Moving to pre-grasp position")
        print("=" * 50)
    pre_grasp_pos = [stable_cube_pos[0], stable_cube_pos[1], stable_cube_pos[2] + pre_grasp_height]
    pre_grasp_orn = p.getQuaternionFromEuler([0, np.pi, 0])
    move_to_position(robotId, pre_grasp_pos, pre_grasp_orn, verbose=verbose)

    # Step5: 向下移动到物体位置
    if verbose:
        print("\n" + "=" * 50)
        print("Step 5: Moving down to grasp position")
        print("=" * 50)
    grasp_pos = [stable_cube_pos[0], stable_cube_pos[1], stable_cube_pos[2]]
    grasp_orn = p.getQuaternionFromEuler([0, np.pi, 0])
    move_to_position(robotId, grasp_pos, grasp_orn, verbose=verbose)

    # Step6: 闭合夹爪
    if verbose:
        print("\n" + "=" * 50)
        print("Step 6: Closing gripper")
        print("=" * 50)
    gripper_close = [0.0, 0.0]
    set_joint_positions(robotId, gripper_fingers, gripper_close)

    # 等待夹爪闭合
    for _ in range(100):
        p.stepSimulation()
        time.sleep(0.01)

    # Step7: 抬起物体
    if verbose:
        print("\n" + "=" * 50)
        print("Step 7: Lifting object")
        print("=" * 50)
    lift_pos = [stable_cube_pos[0], stable_cube_pos[1], stable_cube_pos[2] + lift_height]
    lift_orn = p.getQuaternionFromEuler([0, np.pi, 0])
    move_to_position(robotId, lift_pos, lift_orn, verbose=verbose, max_velocity=max_velocity, use_trajectory=use_trajectory, num_waypoints=num_waypoints)

    # Step8: 移动到目标位置
    if verbose:
        print("\n" + "=" * 50)
        print(f"Step 8: Moving object to target position {target_pos}")
        print("=" * 50)
    target_orn = p.getQuaternionFromEuler([0, np.pi, 0])
    move_to_position(robotId, target_pos, target_orn, verbose=verbose, max_velocity=max_velocity, use_trajectory=use_trajectory, num_waypoints=num_waypoints)

    # 短暂停留
    for _ in range(50):
        p.stepSimulation()
        time.sleep(0.01)
    
    # Step9: 打开夹爪释放物体
    if verbose:
        print("\n" + "=" * 50)
        print("Step 9: Opening gripper to release object")
        print("=" * 50)
    gripper_open = [0.04, 0.04]
    set_joint_positions(robotId, gripper_fingers, gripper_open)
    wait_for_reach(robotId, gripper_fingers, gripper_open)

    # 短暂停留
    for _ in range(50):
        p.stepSimulation()
        time.sleep(0.01)

    # Step10: 返回初始位置
    if verbose:
        print("\n" + "=" * 50)
        print("Step 10: Returning to home position")
        print("=" * 50)
    set_joint_positions(robotId, arm_joints, home_pos)
    wait_for_reach(robotId, arm_joints, home_pos)

    if verbose:
        print("\n" + "=" * 50)
        print("✓ Pick-and-place completed successfully!")
        print("=" * 50)

    # 保持仿真运行一段时间以便观察
    if verbose:
        print("\nHolding simulation for observation...")
    for _ in range(1000):
        p.stepSimulation()
        time.sleep(0.01)
    
    p.disconnect()

if __name__ == "__main__":
    pick_and_place(
        cube_start_pos=[0.4, 0.2, 0.5],
        target_pos=[0.4, -0.6, 0.1],
        cube_scaling=0.05,
        pre_grasp_height=0.5,
        lift_height=0.5,
        verbose=True,
        max_velocity=1.0,
        use_trajectory=True,
        num_waypoints=2
    )