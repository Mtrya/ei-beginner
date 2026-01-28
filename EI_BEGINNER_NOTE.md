# EI-Beginner 学习笔记

## 任务一：基于传统运动学的机械臂物体抓取

### 1. 理论基础
本任务主要基于斯坦福《Introduction to Robotics: Mechanics and Control》第三章和MIT《Robotic Manipulation》相关内容，学习了以下核心概念：

- **坐标系与变换**：理解了世界坐标系（World Frame）、体坐标系（Body Frame）以及它们之间的变换矩阵。
- **正运动学 (Forward Kinematics)**：从关节角度计算末端执行器位置。
- **逆运动学 (Inverse Kinematics)**：从末端执行器目标位置反推关节角度。特别是理解了 PyBullet 中 `calculateInverseKinematics` 的使用及其背后的数学原理（如雅可比矩阵、伪逆、阻尼最小二乘法等）。
- **空间代数**：旋转矩阵、欧拉角、RPY角以及四元数（Quaternion）的转换与应用，特别是四元数如何避免万向节死锁（Gimbal Lock）。

### 2. 代码实现 (`task1/`)

#### 2.1 核心脚本
- `hello_pybullet.py`: PyBullet 环境的基础搭建与测试。
- `load_arm.py`: 加载 Franka Panda 机械臂与环境。
- `control_arm.py`: 机械臂的基础控制逻辑。
- `grasp_object.py`: 完整的 Pick-and-Place 任务实现。

#### 2.2 关键实现细节 (`grasp_object.py`)
实现了一个完整的抓取流程：
1. **环境初始化**：加载平面、Franka Panda 机械臂和待抓取的立方体。
2. **状态稳定**：等待物体物理状态稳定。
3. **分步操作**：
   - 移动到预抓取位置（物体上方）。
   - 下降到抓取位置。
   - 闭合夹爪。
   - 抬起物体。
   - 移动到目标位置（使用轨迹规划避免物体甩飞）。
   - 释放物体。
   - 复位。

#### 2.3 遇到的问题与解决方案
- **物体甩飞问题**：在机械臂移动速度过快时，物体容易因惯性滑落。
  - **解决方案**：在 `set_joint_positions` 中限制 `maxVelocity`，并实现 `move_to_position` 函数，支持插值轨迹规划（`use_trajectory=True`），将长距离移动拆分为多个路点（waypoints）。
- **IK 解不精确**：简单的 IK 调用有时无法达到精确位置。
  - **解决方案**：在 `calculate_ik_with_constraints` 中设置更严格的 `residualThreshold` 和更多的迭代次数，并明确指定关节限位 (`lowerLimits`, `upperLimits`)。

### 3. 运行方法

确保已安装依赖：
```bash
uv sync
```

运行抓取任务：
```bash
uv run python task1/grasp_object.py
```

## 任务二：
