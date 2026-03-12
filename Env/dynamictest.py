import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from QuadrotorDynamics import QuadrotorDynamics


def test_aerodynamics():
    # 初始化无人机模型
    quad = QuadrotorDynamics()
    quad.dt = 0.02  # 确保时间步长一致

    # 测试时长和数据存储
    total_time = 100.0  # 10秒测试
    steps = int(total_time / quad.dt)
    time_points = np.linspace(0, total_time, steps)

    # 数据存储数组
    positions = np.zeros((steps, 3))
    velocities = np.zeros((steps, 3))
    orientations = np.zeros((steps, 3))
    angular_velocities = np.zeros((steps, 3))
    linear_accs = np.zeros((steps, 3))
    angular_accs = np.zeros((steps, 3))
    motor_speeds_log = np.zeros((steps, 4))

    # 生成电机转速序列（测试不同工况）
    def generate_motor_speeds(t):
        # 0-2秒：悬停测试
        if t < 20:
            base_speed = (quad.min_motor_speed + quad.max_motor_speed) / 2
            return np.array([base_speed, base_speed, base_speed, base_speed])

        # 2-4秒：滚转测试（左右电机转速差）
        elif t < 40:
            base = 600
            diff = 200 * np.sin((t - 2) * 2)
            return np.array([base + diff, base + diff, base - diff, base - diff])

        # 4-6秒：俯仰测试（前后电机转速差）
        elif t < 60:
            base = 600
            diff = 200 * np.sin((t - 4) * 2)
            return np.array([base - diff, base + diff, base + diff, base - diff])

        # 6-8秒：偏航测试（正反桨转速差）
        elif t < 80:
            base = 600
            diff = 150 * np.sin((t - 6) * 1.5)
            return np.array([base - diff, base + diff, base - diff, base + diff])

        # 8-10秒：总推力变化（上升下降）
        else:
            base = 600 + 200 * np.sin((t - 8) * 1.5)
            return np.array([base, base, base, base])

    # 运行仿真
    for i in range(steps):
        t = time_points[i]
        motor_speeds = generate_motor_speeds(t)

        # 记录电机转速
        motor_speeds_log[i] = motor_speeds

        # 计算角加速度（用于记录）
        _, ang_acc = quad.compute_angle_acc(
            motor_speeds,
            quad.orientation,
            quad.angular_velocity
        )
        angular_accs[i] = ang_acc

        # 计算线加速度（用于记录）
        rot = quad.rot
        thrust = quad.C_T * np.sum(np.square(motor_speeds))
        lin_acc = quad.compute_linear_acc(thrust, rot, quad.velocity)
        linear_accs[i] = lin_acc

        # 更新无人机状态
        quad.update(motor_speeds, dt=quad.dt)

        # 记录状态
        positions[i] = quad.position
        velocities[i] = quad.velocity
        orientations[i] = quad.orientation
        angular_velocities[i] = quad.angular_velocity

    # 绘制结果
    fig, axs = plt.subplots(6, 1, figsize=(12, 20), sharex=True)
    fig.suptitle('Quadrotor Aerodynamics Test', fontsize=16)

    # 1. 电机转速
    axs[0].plot(time_points, motor_speeds_log)
    axs[0].set_title('Motor Speeds (rad/s)')
    axs[0].legend(['Motor 0', 'Motor 1', 'Motor 2', 'Motor 3'])
    axs[0].grid(True)

    # 2. 位置
    axs[1].plot(time_points, positions)
    axs[1].set_title('Position (m)')
    axs[1].legend(['X', 'Y', 'Z'])
    axs[1].grid(True)

    # 3. 速度
    axs[2].plot(time_points, velocities)
    axs[2].set_title('Velocity (m/s)')
    axs[2].legend(['Vx', 'Vy', 'Vz'])
    axs[2].grid(True)

    # 4. 线加速度
    axs[3].plot(time_points, linear_accs)
    axs[3].set_title('Linear Acceleration (m/s²)')
    axs[3].legend(['Ax', 'Ay', 'Az'])
    axs[3].grid(True)

    # 5. 姿态角 (转换为度)
    axs[4].plot(time_points, orientations)
    axs[4].set_title('Orientation (deg)')
    axs[4].legend(['Roll', 'Pitch', 'Yaw'])
    axs[4].grid(True)

    # 6. 角速度 (转换为度/秒)
    axs[5].plot(time_points,angular_velocities)
    axs[5].set_title('Angular Velocity (deg/s)')
    axs[5].legend(['ωx', 'ωy', 'ωz'])
    axs[5].set_xlabel('Time (s)')
    axs[5].grid(True)

    plt.tight_layout()

    # 额外绘制角加速度图
    fig2, ax = plt.subplots(figsize=(12, 5))
    ax.plot(time_points, angular_accs)
    ax.set_title('Angular Acceleration (deg/s²)')
    ax.legend(['αx', 'αy', 'αz'])
    ax.set_xlabel('Time (s)')
    ax.grid(True)

    plt.show()

    # 3D位置轨迹图
    fig3 = plt.figure(figsize=(10, 10))
    ax3d = fig3.add_subplot(111, projection='3d')
    ax3d.plot(positions[:, 0], positions[:, 1], positions[:, 2])
    ax3d.set_title('3D Position Trajectory')
    ax3d.set_xlabel('X (m)')
    ax3d.set_ylabel('Y (m)')
    ax3d.set_zlabel('Z (m)')
    ax3d.grid(True)

    plt.show()


if __name__ == "__main__":
    test_aerodynamics()