import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import time
import multiprocessing
import os

# --- 核心计算函数 (与原版相同) ---


def is_effectively_shielded(
    missile_pos,
    target_points,
    cloud_center,
    explosion_time,
    current_time,
    smoke_radius=10,
    last_time=20,
):
    """
    判断导弹是否被有效遮蔽 (单个云团)
    """
    # 时间检查
    if current_time - explosion_time > last_time or current_time < explosion_time:
        return False

    # 位置检查
    for target_point in target_points:
        sight_vector = np.array(target_point) - np.array(missile_pos)
        sight_length = np.linalg.norm(sight_vector)
        if sight_length < 1e-6:
            return True

        mc_vector = np.array(cloud_center) - np.array(missile_pos)
        t = np.dot(mc_vector, sight_vector) / (sight_length**2)
        distance = np.linalg.norm(mc_vector - t * sight_vector)
        in_between = 0 <= t <= 1

        if distance < smoke_radius and in_between:
            # 只要有一个采样点的视线被遮挡，就继续检查下一个采样点
            pass
        else:
            # 如果有一个采样点的视线未被遮挡，则目标未被完全遮蔽
            return False

    # 所有采样点的视线都被遮挡
    return True


def calculate_missile_position(initial_pos, time, velocity=300, target_pos=(0, 0, 0)):
    """计算导弹在给定时间的位置"""
    initial_pos = np.array(initial_pos)
    target_pos = np.array(target_pos)
    direction = target_pos - initial_pos
    # 处理导弹已经到达目标的情况
    norm_direction = np.linalg.norm(direction)
    if norm_direction < 1e-6:
        return tuple(initial_pos)
    direction_unit = direction / norm_direction
    velocity_vector = direction_unit * velocity
    new_pos = initial_pos + velocity_vector * time
    return tuple(new_pos)


def calculate_plane_position(initial_pos, direction, speed, time):
    """计算飞机在给定时间的位置"""
    initial_pos = np.array(initial_pos)
    direction = np.array(direction)
    norm_direction = np.linalg.norm(direction)
    if norm_direction < 1e-6:
        return tuple(initial_pos)
    direction_unit = direction / norm_direction
    velocity_vector = direction_unit * speed
    new_pos = initial_pos + velocity_vector * time
    return tuple(new_pos)


def calculate_bomb_position(
    plane_initial_pos,
    plane_direction,
    plane_speed,
    release_time,
    fall_time,
    gravity=9.8,
):
    """计算烟幕弹轨迹和爆炸位置"""
    release_position = calculate_plane_position(
        plane_initial_pos, plane_direction, plane_speed, release_time
    )
    plane_direction = np.array(plane_direction)
    norm_direction = np.linalg.norm(plane_direction)
    if norm_direction < 1e-6:
        velocity_vector = np.zeros(3)
    else:
        direction_unit = plane_direction / norm_direction
        velocity_vector = direction_unit * plane_speed

    explosion_pos = (
        release_position[0] + velocity_vector[0] * fall_time,
        release_position[1] + velocity_vector[1] * fall_time,
        release_position[2] - 0.5 * gravity * fall_time**2,
    )
    return explosion_pos


def calculate_cloud_position(t, explosion_position, explosion_time, sink_speed=3):
    """计算特定时刻的云团位置"""
    if t < explosion_time:
        return None
    else:
        return [
            explosion_position[0],
            explosion_position[1],
            explosion_position[2] - sink_speed * (t - explosion_time),
        ]


def generate_cylinder_sample_points(
    base_center=(0, 200, 0), height=10, radius=7, num_points=100
):
    """生成圆柱体目标上的采样点"""
    base_center = np.array(base_center)
    points = []
    # 为了效率，只采样边缘和中心
    # 顶面和底面中心
    points.append(tuple(base_center))
    points.append((base_center[0], base_center[1], base_center[2] + height))
    # 顶面和底面边缘
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        x = base_center[0] + radius * np.cos(angle)
        y = base_center[1] + radius * np.sin(angle)
        points.append((x, y, base_center[2]))
        points.append((x, y, base_center[2] + height))
    return points


def calculate_shielding_duration(
    missile_initial_pos=(20000, 0, 2000),
    missile_velocity=300,
    missile_target=(0, 0, 0),
    plane_initial_pos=(17800, 0, 1800),
    plane_target=(0, 0, 1800),
    plane_speed=120,
    release_time=1.5,
    explosion_delay=3.6,
    smoke_radius=10,
    smoke_last_time=20,
    cloud_sink_speed=3,
    target_center=(0, 200, 0),
    target_height=10,
    target_radius=7,
    sample_points=50,
    time_step=0.01,
    verbose=False,  # 默认为False以减少输出
):
    """计算单个烟幕弹的有效遮蔽总时长"""
    plane_direction = np.array(plane_target) - np.array(plane_initial_pos)
    explosion_position = calculate_bomb_position(
        plane_initial_pos, plane_direction, plane_speed, release_time, explosion_delay
    )
    explosion_time = release_time + explosion_delay
    target_points = generate_cylinder_sample_points(
        target_center, target_height, target_radius, sample_points
    )

    time_end = (np.linalg.norm(np.array(missile_initial_pos)) / missile_velocity) + 5
    shielded_times = []
    is_currently_shielded = False
    shield_start_time = None

    for t in np.arange(explosion_time, time_end, time_step):
        missile_pos = calculate_missile_position(
            missile_initial_pos, t, missile_velocity, missile_target
        )
        cloud_pos = calculate_cloud_position(
            t, explosion_position, explosion_time, cloud_sink_speed
        )

        if np.linalg.norm(np.array(missile_pos)) < np.linalg.norm(
            np.array(target_center)
        ):
            if is_currently_shielded:
                shielded_times.append((shield_start_time, t))
            break

        is_shielded = is_effectively_shielded(
            missile_pos,
            target_points,
            cloud_pos,
            explosion_time,
            t,
            smoke_radius,
            smoke_last_time,
        )

        if is_shielded and not is_currently_shielded:
            is_currently_shielded = True
            shield_start_time = t
        elif not is_shielded and is_currently_shielded:
            is_currently_shielded = False
            shielded_times.append((shield_start_time, t))

    if is_currently_shielded:
        shielded_times.append((shield_start_time, t))

    return sum(end - start for start, end in shielded_times), shielded_times


# --- 适应度函数 ---

fitness_cache = {}


def position_to_key(pos):
    """将位置转换为缓存用的键"""
    return tuple(float(f"{x:.4f}") for x in pos)


def evaluate_position(position):
    """评估一个位置(解)的适应度"""
    key = position_to_key(position)
    if key in fitness_cache:
        return fitness_cache[key]

    (azimuth, speed, r1, d1, r2, d2, r3, d3) = position

    # 确保投放时间是递增的
    release_times = sorted([r1, r2, r3])
    # 找到原始延迟与排序后时间的对应关系
    delays = {r1: d1, r2: d2, r3: d3}
    explosion_delays = [delays[t] for t in release_times]

    direction_x, direction_y = np.cos(azimuth), np.sin(azimuth)
    plane_direction = (direction_x, direction_y, 0)
    plane_target = (17800 + direction_x * 20000, 0 + direction_y * 20000, 1800)

    total_shielding_periods = []
    for release_time, explosion_delay in zip(release_times, explosion_delays):
        try:
            _, shield_times = calculate_shielding_duration(
                plane_target=plane_target,
                plane_speed=speed,
                release_time=release_time,
                explosion_delay=explosion_delay,
                sample_points=20,
                time_step=0.05,
                verbose=False,
            )
            total_shielding_periods.extend(shield_times)
        except Exception:
            pass

    if not total_shielding_periods:
        fitness_cache[key] = 0.0
        return 0.0

    total_shielding_periods.sort(key=lambda x: x[0])
    merged_periods = [total_shielding_periods[0]] if total_shielding_periods else []
    for current in total_shielding_periods[1:]:
        previous = merged_periods[-1]
        if current[0] <= previous[1]:
            merged_periods[-1] = (previous[0], max(previous[1], current[1]))
        else:
            merged_periods.append(current)

    total_duration = sum(end - start for start, end in merged_periods)

    # 奖励投放间隔
    gaps = np.diff(release_times)
    gap_penalty = sum(max(0, 1 - g) for g in gaps)  # 惩罚小于1s的间隔
    total_duration -= gap_penalty * 0.5

    fitness_cache[key] = total_duration
    return total_duration


# --- 新增的可视化函数 ---


def plot_sphere(ax, center, radius, color="orange", alpha=0.3):
    """在3D坐标轴上绘制一个球体"""
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
    ax.plot_surface(x, y, z, color=color, alpha=alpha, rstride=5, cstride=5)


def plot_cylinder(ax, center, height, radius, color="b", alpha=0.5):
    """在3D坐标轴上绘制一个圆柱体"""
    x_center, y_center, z_center = center
    z = np.linspace(z_center, z_center + height, 50)
    theta = np.linspace(0, 2 * np.pi, 50)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid) + x_center
    y_grid = radius * np.sin(theta_grid) + y_center
    ax.plot_surface(x_grid, y_grid, z_grid, alpha=alpha, color=color)


def plot_3d_scenario(best_params, shield_periods):
    """可视化最优策略下的3D攻防场景"""
    print("\n正在生成3D场景可视化图...")
    (azimuth, speed, r1, d1, r2, d2, r3, d3) = best_params
    release_times = sorted([r1, r2, r3])
    delays = {r1: d1, r2: d2, r3: d3}
    explosion_delays = [delays[t] for t in release_times]

    # 场景参数
    missile_initial_pos = (20000, 0, 2000)
    plane_initial_pos = (17800, 0, 1800)
    target_center = (0, 200, 0)

    direction_x, direction_y = np.cos(azimuth), np.sin(azimuth)
    plane_direction = (direction_x, direction_y, 0)

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("3D场景", fontproperties="SimHei", fontsize=20)

    # 1. 绘制目标
    plot_cylinder(ax, target_center, height=10, radius=7, color="blue", alpha=0.6)
    ax.text(
        target_center[0],
        target_center[1],
        target_center[2] + 20,
        "真目标",
        color="blue",
        fontproperties="SimHei",
    )

    # 2. 绘制导弹和无人机轨迹
    flight_time = np.linalg.norm(np.array(missile_initial_pos)) / 300
    time_points = np.linspace(0, flight_time, 100)

    missile_traj = np.array(
        [calculate_missile_position(missile_initial_pos, t) for t in time_points]
    )
    plane_traj = np.array(
        [
            calculate_plane_position(plane_initial_pos, plane_direction, speed, t)
            for t in time_points
        ]
    )

    ax.plot(
        missile_traj[:, 0],
        missile_traj[:, 1],
        missile_traj[:, 2],
        "r--",
        label="导弹M1轨迹",
    )
    ax.plot(
        plane_traj[:, 0],
        plane_traj[:, 1],
        plane_traj[:, 2],
        "g-",
        label="无人机FY1轨迹",
    )

    # 3. 绘制烟幕弹和云团
    for i, (rt, ed) in enumerate(zip(release_times, explosion_delays)):
        release_pos = calculate_plane_position(
            plane_initial_pos, plane_direction, speed, rt
        )
        ax.scatter(
            *release_pos,
            c="purple",
            marker="x",
            s=100,
            label=f"烟幕弹{i+1}投放点" if i == 0 else "",
        )

    if shield_periods:
        start_shield_t = shield_periods[0][0]
        end_shield_t = shield_periods[-1][1]

        # 绘制遮蔽开始时的状态
        missile_pos_start = calculate_missile_position(
            missile_initial_pos, start_shield_t
        )
        ax.scatter(
            *missile_pos_start, c="red", marker="o", s=80, label="导弹 (遮蔽开始时)"
        )
        ax.plot(
            [missile_pos_start[0], target_center[0]],
            [missile_pos_start[1], target_center[1]],
            [missile_pos_start[2], target_center[2] + 5],
            "k:",
            alpha=0.5,
            label="视线 (被遮蔽)",
        )

        for rt, ed in zip(release_times, explosion_delays):
            exp_time = rt + ed
            if start_shield_t >= exp_time and start_shield_t <= exp_time + 20:
                exp_pos = calculate_bomb_position(
                    plane_initial_pos, plane_direction, speed, rt, ed
                )
                cloud_pos = calculate_cloud_position(start_shield_t, exp_pos, exp_time)
                plot_sphere(ax, cloud_pos, 10, color="gray", alpha=0.4)
        ax.text(
            cloud_pos[0],
            cloud_pos[1],
            cloud_pos[2] + 20,
            "烟幕云团",
            color="gray",
            fontproperties="SimHei",
        )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend(prop={"family": "SimHei"})
    ax.view_init(elev=20, azim=-120)
    plt.savefig("优化策略_3D场景图.png", dpi=300)
    plt.show()


def plot_pso_search_space(history, bounds, best_pos):
    """可视化PSO粒子在二维空间的搜索过程 (速度 vs 方位角)"""
    print("正在生成PSO搜索过程可视化图...")
    history = np.array(history)
    azimuth_idx, speed_idx = 0, 1

    plt.figure(figsize=(12, 8))

    # 绘制边界
    rect = plt.Rectangle(
        (np.rad2deg(bounds[azimuth_idx][0]), bounds[speed_idx][0]),
        np.rad2deg(bounds[azimuth_idx][1]) - np.rad2deg(bounds[azimuth_idx][0]),
        bounds[speed_idx][1] - bounds[speed_idx][0],
        facecolor="lightblue",
        alpha=0.3,
        label="搜索空间",
    )
    plt.gca().add_patch(rect)

    # 绘制不同阶段的粒子
    iterations_to_plot = [0, len(history) // 2, -1]
    colors = ["green", "orange", "blue"]
    labels = ["初始粒子", "中期粒子", "最终粒子"]

    for i, it in enumerate(iterations_to_plot):
        positions = history[it]
        plt.scatter(
            np.rad2deg(positions[:, azimuth_idx]),
            positions[:, speed_idx],
            c=colors[i],
            alpha=0.6,
            label=labels[i],
        )

    # 标记最优解
    plt.scatter(
        np.rad2deg(best_pos[azimuth_idx]),
        best_pos[speed_idx],
        c="red",
        marker="*",
        s=200,
        edgecolor="black",
        zorder=5,
        label="最优解",
    )

    plt.xlabel("无人机飞行方位角 (度)")
    plt.ylabel("无人机飞行速度 (m/s)")
    plt.title("PSO 搜索过程 (方位角 vs 速度)")
    plt.legend()
    plt.grid(True)
    plt.savefig("PSO_搜索过程.png", dpi=300)
    plt.show()


def plot_shielding_timeline(merged_periods, total_duration):
    """可视化遮蔽时间轴"""
    print("正在生成遮蔽时间轴可视化图...")
    fig, ax = plt.subplots(figsize=(12, 4))

    for i, (start, end) in enumerate(merged_periods):
        ax.broken_barh(
            [(start, end - start)], (10, 9), facecolors="tab:blue", alpha=0.7
        )
        ax.text(
            start + (end - start) / 2,
            14.5,
            f"{end-start:.2f}s",
            ha="center",
            va="center",
            color="black",
        )

    ax.set_ylim(5, 25)
    ax.set_xlabel("时间 (s)")
    ax.set_yticks([])
    ax.grid(True, axis="x")
    ax.set_title(f"有效遮蔽时间轴 (总时长: {total_duration:.2f}s)")
    plt.savefig("遮蔽时间轴.png", dpi=300)
    plt.show()


# --- PSO 优化主函数 ---


def optimize_with_pso_three_bombs():
    """使用PSO优化一架无人机投放三枚烟幕弹的策略"""
    start_time = time.time()

    # 定义参数边界 - 8个参数
    bounds = [
        (-np.pi / 10, np.pi / 10),  # 方位角
        (80, 140),  # 飞行速度
        (0, 0),  # 第一枚投放时间
        (0, 1),  # 第一枚延迟
        (1, 1),  # 第二枚投放时间
        (0, 1),  # 第二枚延迟
        (2, 4),  # 第三枚投放时间
        (0, 1),  # 第三枚延迟
    ]

    manager = multiprocessing.Manager()
    global fitness_cache
    fitness_cache = manager.dict()

    class Particle:
        def __init__(self, dimensions):
            self.position = np.array(
                [random.uniform(low, high) for low, high in bounds]
            )
            self.velocity = np.array(
                [
                    random.uniform(-(high - low) * 0.1, (high - low) * 0.1)
                    for low, high in bounds
                ]
            )
            self.best_position = self.position.copy()
            self.best_fitness = -float("inf")
            self.fitness = -float("inf")

        def update_velocity(self, global_best_position, w, c1, c2):
            r1, r2 = np.random.rand(len(self.position)), np.random.rand(
                len(self.position)
            )
            cognitive = c1 * r1 * (self.best_position - self.position)
            social = c2 * r2 * (global_best_position - self.position)
            self.velocity = w * self.velocity + cognitive + social

        def update_position(self):
            self.position += self.velocity
            for i in range(len(self.position)):
                low, high = bounds[i]
                if self.position[i] < low:
                    self.position[i] = low
                    self.velocity[i] *= -0.5
                elif self.position[i] > high:
                    self.position[i] = high
                    self.velocity[i] *= -0.5

    # PSO参数
    num_particles = 100
    max_iterations = 50
    w_start, w_end = 0.9, 0.4
    c1, c2 = 1.5, 1.5

    particles = [Particle(len(bounds)) for _ in range(num_particles)]
    global_best_position = None
    global_best_fitness = -float("inf")

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    history = {"best_fitness": [], "avg_fitness": []}
    particle_history = []

    try:
        print(f"使用PSO优化，粒子数量={num_particles}，最大迭代={max_iterations}")
        print(f"使用{multiprocessing.cpu_count()}个CPU核心进行并行计算...")

        for iteration in range(max_iterations):
            particle_positions = [p.position for p in particles]
            fitnesses = pool.map(evaluate_position, particle_positions)

            # 记录历史
            particle_history.append([p.copy() for p in particle_positions])

            current_best_fitness_in_iter = -float("inf")

            for i, fitness in enumerate(fitnesses):
                particles[i].fitness = fitness
                if fitness > particles[i].best_fitness:
                    particles[i].best_fitness = fitness
                    particles[i].best_position = particles[i].position.copy()
                if fitness > global_best_fitness:
                    global_best_fitness = fitness
                    global_best_position = particles[i].position.copy()
                current_best_fitness_in_iter = max(
                    current_best_fitness_in_iter, fitness
                )

            w = w_start - (w_start - w_end) * iteration / max_iterations
            for p in particles:
                p.update_velocity(global_best_position, w, c1, c2)
                p.update_position()

            avg_fitness = sum(fitnesses) / len(fitnesses)
            history["best_fitness"].append(global_best_fitness)
            history["avg_fitness"].append(avg_fitness)
            print(
                f"迭代 {iteration+1}/{max_iterations}: 最佳适应度 = {global_best_fitness:.4f}, 平均适应度 = {avg_fitness:.4f}"
            )

        print("\nPSO优化完成!")
        print(f"最佳适应度值 (总遮蔽时长): {global_best_fitness:.4f} 秒")

        # --- 结果分析与输出 ---
        end_time = time.time()
        (azimuth, speed, r1, d1, r2, d2, r3, d3) = global_best_position
        release_times = sorted([r1, r2, r3])
        delays = {r1: d1, r2: d2, r3: d3}
        explosion_delays = [delays[t] for t in release_times]

        direction_angle = np.rad2deg(azimuth)

        # 重新计算一次最终结果以获取详细数据
        total_shielding_periods = []
        bombs_data = []
        plane_direction = (np.cos(azimuth), np.sin(azimuth), 0)
        plane_target = (
            17800 + plane_direction[0] * 20000,
            0 + plane_direction[1] * 20000,
            1800,
        )

        for i, (rt, ed) in enumerate(zip(release_times, explosion_delays)):
            duration, shield_times = calculate_shielding_duration(
                plane_target=plane_target,
                plane_speed=speed,
                release_time=rt,
                explosion_delay=ed,
                sample_points=100,
                time_step=0.01,
                verbose=False,
            )
            total_shielding_periods.extend(shield_times)
            release_pos = calculate_plane_position(
                (17800, 0, 1800), plane_direction, speed, rt
            )
            explosion_pos = calculate_bomb_position(
                (17800, 0, 1800), plane_direction, speed, rt, ed
            )

            bombs_data.append(
                {
                    "无人机运动方向 (度)": direction_angle,
                    "无人机运动速度 (m/s)": speed,
                    "烟幕干扰弹编号": i + 1,
                    "投放时间 (s)": rt,
                    "爆炸延迟 (s)": ed,
                    "烟幕干扰弹投放点的x坐标 (m)": release_pos[0],
                    "烟幕干扰弹投放点的y坐标 (m)": release_pos[1],
                    "烟幕干扰弹投放点的z坐标 (m)": release_pos[2],
                    "烟幕干扰弹起爆点的x坐标 (m)": explosion_pos[0],
                    "烟幕干扰弹起爆点的y坐标 (m)": explosion_pos[1],
                    "烟幕干扰弹起爆点的z坐标 (m)": explosion_pos[2],
                }
            )

        total_shielding_periods.sort(key=lambda x: x[0])
        merged_periods = [total_shielding_periods[0]] if total_shielding_periods else []
        for current in total_shielding_periods[1:]:
            previous = merged_periods[-1]
            if current[0] <= previous[1]:
                merged_periods[-1] = (previous[0], max(previous[1], current[1]))
            else:
                merged_periods.append(current)
        total_duration = sum(end - start for start, end in merged_periods)

        # 结果表格
        df_results = pd.DataFrame(bombs_data)

        # --- 生成报告 ---
        report = []
        report.append("=" * 80)
        report.append("问题三：单架无人机多烟幕弹最优干扰策略分析报告")
        report.append("=" * 80)
        report.append(f"\n优化算法: 粒子群优化 (PSO)")
        report.append(f"优化目标: 最大化对真目标的总有效遮蔽时长")
        report.append(f"总耗时: {end_time - start_time:.2f} 秒\n")

        report.append("--- 最优策略核心参数 ---")
        report.append(f"无人机飞行方位角: {direction_angle:.2f} 度 (相对于X轴正方向)")
        report.append(f"无人机飞行速度: {speed:.2f} m/s")
        report.append(f"最终实现的总有效遮蔽时长: {total_duration:.4f} 秒\n")

        report.append("--- 烟幕弹投放详情 ---")
        report_df_string = df_results.to_string()
        report.append(report_df_string)

        report.append("\n\n--- 遮蔽时间段分析 ---")
        report.append(f"共形成 {len(merged_periods)} 个连续的遮蔽时间段：")
        for i, (start, end) in enumerate(merged_periods):
            report.append(
                f"  时间段 {i+1}: 从 {start:.2f} 秒 到 {end:.2f} 秒 (持续 {end-start:.2f} 秒)"
            )

        report.append("\n\n--- 结论 ---")
        report.append(
            "通过PSO优化，确定了无人机的最佳飞行方向、速度以及三枚烟幕弹的投放时序。"
        )
        report.append(
            "该策略旨在通过精确控制烟幕云团形成的位置和时间，使得多个云团产生的遮蔽效果能够有效衔接，"
        )
        report.append(
            f"从而形成总时长为 {total_duration:.2f} 秒的连续或近连续干扰窗口，最大化地阻碍导弹对真目标的锁定。"
        )
        report.append("详细的3D场景、PSO搜索过程和时间轴图表已保存为文件。")
        report.append("=" * 80)

        final_report_str = "\n".join(report)
        print("\n\n" + final_report_str)

        # 保存报告到文件
        with open("优化策略总结报告_问题三.txt", "w", encoding="utf-8") as f:
            f.write(final_report_str)
        print("\n优化策略总结报告已保存到 '优化策略总结报告_问题三.txt'")

        # --- 绘制所有可视化图表 ---
        # 1. PSO收敛曲线
        plt.figure(figsize=(12, 6))
        plt.plot(
            range(max_iterations),
            history["best_fitness"],
            label="最佳适应度",
            linewidth=2,
        )
        plt.plot(
            range(max_iterations),
            history["avg_fitness"],
            label="平均适应度",
            linestyle="--",
            alpha=0.7,
        )
        plt.xlabel("迭代次数")
        plt.ylabel("适应度 (总遮蔽时长 s)")
        plt.title("PSO 优化收敛过程 ")
        plt.legend()
        plt.grid(True)
        plt.savefig("PSO优化收敛曲线.png", dpi=300)
        plt.show()

        # 2. PSO搜索过程
        plot_pso_search_space(particle_history, bounds, global_best_position)

        # 3. 遮蔽时间轴
        plot_shielding_timeline(merged_periods, total_duration)

        # 4. 3D场景
        plot_3d_scenario(global_best_position, merged_periods)

        return global_best_position, df_results, total_duration

    finally:
        pool.close()
        pool.join()


if __name__ == "__main__":
    # 设置中文字体，以防可视化图表中的中文乱码
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 指定默认字体
    plt.rcParams["axes.unicode_minus"] = False  # 解决保存图像是负号'-'显示为方块的问题

    # 设置随机种子以保证结果可复现
    random.seed(42)
    np.random.seed(42)

    # 如果输出文件目录不存在，则创建
    if not os.path.exists("output"):
        os.makedirs("output")

    # 运行主程序
    optimize_with_pso_three_bombs()
