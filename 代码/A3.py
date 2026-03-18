import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import time
import multiprocessing


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
    判断导弹是否被有效遮蔽

    参数:
    missile_pos -- 导弹当前位置坐标 (x,y,z)
    target_points -- 目标上的采样点列表
    cloud_center -- 烟幕云团中心位置 (x,y,z)
    explosion_time -- 烟幕弹爆炸时间
    current_time -- 当前时间
    smoke_radius -- 烟幕云团半径,10m
    last_time -- 烟幕有效持续时间,20s

    返回:
    bool -- 是否被有效遮蔽
    """
    # 时间检查
    if current_time - explosion_time > last_time:
        return False

    # 位置检查
    for target_point in target_points:
        # 导弹到目标点的视线向量
        sight_vector = np.array(target_point) - np.array(missile_pos)
        sight_length = np.linalg.norm(sight_vector)
        if sight_length < 1e-6:  # 避免除零
            return True

        # 云团中心到视线的投影比例
        mc_vector = np.array(cloud_center) - np.array(missile_pos)
        t = np.dot(mc_vector, sight_vector) / (sight_length**2)

        # 云团中心到视线的垂直距离
        distance = np.linalg.norm(mc_vector - t * sight_vector)

        # 云团是否在导弹和目标之间
        in_between = 0 <= t <= 1

        # 如果视线未被遮蔽或云团不在中间，则整体未被遮蔽
        if distance >= smoke_radius:
            # 如果云团离视线太远，肯定不遮蔽
            return False
        elif not in_between:
            # 如果云团不在导弹和目标之间，检查是否足够接近目标
            if t > 1:  # 云团在目标"后方"
                distance_to_target = np.linalg.norm(
                    np.array(cloud_center) - np.array(target_point)
                )
                return distance_to_target < smoke_radius
            else:  # 云团在导弹"后方"(t<0)
                distance_to_missile = np.linalg.norm(
                    np.array(cloud_center) - np.array(missile_pos)
                )
                return distance_to_missile < smoke_radius
    return True


def is_effectively_shielded_by_multiple_clouds(
    missile_pos,
    target_points,
    cloud_centers,
    explosion_times,
    current_time,
    smoke_radius=10,
    last_time=20,
):
    """
    判断导弹是否被多个烟幕云团有效遮蔽

    参数:
    missile_pos -- 导弹当前位置坐标 (x,y,z)
    target_points -- 目标上的采样点列表
    cloud_centers -- 多个烟幕云团中心位置列表，每个元素为 (x,y,z) 或 None
    explosion_times -- 多个烟幕弹爆炸时间列表
    current_time -- 当前时间
    smoke_radius -- 烟幕云团半径,10m
    last_time -- 烟幕有效持续时间,20s

    返回:
    bool -- 是否被有效遮蔽
    """
    # 筛选有效的云团(已爆炸且未超过持续时间)
    active_clouds = []
    for cloud_center, explosion_time in zip(cloud_centers, explosion_times):
        if cloud_center is not None and current_time - explosion_time <= last_time:
            active_clouds.append(cloud_center)

    # 如果没有有效云团，直接返回False
    if not active_clouds:
        return False

    # 检查每个目标点
    for target_point in target_points:
        # 导弹到目标点的视线向量
        sight_vector = np.array(target_point) - np.array(missile_pos)
        sight_length = np.linalg.norm(sight_vector)

        # 如果导弹已经到达目标点，视为被遮蔽
        if sight_length < 1e-6:
            return True

        # 检查是否至少有一个云团遮蔽了视线
        line_shielded = False

        for cloud_center in active_clouds:
            # 云团中心到视线的投影比例
            mc_vector = np.array(cloud_center) - np.array(missile_pos)
            t = np.dot(mc_vector, sight_vector) / (sight_length**2)

            # 云团中心到视线的垂直距离
            distance = np.linalg.norm(mc_vector - t * sight_vector)

            # 云团是否在导弹和目标之间
            in_between = 0 <= t <= 1

            # 判断该云团是否遮蔽了视线
            if distance < smoke_radius and in_between:
                line_shielded = True
                break
            elif distance < smoke_radius:
                # 如果云团不在导弹和目标之间，检查是否足够接近目标或导弹
                if t > 1:  # 云团在目标"后方"
                    distance_to_target = np.linalg.norm(
                        np.array(cloud_center) - np.array(target_point)
                    )
                    if distance_to_target < smoke_radius:
                        line_shielded = True
                        break
                else:  # 云团在导弹"后方"(t<0)
                    distance_to_missile = np.linalg.norm(
                        np.array(cloud_center) - np.array(missile_pos)
                    )
                    if distance_to_missile < smoke_radius:
                        line_shielded = True
                        break

        # 如果所有云团都未能遮蔽该视线，则整体未被遮蔽
        if not line_shielded:
            return False

    # 所有视线都被遮蔽，则整体被遮蔽
    return True


def calculate_missile_position(initial_pos, time, velocity=300, target_pos=(0, 0, 0)):
    """
    计算导弹在给定时间的位置

    参数:
    initial_pos -- 导弹初始位置 (x,y,z)
    time -- 时间(秒)
    velocity -- 导弹速度(米/秒),默认300
    target_pos -- 导弹目标位置(假目标)，默认(0,0,0)

    返回:
    tuple -- 导弹当前位置 (x,y,z)
    """
    initial_pos = np.array(initial_pos)
    target_pos = np.array(target_pos)

    direction = target_pos - initial_pos
    direction_unit = direction / np.linalg.norm(direction)
    velocity_vector = direction_unit * velocity

    new_pos = initial_pos + velocity_vector * time
    return tuple(new_pos)


def calculate_plane_position(initial_pos, direction, speed, time):
    """
    计算飞机在给定时间的位置

    参数:
    initial_pos -- 飞机初始位置 (x,y,z)
    direction -- 飞行方向向量
    speed -- 飞行速度(米/秒)
    time -- 时间(秒)

    返回:
    tuple -- 飞机当前位置 (x,y,z)
    """
    initial_pos = np.array(initial_pos)
    direction = np.array(direction)

    direction_unit = direction / np.linalg.norm(direction)
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
    """
    计算烟幕弹轨迹和爆炸位置

    参数:
    plane_initial_pos -- 无人机初始位置 (x,y,z)
    plane_direction -- 无人机飞行方向
    plane_speed -- 无人机速度(米/秒)
    release_time -- 投放时间(秒)
    fall_time -- 烟幕弹下落时间(秒)
    gravity -- 重力加速度,默认9.8

    返回:
    tuple -- 爆炸位置 (x,y,z)
    """
    # 计算投放位置
    release_position = calculate_plane_position(
        plane_initial_pos, plane_direction, plane_speed, release_time
    )

    # 计算水平位移
    plane_direction = np.array(plane_direction)
    direction_unit = plane_direction / np.linalg.norm(plane_direction)
    velocity_vector = direction_unit * plane_speed

    # 计算爆炸位置
    explosion_pos = (
        release_position[0] + velocity_vector[0] * fall_time,
        release_position[1] + velocity_vector[1] * fall_time,
        release_position[2] - 0.5 * gravity * fall_time**2,
    )

    return explosion_pos


def calculate_cloud_position(t, explosion_position, explosion_time, sink_speed=3):
    """
    计算特定时刻的云团位置

    参数:
    t -- 当前时间(秒)
    explosion_position -- 爆炸位置 (x,y,z)
    explosion_time -- 爆炸时间(秒)
    sink_speed -- 云团下沉速度(米/秒),默认3

    返回:
    list/None -- 云团位置 [x,y,z] 或 None(未爆炸)
    """
    # 云团匀速下沉
    if t < explosion_time:
        return None  # 未起爆
    else:
        return [
            explosion_position[0],
            explosion_position[1],
            explosion_position[2] - sink_speed * (t - explosion_time),
        ]


def generate_cylinder_sample_points(
    base_center=(0, 200, 0), height=10, radius=7, num_points=5000
):
    """
    生成圆柱体目标上的采样点

    参数:
    base_center -- 底面圆心坐标，默认(0,200,0)
    height -- 圆柱体高度，默认10
    radius -- 圆柱体半径，默认7
    num_points -- 每个圆面上的采样点数，默认100

    返回:
    list -- 圆柱体表面采样点列表
    """
    base_center = np.array(base_center)
    top_center = np.array([base_center[0], base_center[1], base_center[2] + height])

    points = []

    # 底面采样点
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        x = base_center[0] + radius * np.cos(angle)
        y = base_center[1] + radius * np.sin(angle)
        z = base_center[2]
        points.append((x, y, z))

    # 顶面采样点
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        x = top_center[0] + radius * np.cos(angle)
        y = top_center[1] + radius * np.sin(angle)
        z = top_center[2]
        points.append((x, y, z))

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
    sample_points=5000,
    time_step=0.005,
    verbose=True,
):
    """
    计算导弹被烟幕有效遮蔽的总时长

    参数:
    missile_initial_pos -- 导弹初始位置，默认(20000,0,2000)
    missile_velocity -- 导弹速度(米/秒)，默认300
    missile_target -- 导弹目标位置(假目标)，默认(0,0,0)
    plane_initial_pos -- 无人机初始位置，默认(17800,0,1800)
    plane_target -- 无人机目标位置，默认(0,0,0)
    plane_speed -- 无人机速度(米/秒)，默认120
    release_time -- 烟幕弹投放时间(秒)，默认1.5
    explosion_delay -- 烟幕弹起爆延迟(秒)，默认3.6
    smoke_radius -- 烟幕云团半径(米)，默认10
    smoke_last_time -- 烟幕有效持续时间(秒)，默认20
    cloud_sink_speed -- 云团下沉速度(米/秒)，默认3
    target_center -- 真目标底面圆心坐标，默认(0,200,0)
    target_height -- 真目标高度(米)，默认10
    target_radius -- 真目标半径(米)，默认7
    sample_points -- 圆柱体每个面的采样点数，默认100
    time_step -- 时间步长(秒)，默认0.01
    verbose -- 是否输出详细信息，默认True

    返回:
    tuple -- (总遮蔽时长(秒), 遮蔽时间段列表[(开始时间,结束时间),...]
    """
    # 计算无人机飞行方向
    plane_direction = np.array(plane_target) - np.array(plane_initial_pos)

    # 计算爆炸位置和时间
    explosion_position = calculate_bomb_position(
        plane_initial_pos, plane_direction, plane_speed, release_time, explosion_delay
    )

    explosion_time = release_time + explosion_delay

    # 生成目标采样点
    target_points = generate_cylinder_sample_points(
        target_center, target_height, target_radius, sample_points
    )

    if verbose:
        print("=== 烟幕干扰分析 ===")
        print(f"无人机初始位置: {plane_initial_pos}")
        print(f"飞行速度: {plane_speed} m/s，朝向目标: {plane_target}")
        print(f"投放时间: {release_time}s")
        print(f"起爆延时: {explosion_delay}s")
        print(f"爆炸时间: {explosion_time}s")
        print(
            f"爆炸位置: ({explosion_position[0]:.1f}, {explosion_position[1]:.1f}, {explosion_position[2]:.1f})"
        )
        print(f"烟幕半径: {smoke_radius}m")
        print(f"烟幕有效时间: {smoke_last_time}s")
        print(f"云团下沉速度: {cloud_sink_speed} m/s")
        print(f"\n目标信息:")
        print(
            f"真目标位置: 圆心{target_center}，半径{target_radius}m，高{target_height}m"
        )
        print(f"采样点数: {len(target_points)}")
        print(f"\n开始模拟...")

    # 遍历时间步长
    time_start = 0
    time_end = 100  # 足够长的时间范围

    shielded_times = []
    is_currently_shielded = False
    shield_start_time = None

    # 记录关键时刻信息
    first_shield_info = None
    last_shield_info = None

    for t in np.arange(time_start, time_end, time_step):
        # 计算当前位置
        missile_pos = calculate_missile_position(
            missile_initial_pos, t, missile_velocity, missile_target
        )
        cloud_pos = calculate_cloud_position(
            t, explosion_position, explosion_time, cloud_sink_speed
        )

        # 检查导弹是否已经通过目标区域
        missile_pos_array = np.array(missile_pos)
        if np.linalg.norm(missile_pos_array) < 1:
            if is_currently_shielded:
                shielded_times.append((shield_start_time, t))
                last_shield_info = {
                    "time": t,
                    "missile_pos": missile_pos,
                    "cloud_pos": cloud_pos,
                    "missile_distance": np.linalg.norm(
                        missile_pos_array - np.array(target_center)
                    ),
                }
            break

        # 检查云团是否已形成
        if cloud_pos is None:
            continue

        # 判断是否有效遮蔽
        is_shielded = is_effectively_shielded(
            missile_pos,
            target_points,
            cloud_pos,
            explosion_time,
            t,
            smoke_radius,
            smoke_last_time,
        )

        # 记录遮蔽时间段
        if is_shielded and not is_currently_shielded:
            is_currently_shielded = True
            shield_start_time = t
            if first_shield_info is None:
                first_shield_info = {
                    "time": t,
                    "missile_pos": missile_pos,
                    "cloud_pos": cloud_pos,
                    "missile_distance": np.linalg.norm(
                        np.array(missile_pos) - np.array(target_center)
                    ),
                }
        elif not is_shielded and is_currently_shielded:
            is_currently_shielded = False
            shielded_times.append((shield_start_time, t))
            last_shield_info = {
                "time": t,
                "missile_pos": missile_pos,
                "cloud_pos": cloud_pos,
                "missile_distance": np.linalg.norm(
                    np.array(missile_pos) - np.array(target_center)
                ),
            }

    # 输出详细信息
    if verbose:
        print(f"\n=== 遮蔽时间段详情 ===")
        if len(shielded_times) == 0:
            print("未检测到有效遮蔽！")
            # 调试信息
            t_test = 10.0  # 测试时刻
            missile_test = calculate_missile_position(
                missile_initial_pos, t_test, missile_velocity, missile_target
            )
            cloud_test = calculate_cloud_position(
                t_test, explosion_position, explosion_time, cloud_sink_speed
            )
            if cloud_test:
                print(f"\n调试信息:")
                print(f"t={t_test}s时:")
                print(f"  导弹位置: {missile_test}")
                print(f"  云团位置: {cloud_test}")
                print(
                    f"  导弹到真目标距离: {np.linalg.norm(np.array(missile_test) - np.array(target_center)):.1f}m"
                )
        else:
            for i, (start, end) in enumerate(shielded_times):
                print(f"第{i+1}段: {start:.2f}s - {end:.2f}s (持续 {end-start:.2f}s)")

            if first_shield_info:
                print(f"\n首次遮蔽时刻 (t={first_shield_info['time']:.2f}s):")
                print(f"  导弹位置: {first_shield_info['missile_pos']}")
                print(f"  云团位置: {first_shield_info['cloud_pos']}")
                print(f"  导弹距真目标: {first_shield_info['missile_distance']:.1f}m")

            if last_shield_info:
                print(f"\n最后遮蔽时刻 (t={last_shield_info['time']:.2f}s):")
                print(f"  导弹位置: {last_shield_info['missile_pos']}")
                print(f"  云团位置: {last_shield_info['cloud_pos']}")
                print(f"  导弹距真目标: {last_shield_info['missile_distance']:.1f}m")

        # 计算总遮蔽时长
        total_duration = sum(end - start for start, end in shielded_times)
        print(f"\n=== 结果汇总 ===")
        print(f"遮蔽段数: {len(shielded_times)}")
        print(f"总遮蔽时长: {total_duration:.4f} 秒")

    return sum(end - start for start, end in shielded_times), shielded_times


fitness_cache = {}


def position_to_key(pos):
    """将位置转换为缓存用的键"""
    return tuple(float(f"{x:.4f}") for x in pos)


def evaluate_position(position):
    """评估一个位置(解)的适应度"""
    key = position_to_key(position)
    if key in fitness_cache:
        return fitness_cache[key]

    # 解析参数
    (
        azimuth,
        speed,
        release_time1,
        explosion_delay1,
        release_time2,
        explosion_delay2,
        release_time3,
        explosion_delay3,
    ) = position

    # 计算飞行方向向量
    direction_x = np.cos(azimuth)
    direction_y = np.sin(azimuth)
    direction_z = 0  # 水平飞行

    plane_direction = (direction_x, direction_y, direction_z)

    # 计算目标位置（保持z坐标不变）
    dir_norm = np.sqrt(direction_x**2 + direction_y**2)
    if dir_norm < 1e-6:  # 避免除零
        return 0.0

    target_distance = 20000  # 足够远的距离
    plane_target = (
        17800 + direction_x / dir_norm * target_distance,
        0 + direction_y / dir_norm * target_distance,
        1800,  # 保持z坐标不变
    )

    # 计算三枚烟幕弹的遮蔽区间
    total_shielding_periods = []

    # 三枚烟幕弹的计算
    for release_time, explosion_delay in [
        (release_time1, explosion_delay1),
        (release_time2, explosion_delay2),
        (release_time3, explosion_delay3),
    ]:
        try:
            _, shield_times = calculate_shielding_duration(
                missile_initial_pos=(20000, 0, 2000),
                missile_velocity=300,
                missile_target=(0, 0, 0),
                plane_initial_pos=(17800, 0, 1800),
                plane_target=plane_target,
                plane_speed=speed,
                release_time=release_time,
                explosion_delay=explosion_delay,
                sample_points=100,  # 降低精度加快评估
                time_step=0.02,
                verbose=False,
            )
            total_shielding_periods.extend(shield_times)
        except Exception:
            pass

    # 如果没有有效遮蔽，返回0
    if not total_shielding_periods:
        result = 0.0
        fitness_cache[key] = result
        return result

    # 合并重叠的遮蔽时间段
    total_shielding_periods.sort(key=lambda x: x[0])
    merged_periods = [total_shielding_periods[0]]

    for current in total_shielding_periods[1:]:
        previous = merged_periods[-1]
        if current[0] <= previous[1]:
            # 有重叠，合并区间
            merged_periods[-1] = (previous[0], max(previous[1], current[1]))
        else:
            # 无重叠，添加新区间
            merged_periods.append(current)

    # 计算总遮蔽时长
    total_duration = sum(end - start for start, end in merged_periods)

    # 添加奖励 - 奖励更均匀分布的投放时间
    time_gaps = [release_time2 - release_time1, release_time3 - release_time2]
    if min(time_gaps) >= 1.0 and max(time_gaps) <= 5.0:
        # 投放间隔在1-5秒之间，奖励均衡的投放策略
        total_duration *= 1.1

    fitness_cache[key] = total_duration
    return total_duration


def optimize_with_pso_three_bombs():
    """使用PSO优化一架无人机投放三枚烟幕弹的策略"""
    start_time = time.time()

    # 定义FY1和M1的初始位置
    FY1_initial_pos = (17800, 0, 1800)
    M1_initial_pos = (20000, 0, 2000)

    # 定义参数边界 - 8个参数
    bounds = [
        (-np.pi / 10, np.pi / 10),  # 方位角
        (80, 140),  # 飞行速度
        (0, 4),  # 第一枚投放时间
        (0, 1),  # 第一枚延迟
        (0, 4),  # 第二枚投放时间
        (0, 1),  # 第二枚延迟
        (0, 4),  # 第三枚投放时间
        (0, 1),  # 第三枚延迟
    ]

    # 创建缓存
    manager = multiprocessing.Manager()
    fitness_cache = manager.dict()

    # 创建粒子类
    class Particle:
        def __init__(self, dimensions):
            # 初始化位置和速度
            self.position = np.array(
                [random.uniform(low, high) for low, high in bounds]
            )
            self.velocity = np.array(
                [
                    random.uniform(-0.5 * (high - low), 0.5 * (high - low))
                    for low, high in bounds
                ]
            )

            # 调整投放时间顺序
            self._adjust_release_times()

            # 初始化粒子的最佳位置和适应度
            self.best_position = self.position.copy()
            self.best_fitness = -float("inf")
            self.fitness = -float("inf")

        def _adjust_release_times(self):
            """确保投放时间顺序: t1 < t2 < t3"""
            # 确保第二枚比第一枚晚至少1秒
            if self.position[4] <= self.position[2] + 1:
                self.position[4] = self.position[2] + 1

            # 确保第三枚比第二枚晚至少1秒
            if self.position[6] <= self.position[4] + 1:
                self.position[6] = self.position[4] + 1

        def update_velocity(self, global_best_position, w=0.7, c1=1.5, c2=1.5):
            """更新粒子的速度"""
            r1 = np.random.rand(len(self.position))
            r2 = np.random.rand(len(self.position))

            cognitive_component = c1 * r1 * (self.best_position - self.position)
            social_component = c2 * r2 * (global_best_position - self.position)

            self.velocity = w * self.velocity + cognitive_component + social_component

        def update_position(self):
            """更新粒子的位置并确保在边界内"""
            self.position = self.position + self.velocity

            # 边界处理
            for i in range(len(self.position)):
                if self.position[i] < bounds[i][0]:
                    self.position[i] = bounds[i][0]
                    self.velocity[i] *= -0.5  # 反弹
                elif self.position[i] > bounds[i][1]:
                    self.position[i] = bounds[i][1]
                    self.velocity[i] *= -0.5  # 反弹

            # 调整投放时间顺序
            self._adjust_release_times()

    # PSO参数
    num_particles = 300
    max_iterations = 50

    # 自适应惯性权重
    w_start = 0.9  # 初始较大的惯性权重促进全局搜索
    w_end = 0.4  # 最终较小的惯性权重促进局部精细搜索

    # 创建粒子群
    particles = [Particle(len(bounds)) for _ in range(num_particles)]

    # 初始化全局最佳
    global_best_position = None
    global_best_fitness = -float("inf")

    # 设置并行池
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    # 创建历史记录
    history = {"best_fitness": [], "avg_fitness": []}

    try:
        print(f"使用PSO优化，粒子数量={num_particles}，最大迭代={max_iterations}")
        print(f"使用{multiprocessing.cpu_count()}个CPU核心进行并行计算")

        # 初始评估
        fitnesses = pool.map(evaluate_position, [p.position for p in particles])

        for i, fitness in enumerate(fitnesses):
            particles[i].fitness = fitness
            particles[i].best_fitness = fitness
            particles[i].best_position = particles[i].position.copy()

            if fitness > global_best_fitness:
                global_best_fitness = fitness
                global_best_position = particles[i].position.copy()

        # PSO主循环
        stagnation_counter = 0
        prev_best = global_best_fitness

        for iteration in range(max_iterations):
            # 计算当前的惯性权重 (线性递减)
            w = w_start - (w_start - w_end) * iteration / max_iterations

            # 更新所有粒子
            for p in particles:
                p.update_velocity(global_best_position, w=w)
                p.update_position()

            # 并行评估
            fitnesses = pool.map(evaluate_position, [p.position for p in particles])

            for i, fitness in enumerate(fitnesses):
                particles[i].fitness = fitness

                # 更新个体最佳
                if fitness > particles[i].best_fitness:
                    particles[i].best_fitness = fitness
                    particles[i].best_position = particles[i].position.copy()

                # 更新全局最佳
                if fitness > global_best_fitness:
                    global_best_fitness = fitness
                    global_best_position = particles[i].position.copy()
                    stagnation_counter = 0

            # 检测停滞并应用扰动
            if abs(global_best_fitness - prev_best) < 0.001:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
                prev_best = global_best_fitness

            if stagnation_counter >= 10:
                print(f"迭代 {iteration}: 检测到停滞，应用粒子重置")

                # 重置40%的粒子
                reset_count = int(0.4 * num_particles)
                indices = np.argsort([p.fitness for p in particles])[:reset_count]

                for idx in indices:
                    if random.random() < 0.7:
                        # 随机重置
                        particles[idx] = Particle(len(bounds))
                    else:
                        # 基于全局最佳的变异重置
                        new_position = global_best_position + np.random.normal(
                            0, 0.2, size=len(bounds)
                        ) * np.array([(high - low) for low, high in bounds])
                        # 确保新位置在边界内
                        for j in range(len(new_position)):
                            new_position[j] = min(
                                max(new_position[j], bounds[j][0]), bounds[j][1]
                            )

                        # 创建新粒子并替换
                        new_particle = Particle(len(bounds))
                        new_particle.position = new_position
                        particles[idx] = new_particle

                stagnation_counter = 0

            # 收集历史数据
            current_avg = sum(p.fitness for p in particles) / len(particles)
            history["best_fitness"].append(global_best_fitness)
            history["avg_fitness"].append(current_avg)

            # 打印进度
            if iteration % 5 == 0 or iteration == max_iterations - 1:
                print(
                    f"迭代 {iteration}: 最佳适应度 = {global_best_fitness:.4f}, 平均适应度 = {current_avg:.4f}, 缓存数 = {len(fitness_cache)}"
                )

        print("\nPSO优化完成!")
        print(f"最佳适应度值: {global_best_fitness:.4f}")

        # 解析最佳位置参数
        (
            azimuth,
            speed,
            release_time1,
            explosion_delay1,
            release_time2,
            explosion_delay2,
            release_time3,
            explosion_delay3,
        ) = global_best_position

        # 计算飞行方向
        direction_x = np.cos(azimuth)
        direction_y = np.sin(azimuth)
        direction_z = 0
        direction = (direction_x, direction_y, direction_z)

        # 计算目标位置
        dir_norm = np.sqrt(direction_x**2 + direction_y**2)
        target_distance = 20000
        plane_target = (
            FY1_initial_pos[0] + direction_x / dir_norm * target_distance,
            FY1_initial_pos[1] + direction_y / dir_norm * target_distance,
            FY1_initial_pos[2],
        )

        # 转换为角度（0-360度）
        direction_angle = (np.arctan2(direction_y, direction_x) * 180 / np.pi) % 360

        # 计算三枚烟幕弹的详细参数
        bombs_data = []
        for idx, (release_time, explosion_delay) in enumerate(
            [
                (release_time1, explosion_delay1),
                (release_time2, explosion_delay2),
                (release_time3, explosion_delay3),
            ]
        ):
            # 计算投放点和起爆点
            release_position = calculate_plane_position(
                FY1_initial_pos, direction, speed, release_time
            )

            explosion_position = calculate_bomb_position(
                FY1_initial_pos, direction, speed, release_time, explosion_delay
            )

            # 计算单枚干扰时长
            duration, _ = calculate_shielding_duration(
                missile_initial_pos=M1_initial_pos,
                missile_velocity=300,
                missile_target=(0, 0, 0),
                plane_initial_pos=FY1_initial_pos,
                plane_target=plane_target,
                plane_speed=speed,
                release_time=release_time,
                explosion_delay=explosion_delay,
                sample_points=300,
                verbose=False,
            )

            bombs_data.append(
                {
                    "编号": idx + 1,
                    "运动方向": direction_angle,
                    "运动速度": speed,
                    "投放点x": release_position[0],
                    "投放点y": release_position[1],
                    "投放点z": release_position[2],
                    "起爆点x": explosion_position[0],
                    "起爆点y": explosion_position[1],
                    "起爆点z": explosion_position[2],
                    "有效干扰时长": duration,
                }
            )

        # 计算合并后的总有效干扰时长
        total_shielding_periods = []
        for release_time, explosion_delay in [
            (release_time1, explosion_delay1),
            (release_time2, explosion_delay2),
            (release_time3, explosion_delay3),
        ]:
            _, shield_times = calculate_shielding_duration(
                missile_initial_pos=M1_initial_pos,
                missile_velocity=300,
                missile_target=(0, 0, 0),
                plane_initial_pos=FY1_initial_pos,
                plane_target=plane_target,
                plane_speed=speed,
                release_time=release_time,
                explosion_delay=explosion_delay,
                sample_points=300,
                verbose=False,
            )
            total_shielding_periods.extend(shield_times)

        # 合并重叠的遮蔽时间段
        if total_shielding_periods:
            total_shielding_periods.sort(key=lambda x: x[0])
            merged_periods = [total_shielding_periods[0]]

            for current in total_shielding_periods[1:]:
                previous = merged_periods[-1]
                if current[0] <= previous[1]:
                    # 有重叠，合并区间
                    merged_periods[-1] = (previous[0], max(previous[1], current[1]))
                else:
                    # 无重叠，添加新区间
                    merged_periods.append(current)

            total_duration = sum(end - start for start, end in merged_periods)
        else:
            total_duration = 0.0
            merged_periods = []

        # 创建结果DataFrame
        df = pd.DataFrame(
            columns=[
                "无人机运动方向",
                "无人机运动速度 (m/s)",
                "烟幕干扰弹编号",
                "烟幕干扰弹投放点的x坐标 (m)",
                "烟幕干扰弹投放点的y坐标 (m)",
                "烟幕干扰弹投放点的z坐标 (m)",
                "烟幕干扰弹起爆点的x坐标 (m)",
                "烟幕干扰弹起爆点的y坐标 (m)",
                "烟幕干扰弹起爆点的z坐标 (m)",
                "有效干扰时长 (s)",
            ]
        )

        # 填充数据
        for i, bomb in enumerate(bombs_data):
            df.loc[i] = [
                bomb["运动方向"],
                bomb["运动速度"],
                bomb["编号"],
                bomb["投放点x"],
                bomb["投放点y"],
                bomb["投放点z"],
                bomb["起爆点x"],
                bomb["起爆点y"],
                bomb["起爆点z"],
                bomb["有效干扰时长"],
            ]

        # 打印优化结果
        print("\n=== 优化结果 ===")
        print(f"无人机运动方向: {direction_angle:.2f}度")
        print(f"无人机运动速度: {speed:.2f} m/s")
        print(f"总有效干扰时长: {total_duration:.2f}秒")
        print(
            f"烟幕弹1: 投放时间={release_time1:.2f}s, 延迟={explosion_delay1:.2f}s, 干扰时长={bombs_data[0]['有效干扰时长']:.2f}s"
        )
        print(
            f"烟幕弹2: 投放时间={release_time2:.2f}s, 延迟={explosion_delay2:.2f}s, 干扰时长={bombs_data[1]['有效干扰时长']:.2f}s"
        )
        print(
            f"烟幕弹3: 投放时间={release_time3:.2f}s, 延迟={explosion_delay3:.2f}s, 干扰时长={bombs_data[2]['有效干扰时长']:.2f}s"
        )

        # 绘制优化过程图
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
            linewidth=1,
            alpha=0.7,
        )
        plt.xlabel("迭代次数")
        plt.ylabel("适应度(总遮蔽时长)")
        plt.title("PSO优化过程 - 问题三")
        plt.legend()
        plt.grid(True)
        plt.savefig("PSO优化进程_问题三.png", dpi=300)
        plt.show()

        # 分析遮蔽时间段
        print("\n=== 遮蔽时间段分析 ===")
        print(f"共有{len(merged_periods)}个遮蔽时间段:")
        for i, (start, end) in enumerate(merged_periods):
            print(f"段{i+1}: {start:.2f}s - {end:.2f}s (持续{end-start:.2f}s)")

        end_time = time.time()
        print(f"\nPSO优化总耗时: {end_time - start_time:.2f}秒")

        return global_best_position, bombs_data, total_duration, merged_periods

    finally:
        # 关闭进程池
        pool.close()
        pool.join()


def run_pso_three_bombs():
    """运行PSO优化三枚烟幕弹投放策略"""

    print("开始粒子群优化一架无人机投放三枚烟幕弹策略...")

    try:
        # 运行优化
        best_position, bombs_data, total_duration, shield_periods = (
            optimize_with_pso_three_bombs()
        )
        print(f"\n优化成功完成！最大总遮蔽时长: {total_duration:.4f}秒")

    except Exception as e:
        print(f"优化过程中出错: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_pso_three_bombs()
