import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import time
import multiprocessing


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


def calculate_multi_bombs_shielding_duration(
    missile_initial_pos=(20000, 0, 2000),
    missile_velocity=300,
    missile_target=(0, 0, 0),
    plane_initial_pos=(17800, 0, 1800),
    plane_target=(0, 0, 1800),
    plane_speed=120,
    release_times=[1.5, 5.0, 10.0],
    explosion_delays=[3.6, 3.6, 3.6],
    smoke_radius=10,
    smoke_last_time=20,
    cloud_sink_speed=3,
    target_center=(0, 200, 0),
    target_height=10,
    target_radius=7,
    sample_points=100,
    time_step=0.05,
    verbose=False,
):
    """
    计算三枚烟幕弹协同作用下导弹被有效遮蔽的总时长

    参数:
    missile_initial_pos -- 导弹初始位置
    missile_velocity -- 导弹速度(米/秒)
    missile_target -- 导弹目标位置(假目标)
    plane_initial_pos -- 无人机初始位置
    plane_target -- 无人机目标位置
    plane_speed -- 无人机速度(米/秒)
    release_times -- 三枚烟幕弹的投放时间列表
    explosion_delays -- 三枚烟幕弹的起爆延迟列表
    其他参数与单枚烟幕弹相同

    返回:
    tuple -- (总遮蔽时长(秒), 遮蔽时间段列表[(开始时间,结束时间),...]
    """
    # 计算无人机飞行方向
    plane_direction = np.array(plane_target) - np.array(plane_initial_pos)
    plane_direction = plane_direction / np.linalg.norm(plane_direction)

    # 计算每枚烟幕弹的爆炸位置和时间
    explosion_positions = []
    explosion_times = []

    for release_time, explosion_delay in zip(release_times, explosion_delays):
        explosion_pos = calculate_bomb_position(
            plane_initial_pos,
            plane_direction,
            plane_speed,
            release_time,
            explosion_delay,
        )
        explosion_time = release_time + explosion_delay

        explosion_positions.append(explosion_pos)
        explosion_times.append(explosion_time)

    # 生成目标采样点
    target_points = generate_cylinder_sample_points(
        target_center, target_height, target_radius, sample_points
    )

    if verbose:
        print("=== 三枚烟幕弹协同干扰分析 ===")
        print(f"无人机初始位置: {plane_initial_pos}")
        print(f"飞行速度: {plane_speed} m/s，朝向目标: {plane_target}")
        for i, (rt, ed, ep, et) in enumerate(
            zip(release_times, explosion_delays, explosion_positions, explosion_times)
        ):
            print(f"\n烟幕弹 {i+1}:")
            print(f"  投放时间: {rt:.2f}s")
            print(f"  起爆延时: {ed:.2f}s")
            print(f"  爆炸时间: {et:.2f}s")
            print(f"  爆炸位置: ({ep[0]:.1f}, {ep[1]:.1f}, {ep[2]:.1f})")

        print(f"\n烟幕参数:")
        print(f"  烟幕半径: {smoke_radius}m")
        print(f"  烟幕有效时间: {smoke_last_time}s")
        print(f"  云团下沉速度: {cloud_sink_speed} m/s")
        print(f"\n目标信息:")
        print(
            f"  真目标位置: 圆心{target_center}，半径{target_radius}m，高{target_height}m"
        )
        print(f"  采样点数: {len(target_points)}")
        print(f"\n开始模拟协同遮蔽效果...")

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
        # 计算当前导弹位置
        missile_pos = calculate_missile_position(
            missile_initial_pos, t, missile_velocity, missile_target
        )

        # 检查导弹是否已经通过目标区域
        missile_pos_array = np.array(missile_pos)
        if np.linalg.norm(missile_pos_array) < 1:
            if is_currently_shielded:
                shielded_times.append((shield_start_time, t))
                last_shield_info = {
                    "time": t,
                    "missile_pos": missile_pos,
                    "missile_distance": np.linalg.norm(
                        missile_pos_array - np.array(target_center)
                    ),
                }
            break

        # 计算当前各云团位置
        cloud_positions = []
        for explosion_pos, explosion_time in zip(explosion_positions, explosion_times):
            cloud_pos = calculate_cloud_position(
                t, explosion_pos, explosion_time, cloud_sink_speed
            )
            cloud_positions.append(cloud_pos)

        # 判断是否有效遮蔽（考虑多云团协同）
        is_shielded = is_effectively_shielded_by_multiple_clouds(
            missile_pos,
            target_points,
            cloud_positions,
            explosion_times,
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
                    "cloud_positions": [cp for cp in cloud_positions if cp is not None],
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
                "cloud_positions": [cp for cp in cloud_positions if cp is not None],
                "missile_distance": np.linalg.norm(
                    np.array(missile_pos) - np.array(target_center)
                ),
            }

    # 输出详细信息
    if verbose:
        print(f"\n=== 协同遮蔽时间段详情 ===")
        if len(shielded_times) == 0:
            print("未检测到有效遮蔽！")
        else:
            for i, (start, end) in enumerate(shielded_times):
                print(f"第{i+1}段: {start:.2f}s - {end:.2f}s (持续 {end-start:.2f}s)")

            if first_shield_info:
                print(f"\n首次遮蔽时刻 (t={first_shield_info['time']:.2f}s):")
                print(f"  导弹位置: {first_shield_info['missile_pos']}")
                print(f"  导弹距真目标: {first_shield_info['missile_distance']:.1f}m")
                print(f"  活跃云团数: {len(first_shield_info['cloud_positions'])}")

            if last_shield_info:
                print(f"\n最后遮蔽时刻 (t={last_shield_info['time']:.2f}s):")
                print(f"  导弹位置: {last_shield_info['missile_pos']}")
                print(f"  导弹距真目标: {last_shield_info['missile_distance']:.1f}m")
                print(f"  活跃云团数: {len(last_shield_info['cloud_positions'])}")

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
        (0, 1),  # 第一枚投放时间
        (0, 1),  # 第一枚延迟
        (0, 2),  # 第二枚投放时间
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
        df.to_excel("result1.xlsx", index=False)

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
    # 设置随机种子确保可重现性
    random.seed(42)
    np.random.seed(42)

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

#########################T4
# 定义无人机和导弹初始位置
FY1_initial_pos = (17800, 0, 1800)
FY2_initial_pos = (12000, 1400, 1400)
FY3_initial_pos = (6000, -3000, 700)
M1_initial_pos = (20000, 0, 2000)


fitness_cache = {}


def cached_evaluate(position):
    """带缓存的评估函数"""
    key = tuple(round(x, 4) for x in position)
    if key in fitness_cache:
        return fitness_cache[key]

    result = evaluate_position(position)
    fitness_cache[key] = result
    return result


class Particle:
    """粒子类，表示PSO中的一个粒子"""

    def __init__(self, dimensions, bounds):
        """初始化一个粒子"""
        # 初始化位置和速度
        self.position = np.array([random.uniform(low, high) for low, high in bounds])
        self.velocity = np.array(
            [
                random.uniform(-0.5 * (high - low), 0.5 * (high - low))
                for low, high in bounds
            ]
        )

        # 初始化粒子的最佳位置和适应度
        self.best_position = self.position.copy()
        self.best_fitness = -float("inf")

        self.fitness = -float("inf")
        self.bounds = bounds

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
            if self.position[i] < self.bounds[i][0]:
                self.position[i] = self.bounds[i][0]
                self.velocity[i] *= -0.5  # 反弹
            elif self.position[i] > self.bounds[i][1]:
                self.position[i] = self.bounds[i][1]
                self.velocity[i] *= -0.5  # 反弹


def evaluate_position(position):
    """评估一个位置(解)的适应度"""
    # 解析参数
    (
        azimuth1,
        speed1,
        release_time1,
        delay1,
        azimuth2,
        speed2,
        release_time2,
        delay2,
        azimuth3,
        speed3,
        release_time3,
        delay3,
    ) = position

    # 记录每架无人机的贡献
    drone_contributions = [False, False, False]
    total_shielding_periods = []

    # 处理三架无人机的遮蔽计算
    for drone_idx, params in enumerate(
        [
            (FY1_initial_pos, azimuth1, speed1, release_time1, delay1),
            (FY2_initial_pos, azimuth2, speed2, release_time2, delay2),
            (FY3_initial_pos, azimuth3, speed3, release_time3, delay3),
        ]
    ):
        initial_pos, azimuth, speed, release_time, delay = params

        # 计算飞行方向
        direction_x = np.cos(azimuth)
        direction_y = np.sin(azimuth)
        direction_z = 0
        direction = (direction_x, direction_y, direction_z)

        # 计算目标位置
        dir_norm = np.sqrt(direction_x**2 + direction_y**2)
        if dir_norm < 1e-6:  # 避免除零
            continue

        target_distance = 20000
        plane_target = (
            initial_pos[0] + direction_x / dir_norm * target_distance,
            initial_pos[1] + direction_y / dir_norm * target_distance,
            initial_pos[2],
        )

        try:
            _, shield_times = calculate_shielding_duration(
                missile_initial_pos=M1_initial_pos,
                missile_velocity=300,
                missile_target=(0, 0, 0),
                plane_initial_pos=initial_pos,
                plane_target=plane_target,
                plane_speed=speed,
                release_time=release_time,
                explosion_delay=delay,
                sample_points=100,  # 降低精度加快评估
                time_step=0.02,
                verbose=False,
            )

            if shield_times:  # 如果产生了有效遮蔽
                drone_contributions[drone_idx] = True
                total_shielding_periods.extend(shield_times)
        except Exception:
            pass

    # 无有效遮蔽
    if not total_shielding_periods:
        return 0.0

    # 合并重叠区间
    total_shielding_periods.sort(key=lambda x: x[0])
    merged_periods = [total_shielding_periods[0]]

    for current in total_shielding_periods[1:]:
        previous = merged_periods[-1]
        if current[0] <= previous[1]:  # 重叠区间
            merged_periods[-1] = (previous[0], max(previous[1], current[1]))
        else:
            merged_periods.append(current)

    # 计算总时长
    total_duration = sum(end - start for start, end in merged_periods)

    # 添加协同奖励
    active_drones = sum(drone_contributions)
    if active_drones == 2:
        total_duration *= 1.2  # 两架无人机协同奖励20%
    elif active_drones == 3:
        total_duration *= 1.5  # 三架无人机协同奖励50%

    return total_duration


def optimize_with_pso():
    """使用PSO优化三架无人机的烟幕弹投放策略"""
    start_time = time.time()

    # 定义参数的边界 - 更宽泛的范围
    bounds = [
        (0, np.pi / 18),  # FY1: azimuth
        (110, 140),  # FY1: speed
        (0, 4),  # FY1: release_time
        (0, 4),  # FY1: delay
        (-np.pi / 4, 0),  # FY2: azimuth
        (100, 140),  # FY2: speed
        (15, 18),  # FY2: release_time
        (0, 3),  # FY2: delay
        (np.pi / 3, np.pi / 2),  # FY3: azimuth
        (100, 140),  # FY3: speed
        (27, 32),  # FY3: release_time
        (0, 3),  # FY3: delay
    ]

    # PSO参数
    num_particles = 150
    max_iterations = 100

    # 自适应惯性权重
    w_start = 0.9  # 初始较大的惯性权重促进全局搜索
    w_end = 0.4  # 最终较小的惯性权重促进局部精细搜索

    # 创建粒子群
    particles = [Particle(len(bounds), bounds) for _ in range(num_particles)]

    # 初始化全局最佳
    global_best_position = None
    global_best_fitness = -float("inf")

    # 设置并行池
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    # 创建历史记录
    history = {"best_fitness": [], "avg_fitness": []}

    try:
        # 初始评估
        fitnesses = pool.map(cached_evaluate, [p.position for p in particles])
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
            fitnesses = pool.map(cached_evaluate, [p.position for p in particles])

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
                        particles[idx] = Particle(len(bounds), bounds)
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
                        new_particle = Particle(len(bounds), bounds)
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
            azimuth1,
            speed1,
            release_time1,
            delay1,
            azimuth2,
            speed2,
            release_time2,
            delay2,
            azimuth3,
            speed3,
            release_time3,
            delay3,
        ) = global_best_position

        # 计算三架无人机的详细数据
        drones_data = []

        # 计算FY1数据
        direction1_x = np.cos(azimuth1)
        direction1_y = np.sin(azimuth1)
        direction1_z = 0
        direction1 = (direction1_x, direction1_y, direction1_z)

        direction1_angle = (np.arctan2(direction1_y, direction1_x) * 180 / np.pi) % 360

        dir1_norm = np.sqrt(direction1_x**2 + direction1_y**2)
        target_distance = 20000
        plane1_target = (
            FY1_initial_pos[0] + direction1_x / dir1_norm * target_distance,
            FY1_initial_pos[1] + direction1_y / dir1_norm * target_distance,
            FY1_initial_pos[2],
        )

        release_position1 = calculate_plane_position(
            FY1_initial_pos, direction1, speed1, release_time1
        )

        explosion_position1 = calculate_bomb_position(
            FY1_initial_pos, direction1, speed1, release_time1, delay1
        )

        # 高精度评估
        duration1, _ = calculate_shielding_duration(
            missile_initial_pos=M1_initial_pos,
            missile_velocity=300,
            missile_target=(0, 0, 0),
            plane_initial_pos=FY1_initial_pos,
            plane_target=plane1_target,
            plane_speed=speed1,
            release_time=release_time1,
            explosion_delay=delay1,
            sample_points=500,
            verbose=False,
        )

        drones_data.append(
            {
                "编号": "FY1",
                "运动方向": direction1_angle,
                "运动速度": speed1,
                "投放点x": release_position1[0],
                "投放点y": release_position1[1],
                "投放点z": release_position1[2],
                "起爆点x": explosion_position1[0],
                "起爆点y": explosion_position1[1],
                "起爆点z": explosion_position1[2],
                "有效干扰时长": duration1,
                "投放时间": release_time1,  # 添加投放时间
                "引爆时间": delay1,  # 添加引爆时间
            }
        )

        # 计算FY2数据
        direction2_x = np.cos(azimuth2)
        direction2_y = np.sin(azimuth2)
        direction2_z = 0
        direction2 = (direction2_x, direction2_y, direction2_z)

        direction2_angle = (np.arctan2(direction2_y, direction2_x) * 180 / np.pi) % 360

        dir2_norm = np.sqrt(direction2_x**2 + direction2_y**2)
        plane2_target = (
            FY2_initial_pos[0] + direction2_x / dir2_norm * target_distance,
            FY2_initial_pos[1] + direction2_y / dir2_norm * target_distance,
            FY2_initial_pos[2],
        )

        release_position2 = calculate_plane_position(
            FY2_initial_pos, direction2, speed2, release_time2
        )

        explosion_position2 = calculate_bomb_position(
            FY2_initial_pos, direction2, speed2, release_time2, delay2
        )

        duration2, _ = calculate_shielding_duration(
            missile_initial_pos=M1_initial_pos,
            missile_velocity=300,
            missile_target=(0, 0, 0),
            plane_initial_pos=FY2_initial_pos,
            plane_target=plane2_target,
            plane_speed=speed2,
            release_time=release_time2,
            explosion_delay=delay2,
            sample_points=500,
            verbose=False,
        )

        drones_data.append(
            {
                "编号": "FY2",
                "运动方向": direction2_angle,
                "运动速度": speed2,
                "投放点x": release_position2[0],
                "投放点y": release_position2[1],
                "投放点z": release_position2[2],
                "起爆点x": explosion_position2[0],
                "起爆点y": explosion_position2[1],
                "起爆点z": explosion_position2[2],
                "有效干扰时长": duration2,
                "投放时间": release_time2,  # 添加投放时间
                "引爆时间": delay2,  # 添加引爆时间
            }
        )

        # 计算FY3数据
        direction3_x = np.cos(azimuth3)
        direction3_y = np.sin(azimuth3)
        direction3_z = 0
        direction3 = (direction3_x, direction3_y, direction3_z)

        direction3_angle = (np.arctan2(direction3_y, direction3_x) * 180 / np.pi) % 360

        dir3_norm = np.sqrt(direction3_x**2 + direction3_y**2)
        plane3_target = (
            FY3_initial_pos[0] + direction3_x / dir3_norm * target_distance,
            FY3_initial_pos[1] + direction3_y / dir3_norm * target_distance,
            FY3_initial_pos[2],
        )

        release_position3 = calculate_plane_position(
            FY3_initial_pos, direction3, speed3, release_time3
        )

        explosion_position3 = calculate_bomb_position(
            FY3_initial_pos, direction3, speed3, release_time3, delay3
        )

        duration3, _ = calculate_shielding_duration(
            missile_initial_pos=M1_initial_pos,
            missile_velocity=300,
            missile_target=(0, 0, 0),
            plane_initial_pos=FY3_initial_pos,
            plane_target=plane3_target,
            plane_speed=speed3,
            release_time=release_time3,
            explosion_delay=delay3,
            sample_points=500,
            verbose=False,
        )

        drones_data.append(
            {
                "编号": "FY3",
                "运动方向": direction3_angle,
                "运动速度": speed3,
                "投放点x": release_position3[0],
                "投放点y": release_position3[1],
                "投放点z": release_position3[2],
                "起爆点x": explosion_position3[0],
                "起爆点y": explosion_position3[1],
                "起爆点z": explosion_position3[2],
                "有效干扰时长": duration3,
                "投放时间": release_time3,  # 添加投放时间
                "引爆时间": delay3,  # 添加引爆时间
            }
        )

        # 计算总有效干扰时长（高精度评估）
        total_shielding_periods = []

        # 收集三架无人机的所有遮蔽区间
        for drone_idx, (pos, dir, speed, release_time, delay) in enumerate(
            [
                (FY1_initial_pos, direction1, speed1, release_time1, delay1),
                (FY2_initial_pos, direction2, speed2, release_time2, delay2),
                (FY3_initial_pos, direction3, speed3, release_time3, delay3),
            ]
        ):
            # 计算目标位置
            dir_x, dir_y, dir_z = dir
            dir_norm = np.sqrt(dir_x**2 + dir_y**2)
            target_distance = 20000
            plane_target = (
                pos[0] + dir_x / dir_norm * target_distance,
                pos[1] + dir_y / dir_norm * target_distance,
                pos[2],
            )

            # 计算遮蔽区间
            try:
                _, shield_times = calculate_shielding_duration(
                    missile_initial_pos=M1_initial_pos,
                    missile_velocity=300,
                    missile_target=(0, 0, 0),
                    plane_initial_pos=pos,
                    plane_target=plane_target,
                    plane_speed=speed,
                    release_time=release_time,
                    explosion_delay=delay,
                    sample_points=500,
                    verbose=False,
                )
                total_shielding_periods.extend(shield_times)
            except Exception as e:
                print(f"计算无人机{drone_idx+1}的遮蔽区间时出错: {e}")

        # 合并重叠区间
        if total_shielding_periods:
            total_shielding_periods.sort(key=lambda x: x[0])
            merged_periods = [total_shielding_periods[0]]

            for current in total_shielding_periods[1:]:
                previous = merged_periods[-1]
                if current[0] <= previous[1]:
                    merged_periods[-1] = (previous[0], max(previous[1], current[1]))
                else:
                    merged_periods.append(current)

            total_duration = sum(end - start for start, end in merged_periods)
        else:
            total_duration = 0.0
            merged_periods = []

        # 创建结果DataFrame
        df = pd.DataFrame(
            columns=[
                "无人机编号",
                "无人机运动方向",
                "无人机运动速度 (m/s)",
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
        for i, drone in enumerate(drones_data):
            df.loc[i] = [
                drone["编号"],
                drone["运动方向"],
                drone["运动速度"],
                drone["投放点x"],
                drone["投放点y"],
                drone["投放点z"],
                drone["起爆点x"],
                drone["起爆点y"],
                drone["起爆点z"],
                drone["有效干扰时长"],
            ]

        # 输出优化结果
        print("\n=== 三架无人机协同投放PSO优化结果 ===")
        print(f"总有效干扰时长: {total_duration:.4f}秒")
        for i, drone in enumerate(drones_data):
            drone_id = drone["编号"]
            print(f"\n{drone_id}:")
            print(f"  运动方向: {drone['运动方向']:.2f}度")
            print(f"  运动速度: {drone['运动速度']:.2f} m/s")
            print(
                f"  投放点: ({drone['投放点x']:.2f}, {drone['投放点y']:.2f}, {drone['投放点z']:.2f})"
            )
            print(
                f"  起爆点: ({drone['起爆点x']:.2f}, {drone['起爆点y']:.2f}, {drone['起爆点z']:.2f})"
            )
            print(f"  有效干扰时长: {drone['有效干扰时长']:.2f}秒")
            print(f"  投放时间: {drone['投放时间']:.2f}秒")  # 输出投放时间
            print(f"  引爆时间: {drone['引爆时间']:.2f}秒")  # 输出引爆时间

        # 分析遮蔽时间段
        print("\n=== 遮蔽时间段分析 ===")
        print(f"共有{len(merged_periods)}个遮蔽时间段:")
        for i, (start, end) in enumerate(merged_periods):
            print(f"段{i+1}: {start:.2f}s - {end:.2f}s (持续{end-start:.2f}s)")

        end_time = time.time()
        print(f"\nPSO优化总耗时: {end_time - start_time:.2f}秒")

        return global_best_position, drones_data, total_duration, merged_periods

    finally:
        # 关闭进程池
        pool.close()
        pool.join()


def run_pso_optimization():
    """运行PSO优化"""

    print("开始粒子群优化三架无人机协同投放烟幕弹...")

    try:
        # 运行优化
        best_position, drones_data, total_duration, shield_periods = optimize_with_pso()
        print(f"\n优化成功完成！最大总遮蔽时长: {total_duration:.4f}秒")

    except Exception as e:
        print(f"优化过程中出错: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_pso_optimization()

########################    T5


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


def calculate_multi_smoke_shielding_duration(
    missile_initial_pos,
    missile_velocity,
    missile_target,
    plane_initial_positions,
    plane_targets,
    plane_speeds,
    release_times,
    explosion_delays,
    target_center=(0, 200, 0),
    target_height=10,
    target_radius=7,
    sample_points=100,
    time_step=0.05,
    verbose=False,
):
    """计算多枚烟幕弹对单枚导弹的协同遮蔽时长"""
    # 计算所有无人机的飞行方向
    plane_directions = []
    for plane_initial_pos, plane_target in zip(plane_initial_positions, plane_targets):
        direction = np.array(plane_target) - np.array(plane_initial_pos)
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 0:
            direction = direction / direction_norm
        else:
            direction = np.array([1, 0, 0])  # 默认方向
        plane_directions.append(tuple(direction))

    # 计算每枚烟幕弹的爆炸位置和时间
    explosion_positions = []
    explosion_times = []

    for (
        plane_initial_pos,
        plane_direction,
        plane_speed,
        release_time,
        explosion_delay,
    ) in zip(
        plane_initial_positions,
        plane_directions,
        plane_speeds,
        release_times,
        explosion_delays,
    ):
        explosion_pos = calculate_bomb_position(
            plane_initial_pos,
            plane_direction,
            plane_speed,
            release_time,
            explosion_delay,
        )
        explosion_time = release_time + explosion_delay

        explosion_positions.append(explosion_pos)
        explosion_times.append(explosion_time)

    # 生成目标采样点
    target_points = generate_cylinder_sample_points(
        target_center, target_height, target_radius, sample_points
    )

    # 遍历时间步长
    time_start = 0
    time_end = 100  # 足够长的时间范围

    shielded_times = []
    is_currently_shielded = False
    shield_start_time = None

    for t in np.arange(time_start, time_end, time_step):
        # 计算当前导弹位置
        missile_pos = calculate_missile_position(
            missile_initial_pos, t, missile_velocity, missile_target
        )

        # 检查导弹是否已经通过目标区域
        missile_pos_array = np.array(missile_pos)
        if np.linalg.norm(missile_pos_array) < 1:
            if is_currently_shielded:
                shielded_times.append((shield_start_time, t))
            break

        # 计算当前各云团位置
        cloud_positions = []
        for explosion_pos, explosion_time in zip(explosion_positions, explosion_times):
            cloud_pos = calculate_cloud_position(
                t, explosion_pos, explosion_time, 3  # cloud_sink_speed=3
            )
            cloud_positions.append(cloud_pos)

        # 判断是否有效遮蔽
        is_shielded = is_effectively_shielded_by_multiple_clouds(
            missile_pos,
            target_points,
            cloud_positions,
            explosion_times,
            t,
            10,  # smoke_radius=10
            20,  # smoke_last_time=20
        )

        # 记录遮蔽时间段
        if is_shielded and not is_currently_shielded:
            is_currently_shielded = True
            shield_start_time = t
        elif not is_shielded and is_currently_shielded:
            is_currently_shielded = False
            shielded_times.append((shield_start_time, t))

    # 计算总遮蔽时长
    total_duration = sum(end - start for start, end in shielded_times)
    return total_duration, shielded_times


def enforce_release_time_constraints(position):
    """
    对单个粒子的位置向量强制施加投放时间间隔至少为1秒的约束。
    该函数会按时间顺序重新排列烟幕弹，并确保它们之间有足够的时间间隔。
    """
    for i in range(5):  # 遍历5架无人机
        base_idx = i * 8

        # 提取每枚烟幕弹的 (投放时间, 延迟时间) 对
        releases = []
        for j in range(3):  # 3枚烟幕弹
            time_idx = base_idx + 2 + j * 2
            delay_idx = base_idx + 3 + j * 2
            releases.append([position[time_idx], position[delay_idx]])

        # 1. 关键一步：按投放时间对烟幕弹进行排序
        #    这确保了我们总是比较时间上相邻的两次投放。
        releases.sort(key=lambda x: x[0])

        # 2. 强制施加1秒的时间间隔
        #    比较第2枚和第1枚
        if releases[1][0] < releases[0][0] + 1.0:
            releases[1][0] = releases[0][0] + 1.0
        #    比较第3枚和第2枚
        if releases[2][0] < releases[1][0] + 1.0:
            releases[2][0] = releases[1][0] + 1.0

        # 3. 将修复后的值写回原始位置向量
        for j in range(3):
            time_idx = base_idx + 2 + j * 2
            delay_idx = base_idx + 3 + j * 2
            position[time_idx] = releases[j][0]
            position[delay_idx] = releases[j][1]

    return position


drone_positions = [
    (17800, 0, 1800),  # FY1
    (12000, 1400, 1400),  # FY2
    (6000, -3000, 700),  # FY3
    (15000, -2000, 1200),  # FY4
    (9000, 2500, 900),  # FY5
]

missile_positions = [
    (20000, 0, 2000),  # M1
    (19000, 2000, 2200),  # M2
    (18000, -1500, 1900),  # M3
]

# 真目标位置
target_center = (0, 200, 0)


def calculate_total_shielding_duration(
    drone_params,
    drone_positions,
    missile_positions,
    target_center=(0, 200, 0),
    sample_points=50,
):
    """计算总的遮蔽时长(所有导弹的总和)"""
    total_duration = 0.0

    # 对每枚导弹计算遮蔽效果
    for missile_idx, missile_pos in enumerate(missile_positions):
        # 收集所有无人机针对该导弹的所有烟幕弹参数
        all_bomb_params = []

        for drone_idx, drone_param in enumerate(drone_params):
            # 计算无人机的飞行方向向量
            azimuth = drone_param["azimuth"]
            speed = drone_param["speed"]

            direction_x = np.cos(azimuth)
            direction_y = np.sin(azimuth)
            direction = (direction_x, direction_y, 0)

            # 计算目标位置
            dir_norm = np.sqrt(direction_x**2 + direction_y**2)
            if dir_norm < 1e-6:  # 避免除零
                continue

            target_distance = 20000
            drone_target = (
                drone_positions[drone_idx][0]
                + direction_x / dir_norm * target_distance,
                drone_positions[drone_idx][1]
                + direction_y / dir_norm * target_distance,
                drone_positions[drone_idx][2],
            )

            # 添加该无人机的所有烟幕弹参数
            for release_time, explosion_delay in drone_param["releases"]:
                if release_time <= 20:  # 只考虑实际投放的烟幕弹（不超过20秒）
                    all_bomb_params.append(
                        {
                            "plane_initial_pos": drone_positions[drone_idx],
                            "plane_target": drone_target,
                            "plane_speed": speed,
                            "release_time": release_time,
                            "explosion_delay": explosion_delay,
                        }
                    )

        # 如果有针对该导弹的烟幕弹，计算遮蔽时长
        if all_bomb_params:
            # 限制每枚导弹最多考虑10枚烟幕弹（性能优化）
            all_bomb_params.sort(key=lambda x: x["release_time"])  # 按投放时间排序
            bomb_count = min(len(all_bomb_params), 10)
            selected_bombs = all_bomb_params[:bomb_count]

            # 提取参数
            plane_initial_positions = [
                bomb["plane_initial_pos"] for bomb in selected_bombs
            ]
            plane_targets = [bomb["plane_target"] for bomb in selected_bombs]
            plane_speeds = [bomb["plane_speed"] for bomb in selected_bombs]
            release_times = [bomb["release_time"] for bomb in selected_bombs]
            explosion_delays = [bomb["explosion_delay"] for bomb in selected_bombs]

            # 计算针对该导弹的遮蔽时长
            try:
                duration, _ = calculate_multi_smoke_shielding_duration(
                    missile_initial_pos=missile_pos,
                    missile_velocity=300,
                    missile_target=(0, 0, 0),
                    plane_initial_positions=plane_initial_positions,
                    plane_targets=plane_targets,
                    plane_speeds=plane_speeds,
                    release_times=release_times,
                    explosion_delays=explosion_delays,
                    target_center=target_center,
                    sample_points=sample_points,
                    time_step=0.05,  # 增大时间步长以加速计算
                    verbose=False,
                )
                total_duration += duration
            except Exception as e:
                # 如果计算出错，忽略该导弹
                pass

    # 添加奖励：鼓励平均分配遮蔽效果和资源利用
    if total_duration > 0:
        # 计算无人机参与度 - 每架无人机投放的烟幕弹数量
        drone_participation = [0] * 5
        for i in range(5):
            for release_time, _ in drone_params[i]["releases"]:
                if release_time <= 20:  # 考虑实际投放的烟幕弹
                    drone_participation[i] += 1

        # 奖励多架无人机参与
        active_drones = sum(1 for p in drone_participation if p > 0)
        if active_drones >= 3:
            total_duration *= 1.1  # 奖励至少3架无人机参与
        if active_drones >= 4:
            total_duration *= 1.1  # 更多奖励4架及以上参与

        # 奖励平均分布的投放时间
        all_release_times = []
        for drone in drone_params:
            for release_time, _ in drone["releases"]:
                if release_time <= 20:
                    all_release_times.append(release_time)

        if all_release_times:
            time_range = max(all_release_times) - min(all_release_times)
            if time_range > 10:  # 如果投放时间跨度大，给予奖励
                total_duration *= 1.05

    return total_duration


fitness_cache = {}


def position_to_key(position):
    """将位置向量转换为可哈希的键，用于缓存"""
    # 截断到小数点后4位以增加缓存命中率
    return tuple(float(f"{x:.4f}") for x in position)


def evaluate_position(position):
    """评估一个位置(解)的适应度"""
    # 检查缓存
    key = position_to_key(position)
    if key in fitness_cache:
        return fitness_cache[key]

    # 解码参数
    drone_params = []
    for i in range(5):  # 5架无人机
        drone_param = {
            "azimuth": position[i * 8],
            "speed": position[i * 8 + 1],
            "releases": [],
        }

        # 添加3枚烟幕弹的参数
        for j in range(3):
            release_time = position[i * 8 + 2 + j * 2]
            explosion_delay = position[i * 8 + 3 + j * 2]
            drone_param["releases"].append((release_time, explosion_delay))

        drone_params.append(drone_param)

    # 计算对3枚导弹的总遮蔽时长
    total_duration = calculate_total_shielding_duration(
        drone_params, drone_positions, missile_positions, target_center
    )

    # 缓存结果
    fitness_cache[key] = total_duration
    return total_duration


def optimize_with_pso_five_vs_three():
    """使用粒子群算法优化5架无人机对抗3枚导弹的策略"""
    start_time = time.time()

    # 定义5架无人机和3枚导弹的初始位置

    # 定义参数边界 - 每架无人机的飞行方向、速度、每枚烟幕弹的投放时间和延迟
    # 每架无人机8个参数：方向角、速度、3枚烟幕弹的投放时间和延迟
    bounds = []
    for _ in range(5):  # 5架无人机
        # 方向角范围 [-pi/2, pi/2]
        bounds.append((-np.pi / 2, np.pi / 2))
        # 速度范围 [80, 140] m/s
        bounds.append((80, 140))

        # 3枚烟幕弹的参数
        for _ in range(3):
            # 投放时间范围 [0, 40] 秒
            bounds.append((0, 40))
            # 延迟时间范围 [0, 5] 秒
            bounds.append((0, 5))

    # 总共40个参数：5架无人机 × (1个方向角 + 1个速度 + 3枚烟幕弹×2个参数)

    # 创建并行管理器
    manager = multiprocessing.Manager()
    fitness_cache = manager.dict()

    class Particle:
        """粒子类，代表PSO中的一个候选解"""

        def __init__(self, dimensions):
            """初始化一个粒子"""
            # 随机生成位置向量
            self.position = np.array(
                [random.uniform(low, high) for low, high in bounds]
            )
            self.position = enforce_release_time_constraints(self.position)

            # 随机生成速度向量
            self.velocity = np.array(
                [
                    random.uniform(-0.3 * (high - low), 0.3 * (high - low))
                    for low, high in bounds
                ]
            )

            # 初始化最佳位置和适应度
            self.best_position = self.position.copy()
            self.best_fitness = -float("inf")
            self.fitness = -float("inf")

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
                    self.velocity[i] *= -0.5  # 速度反弹
                elif self.position[i] > bounds[i][1]:
                    self.position[i] = bounds[i][1]
                    self.velocity[i] *= -0.5  # 速度反弹
            self.position = enforce_release_time_constraints(self.position)

    # PSO参数
    num_particles = 300
    max_iterations = 200

    # 自适应惯性权重
    w_start = 0.9
    w_end = 0.4

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
        print(
            f"使用PSO优化5架无人机对抗3枚导弹，粒子数量={num_particles}，最大迭代={max_iterations}"
        )
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
            if abs(global_best_fitness - prev_best) < 0.01:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
                prev_best = global_best_fitness

            if stagnation_counter >= 80:
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
        print(f"最佳适应度值(总干扰时长): {global_best_fitness:.4f}秒")

        # 解码最佳解并生成详细结果
        best_drone_params = []
        for i in range(5):
            drone_params = {
                "drone_id": i + 1,
                "azimuth": global_best_position[i * 8],
                "speed": global_best_position[i * 8 + 1],
                "releases": [],
            }

            for j in range(3):
                release_time = global_best_position[i * 8 + 2 + j * 2]
                explosion_delay = global_best_position[i * 8 + 3 + j * 2]
                drone_params["releases"].append((release_time, explosion_delay))

            best_drone_params.append(drone_params)

        # 创建最终结果数据
        solution_data = []
        bomb_id = 1

        for drone_idx, drone_param in enumerate(best_drone_params):
            drone_id = drone_param["drone_id"]
            azimuth = drone_param["azimuth"]
            speed = drone_param["speed"]

            # 计算飞行方向向量
            direction_x = np.cos(azimuth)
            direction_y = np.sin(azimuth)
            direction = (direction_x, direction_y, 0)

            # 计算方向角度
            direction_angle = (np.arctan2(direction_y, direction_x) * 180 / np.pi) % 360

            # 计算目标位置
            dir_norm = np.sqrt(direction_x**2 + direction_y**2)
            target_distance = 20000
            drone_target = (
                drone_positions[drone_idx][0]
                + direction_x / dir_norm * target_distance,
                drone_positions[drone_idx][1]
                + direction_y / dir_norm * target_distance,
                drone_positions[drone_idx][2],
            )

            # 计算每枚烟幕弹的详细信息
            for release_idx, (release_time, explosion_delay) in enumerate(
                drone_param["releases"]
            ):
                # 如果超过20秒，视为不投放
                if release_time > 40:
                    continue

                # 计算投放点和起爆点
                release_position = calculate_plane_position(
                    drone_positions[drone_idx], direction, speed, release_time
                )

                explosion_position = calculate_bomb_position(
                    drone_positions[drone_idx],
                    direction,
                    speed,
                    release_time,
                    explosion_delay,
                )

                # 计算对各导弹的干扰时长
                missile_durations = []
                for missile_idx, missile_pos in enumerate(missile_positions):
                    try:
                        duration, _ = calculate_shielding_duration(
                            missile_initial_pos=missile_pos,
                            missile_velocity=300,
                            missile_target=(0, 0, 0),
                            plane_initial_pos=drone_positions[drone_idx],
                            plane_target=drone_target,
                            plane_speed=speed,
                            release_time=release_time,
                            explosion_delay=explosion_delay,
                            target_center=target_center,
                            verbose=False,
                            sample_points=200,  # 提高最终评估的精度
                        )
                        missile_durations.append(duration)
                    except:
                        missile_durations.append(0.0)

                # 确保有3个导弹的干扰时长
                while len(missile_durations) < 3:
                    missile_durations.append(0.0)

                solution_data.append(
                    {
                        "无人机编号": f"FY{drone_id}",
                        "无人机运动方向": direction_angle,
                        "无人机运动速度 (m/s)": speed,
                        "烟幕干扰弹编号": bomb_id,
                        "烟幕干扰弹投放点的x坐标 (m)": release_position[0],
                        "烟幕干扰弹投放点的y坐标 (m)": release_position[1],
                        "烟幕干扰弹投放点的z坐标 (m)": release_position[2],
                        "烟幕干扰弹起爆点的x坐标 (m)": explosion_position[0],
                        "烟幕干扰弹起爆点的y坐标 (m)": explosion_position[1],
                        "烟幕干扰弹起爆点的z坐标 (m)": explosion_position[2],
                        "针对导弹M1的有效干扰时长 (s)": missile_durations[0],
                        "针对导弹M2的有效干扰时长 (s)": missile_durations[1],
                        "针对导弹M3的有效干扰时长 (s)": missile_durations[2],
                    }
                )

                bomb_id += 1

        end_time = time.time()
        print(f"\nPSO优化总耗时: {end_time - start_time:.2f}秒")
        df = pd.DataFrame(solution_data)
        df.to_excel("result3.xlsx", index=False)
        print("结果已保存到result3.xlsx")
        return global_best_position, best_drone_params, global_best_fitness

    finally:
        # 关闭并行池
        pool.close()
        pool.join()


if __name__ == "__main__":
    # 设置随机种子

    print("开始粒子群优化五架无人机对抗三枚导弹的策略...")

    try:
        # 运行优化
        best_position, best_params, total_duration = optimize_with_pso_five_vs_three()
        print(f"\n优化成功完成！最大总干扰时长: {total_duration:.4f}秒")

        # 打印各无人机的最佳参数
        for i, drone in enumerate(best_params):
            print(f"\n无人机FY{i+1}:")
            print(
                f"  运动方向: {np.arctan2(np.sin(drone['azimuth']), np.cos(drone['azimuth'])) * 180 / np.pi:.2f}度"
            )
            print(f"  运动速度: {drone['speed']:.2f} m/s")
            for j, (release_time, explosion_delay) in enumerate(drone["releases"]):
                if release_time <= 40:  # 有效投放
                    print(
                        f"  烟幕弹{j+1}: 投放时间={release_time:.2f}s, 延迟={explosion_delay:.2f}s"
                    )
                else:
                    print(f"  烟幕弹{j+1}: 不投放")

    except Exception as e:
        print(f"优化过程中出错: {e}")
        import traceback

        traceback.print_exc()
