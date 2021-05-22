import numpy as np


def line(agent_num, position_range):
    XY = []
    for i in np.linspace(position_range[0], position_range[1], agent_num):
        XY.append([i, 0])
    return np.array(XY, dtype=np.float32)


def nest_radiation(agent_num, position_range):
    max_pos = np.max(position_range) * 0.8
    # 2pi被平均分为20份
    delta_angle = 2*np.pi/20
    # 每条射线上点间的间隔
    each_part_num = int(agent_num / 20)
    delta_dis = max_pos/(each_part_num-1)
    #
    XY = []
    for i in range(20):
        # 每条射线的角度
        angle = i * delta_angle
        # 每条射线上的所有点
        for j in range(each_part_num):
            p = delta_dis * j
            x = np.cos(angle) * p
            y = np.sin(angle) * p
            XY.append([x,y])
    return np.array(XY, dtype=np.float32)


def nest_square(agent_num, r_c=0.5):
    # 每行每列智能体数量
    r_C = r_c
    #
    XY = []
    XY.append(np.array([0,0]))
    while True:
        for idx in range(1, 10):
            i = idx * 0.5
            XY.append([i, i])
            XY.append([-i, i])
            XY.append([-i, -i])
            XY.append([i, -i])
            for quadrant in range(4):
                if quadrant == 0:
                    for x in np.linspace(-i+r_C, i-r_C, int(i*2//0.5-1)):
                        XY.append([x, i])
                elif quadrant == 1:
                    for x in np.linspace(-i+r_C, i-r_C, int(i*2//0.5-1)):
                        XY.append([x, -i])
                elif quadrant == 2:
                    for y in np.linspace(-i+r_C, i-r_C, int(i*2//0.5-1)):
                        XY.append([-i, y])
                else:
                    for y in np.linspace(-i+r_C, i-r_C, int(i*2//0.5-1)):
                        XY.append([i, y])
        if len(XY) >= agent_num:
            break

    total_locs = np.array(XY, dtype=np.float32)
    print(total_locs.shape)
    total_locs = total_locs[:agent_num]
    print(total_locs.shape)
    return total_locs

def square(agent_num):
    def gen_one_square(p1, p2, p3, p4, sample_num, part=1):
        """
        p1 - p2
        |    |
        p4 - p3
        """
        PS = []
        # 分四段，每段采样sample_num个，最后再提取sample_num个
        # p1-p2
        if part == 1 or part == 2:
            ss = int(sample_num // 4)
        elif part == 3:
            ss = int(sample_num // 3.6)
        else:
            ss = int(sample_num // 2.6)
        PS.append(
            np.column_stack((
                np.linspace(p1[0], p2[0], ss).astype(np.float32),
                np.ones(ss, dtype=np.float32) * p1[1],
            ))
        )
        # p2-p3
        PS.append(
            np.column_stack((
                np.ones(ss, dtype=np.float32) * p2[0],
                np.linspace(p3[1], p2[1], ss).astype(np.float32),
            ))
        )
        # p3-p4
        PS.append(
            np.column_stack((
                np.linspace(p3[0], p4[0], ss).astype(np.float32),
                np.ones(ss, dtype=np.float32) * p3[1],
            ))
        )
        # p4-p1
        PS.append(
            np.column_stack((
                np.ones(ss, dtype=np.float32) * p4[0],
                np.linspace(p4[1], p1[1], ss).astype(np.float32),

            ))
        )
        PS = np.vstack(PS[:])
        select_idx = np.random.choice(PS.shape[0], sample_num, replace=False)
        return PS[select_idx]

    # 3层正方形，中心点有散点
    parts_ratios = list(reversed([0.045, 0.18, 0.3, 0.38]))
    base_ratios = [0.74, 0.52, 0.30, 0.08]

    # [1]
    base1 = base_ratios[0] * 5
    num1 = int(parts_ratios[0] * agent_num)
    PS1 = gen_one_square(
        p1=[-base1, base1], p2=[base1, base1],
        p3=[base1, -base1], p4=[-base1, -base1], sample_num=num1, part=1
    )

    # [2]
    base2 = base_ratios[1] * 5
    num2 = int(parts_ratios[1] * agent_num)
    PS2 = gen_one_square(
        p1=[-base2, base2], p2=[base2, base2],
        p3=[base2, -base2], p4=[-base2, -base2], sample_num=num2, part=2
    )

    # [3]
    base3 = base_ratios[2] * 5
    num3 = int(parts_ratios[2] * agent_num)
    PS3 = gen_one_square(
        p1=[-base3, base3], p2=[base3, base3],
        p3=[base3, -base3], p4=[-base3, -base3], sample_num=num3, part=3
    )

    # [4] 中心散点
    base4 = base_ratios[3] * 5
    num4 = int(parts_ratios[3] * agent_num)
    PS4 = gen_one_square(
        p1=[-base4, base4], p2=[base4, base4],
        p3=[base4, -base4], p4=[-base4, -base4], sample_num=num4, part=4
    )

    # [0] 0是随机的扰动和桥梁
    ratios = [0, 0.29, 0.25, 0.24, 0.24]      # 随机扰动和桥梁的占比
    num0 = agent_num - num1 - num2 - num3 - num4
    num0_0 = int(num0 * ratios[0])    # 随机

    num0_0_1 = int(num0_0 * 0.6)
    num0_0_2 = int(num0_0 * 0.4)
    PS_0_0_1_rand_flag = np.random.uniform(0, 1, (num0_0_1, 2))
    PS_0_0_1_select = np.select([PS_0_0_1_rand_flag < 0.5, PS_0_0_1_rand_flag >= 0.5],
                                [1, -1])
    PS0_0_1 = np.random.uniform(3.62, 3.78, (num0_0_1, 2)) * PS_0_0_1_select

    PS_0_0_2_rand_flag = np.random.uniform(0, 1, (num0_0_2, 2))
    PS_0_0_2_select = np.select([PS_0_0_2_rand_flag >= 0.5, PS_0_0_2_rand_flag < 0.5],
                                [1, -1])
    PS0_0_2 = np.random.uniform(2.32, 2.56, (num0_0_2, 2)) * PS_0_0_2_select


    PS0_0 = np.vstack((PS0_0_1, PS0_0_2))

    num0_1 = int(num0 * ratios[1])     # base1-base2_1
    PS0_1 = np.random.uniform(-base1+0.1, -base2-0.1, (num0_1, 2))
    PS0_1[:, 1] = np.random.uniform(-0.15, 0.15, num0_1)

    num0_2 = int(num0 * ratios[2])     # base1-base2_2
    PS0_2 = np.random.uniform(base1, base2, (num0_2, 2))
    PS0_2[:, 1] = np.random.uniform(-0.2, 0.2, num0_2)

    num0_3 = int(num0 * ratios[3])     # base2-base3
    PS0_3 = np.random.uniform(base2, base3, (num0_3, 2))
    PS0_3[:, 0] = np.random.uniform(-0.2, 0.2, num0_3)

    num0_4 = num0 - num0_0 - num0_1 - num0_2 - num0_3     # base3-base4
    PS0_4 = np.random.uniform(-base3, -base4, (num0_4, 2))
    PS0_4[:, 0] = np.random.uniform(-0.2, 0.2, num0_4)

    PS0 = np.vstack((PS0_0, PS0_1, PS0_2, PS0_3, PS0_4))
    return np.vstack((PS0, PS1, PS2, PS3, PS4))

def triangle(agent_num, position_range, r_C=0.5):

    side_length = 0.9 * (position_range[0][1] - position_range[0][0])
    height = np.sqrt(side_length ** 2 - (side_length / 2) ** 2)
    triangle_points = [
        [np.sum(position_range) / 2 - side_length / 2, np.sum(position_range) / 2 - height / 2],
        [np.sum(position_range) / 2, np.sum(position_range) / 2 + height / 2],
        [np.sum(position_range) / 2 + side_length / 2, np.sum(position_range) / 2 - height / 2],
    ]

    def random_point(agent_num, triangle_points):
        x1, y1 = triangle_points[0][0], triangle_points[0][1]
        x3, y3 = triangle_points[1][0], triangle_points[1][1]
        x2, y2 = triangle_points[2][0], triangle_points[2][1]
        # theta = np.arange(0, 1, 0.001)
        rnd1 = np.random.random(size=agent_num)
        rnd2 = np.random.random(size=agent_num)
        rnd2 = np.sqrt(rnd2)
        x = rnd2 * (rnd1 * x1 + (1 - rnd1) * x2) + (1 - rnd2) * x3
        y = rnd2 * (rnd1 * y1 + (1 - rnd1) * y2) + (1 - rnd2) * y3
        return x, y

    x, y = random_point(agent_num*3, triangle_points)
    total_locations = np.vstack((x, y)).T

    np.random.seed(19961110)
    np.random.shuffle(total_locations)
    np.random.shuffle(total_locations)
    print(total_locations.shape)
    total_ids = list(range(total_locations.shape[0]))
    select_ids = [total_ids[0]]
    for _ in range(agent_num):
        print("select:", len(select_ids))
        for tmp_id in total_ids:
            flag = False
            num = 0
            tmp_pos = total_locations[tmp_id]
            for one_sid in range(len(select_ids)):
                between_dis = np.linalg.norm(
                    total_locations[select_ids[one_sid]] -
                    tmp_pos
                )
                if between_dis >= 0.01 and between_dis <= r_C:
                    num += 1

            if num >= 1 and num <= 2:
                flag = True
                total_ids.remove(tmp_id)
                select_ids.append(tmp_id)
            if flag:
                break
        if len(select_ids) >= agent_num:
            break

    print(total_locations.shape)
    print(total_locations[np.array(select_ids)].shape)
    return total_locations[np.array(select_ids)]

def ring(agent_num, position_range):
    sample_num = agent_num * 2

    delta_angle = 360 / sample_num
    goal_circle_r = 4
    circle_center = [np.sum(position_range[0])/2, np.sum(position_range[1])/2]
    angles = np.array([delta_angle * i for i in range(sample_num)])
    radians = angles * np.pi / 180
    x = goal_circle_r * np.cos(radians) + circle_center[0]
    y = goal_circle_r * np.sin(radians) + circle_center[1]

    for i in range(sample_num):
        # 中心螺旋线的智能体80% 随机走几步
        if np.random.rand() < 0.6:
            # for step in range(20):
            noise_x = np.random.uniform(-0.3, 0.3)
            noise_y = np.random.uniform(-0.3, 0.3)
            x[i] = x[i] + noise_x
            y[i] = y[i] + noise_y

    total_locations = np.vstack((x, y)).T
    print(total_locations.shape)
    np.random.shuffle(total_locations)
    np.random.shuffle(total_locations)
    np.random.shuffle(total_locations)

    total_ids = list(range(total_locations.shape[0]))
    select_ids = [total_ids[0]]
    for _ in range(agent_num):
        print("select:", len(select_ids))
        for tmp_id in total_ids:
            flag = False
            num = 0
            tmp_pos = total_locations[tmp_id]
            for one_sid in range(len(select_ids)):
                between_dis = np.linalg.norm(
                    total_locations[select_ids[one_sid]] -
                    tmp_pos
                )
                if between_dis >= 0.001 and between_dis <= 0.5:
                    num += 1
            if num>= 1 and num <= 5:
                flag = True
                total_ids.remove(tmp_id)
                select_ids.append(tmp_id)
            if flag:
                break
        if len(select_ids) >= agent_num:
            break
    print(total_locations.shape)
    print(total_locations[np.array(select_ids)].shape)
    return total_locations[np.array(select_ids)]

def uniform(agent_num, position_range, r_C=0.5):
    L = position_range[0][1] - position_range[0][0]
    print("L:", L)
    total_locations = []
    delta_dis = r_C / 2
    each_num = 4
    for i in range(int(L // delta_dis)):
        for j in range(int(L // delta_dis)):
            one_start_x = position_range[0][0] + i * delta_dis
            one_end_x = one_start_x + delta_dis
            one_start_y = position_range[1][0] + j * delta_dis
            one_end_y = one_start_y + delta_dis
            one_x = np.random.uniform(one_start_x, one_end_x, each_num)
            one_y = np.random.uniform(one_start_y, one_end_y, each_num)
            total_locations.append(np.column_stack((one_x, one_y)))

    total_locations = np.array(total_locations)
    total_locations = np.reshape(total_locations, (-1, 2))
    np.random.seed(19961110)
    np.random.shuffle(total_locations)
    np.random.shuffle(total_locations)
    np.random.shuffle(total_locations)

    total_ids = list(range(total_locations.shape[0]))
    select_ids = [total_ids[0]]
    for _ in range(agent_num):
        print("select:", len(select_ids))
        for tmp_id in total_ids:
            flag = False
            num = 0
            tmp_pos = total_locations[tmp_id]
            for one_sid in range(len(select_ids)):
                between_dis = np.linalg.norm(
                    total_locations[select_ids[one_sid]] -
                    tmp_pos
                )
                if between_dis >= 0.001 and between_dis <= 0.5:
                    num += 1

            if num == 1:
                flag = True
                total_ids.remove(tmp_id)
                select_ids.append(tmp_id)
            if flag:
                break
        if len(select_ids) >= agent_num:
            break
    print(total_locations.shape)
    print(total_locations[np.array(select_ids)].shape)
    return total_locations[np.array(select_ids)]

def taiji(agent_num, position_range):
    num0 = 30

    # 五部分组成
    num_ratios = [0.5, 0.24, 0.24, 0.08, 0.08]
    num1 = int(num_ratios[0] * agent_num - num0)
    num2 = int(num_ratios[1] * agent_num - num0)
    num3 = int(num_ratios[2] * agent_num - num0)
    num45 = agent_num - num0 - num1 - num2 - num3
    flag = False
    if num45 % 2 == 0:
        num4 = int(num45 / 2)
        num5 = int(num45 / 2)
    else:
        flag = True
        PSO = np.array([0, 0])

    if num0 % 2 == 0:  # 偶数
        line_ys1 = np.linspace(position_range[1]-1, 0, num0 // 2)
        line_ys2 = np.linspace(position_range[0]+1, 0, num0 // 2)
        PS00 = np.zeros(shape=(num0 // 2, 2), dtype=np.float32)
        PS01 = np.zeros(shape=(num0 // 2, 2), dtype=np.float32)
        PS00[:, 1] = line_ys1
        PS01[:, 1] = line_ys2
        PS0 = np.vstack((PS00, PS01))
    else:
        num0 = num0 - 1
        line_ys1 = np.linspace(position_range[1]-1, 0, num0 // 2)
        line_ys2 = np.linspace(position_range[0]+1, 0, num0 // 2)
        PS00 = np.zeros(shape=(num0 // 2, 2), dtype=np.float32)
        PS01 = np.zeros(shape=(num0 // 2, 2), dtype=np.float32)
        PS00[:, 1] = line_ys1
        PS01[:, 1] = line_ys2
        PS0 = np.vstack((np.array([[0, 0]]), PS00, PS01))
    # [1] 大圆
    rrr = 0.8  # 系数
    out_r = 5 * rrr
    delta_angle = 360 / num1
    circle_center = [0, 0]
    angles = np.array([delta_angle * i for i in range(num1)])
    radians = angles * np.pi / 180
    x = out_r * np.cos(radians) + circle_center[0]
    y = out_r * np.sin(radians) + circle_center[1]
    PS1 = np.vstack((x, y)).T
    # [2] 上小半圆
    upper_r = 5 * rrr / 2
    delta_angle = 180 / num2
    circle_center = [0, upper_r]
    angles = np.array([delta_angle * i for i in np.arange(-num2 / 2, num2 / 2)])
    radians = angles * np.pi / 180
    x = upper_r * np.cos(radians) + circle_center[0]
    y = upper_r * np.sin(radians) + circle_center[1]
    PS2 = np.vstack((x, y)).T

    # [3] 下小半圆
    down_r = 5 * rrr / 2
    delta_angle = 180 / num3
    circle_center = [0, -down_r]
    angles = np.array([delta_angle * i for i in np.arange(num2 / 2, 3 * num2 / 2)])
    radians = angles * np.pi / 180
    x = down_r * np.cos(radians) + circle_center[0]
    y = down_r * np.sin(radians) + circle_center[1]
    PS3 = np.vstack((x, y)).T

    point_r = upper_r / 4
    # [4] 上点
    delta_angle = 360 / num4
    circle_center = [0, upper_r]
    angles = np.array([delta_angle * i for i in range(num4)])
    radians = angles * np.pi / 180
    x = point_r * np.cos(radians) + circle_center[0]
    y = point_r * np.sin(radians) + circle_center[1]
    PS4 = np.vstack((x, y)).T

    # [5] 下点
    delta_angle = 360 / num5
    circle_center = [0, -down_r]
    angles = np.array([delta_angle * i for i in range(num5)])
    radians = angles * np.pi / 180
    x = point_r * np.cos(radians) + circle_center[0]
    y = point_r * np.sin(radians) + circle_center[1]
    PS5 = np.vstack((x, y)).T

    if flag:
        return np.vstack((PSO, PS0, PS1, PS2, PS3, PS4, PS5))
    else:
        return np.vstack((PS0, PS1, PS2, PS3, PS4, PS5))
