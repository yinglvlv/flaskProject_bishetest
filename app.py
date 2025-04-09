import pandas as pd
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# 加载数据
def load_data():
    circuits = pd.read_csv('csvpack/circuits.csv')
    constructor_results = pd.read_csv('csvpack/constructor_results.csv')
    constructor_standings = pd.read_csv('csvpack/constructor_standings.csv')
    constructors = pd.read_csv('csvpack/constructors.csv')
    driver_standings = pd.read_csv('csvpack/driver_standings.csv')
    drivers = pd.read_csv('csvpack/drivers.csv')
    lap_times = pd.read_csv('csvpack/lap_times.csv')
    pit_stops = pd.read_csv('csvpack/pit_stops.csv')
    qualifying = pd.read_csv('csvpack/qualifying.csv')
    races = pd.read_csv('csvpack/races.csv')
    results = pd.read_csv('csvpack/results.csv')
    seasons = pd.read_csv('csvpack/seasons.csv')
    sprint_results = pd.read_csv('csvpack/sprint_results.csv')
    status = pd.read_csv('csvpack/status.csv')
    return (circuits, constructor_results, constructor_standings, constructors,
            driver_standings, drivers, lap_times, pit_stops, qualifying,
            races, results, seasons, sprint_results, status)

data = load_data()
(circuits, constructor_results, constructor_standings, constructors,
 driver_standings, drivers, lap_times, pit_stops, qualifying,
 races, results, seasons, sprint_results, status) = data

# 获取某年份的比赛日历
def get_calendar(year):
    year_races = races[races['year'] == year].merge(circuits[['circuitId', 'country']], on='circuitId')
    year_races['date'] = pd.to_datetime(year_races['date'])
    return year_races.sort_values('date')

# 新增函数：获取详细的比赛日历
def get_detailed_calendar(year):
    detailed_races = races[races['year'] == year].merge(
        circuits[['circuitId', 'country', 'name', 'location', 'lat', 'lng']],
        on='circuitId'
    )
    detailed_races['date'] = pd.to_datetime(detailed_races['date'])
    detailed_races['time'] = detailed_races['time'].fillna('待定')
    detailed_races = detailed_races.rename(columns={
        'name_x': 'race_name',
        'name_y': 'circuit_name'
    })
    return detailed_races.sort_values('date').to_dict(orient='records')

# 车手积分榜
def prepare_driver_table(year):
    year_races = races[races['year'] == year]
    race_ids = year_races['raceId'].tolist()
    year_results = results[results['raceId'].isin(race_ids)].merge(drivers[['driverId', 'forename', 'surname', 'code']], on='driverId')
    year_sprint = sprint_results[sprint_results['raceId'].isin(race_ids)].merge(drivers[['driverId', 'forename', 'surname', 'code']], on='driverId')
    year_results = pd.concat([year_results[['raceId', 'driverId', 'points']], year_sprint[['raceId', 'driverId', 'points']]])
    year_races = year_races.merge(circuits[['circuitId', 'country']], on='circuitId')
    driver_mapping = dict(zip(drivers['driverId'], drivers['forename'] + ' ' + drivers['surname']))

    race_mapping = dict(zip(year_races['raceId'], zip(year_races['name'], year_races['country'])))
    sorted_races = sorted(race_mapping.items(), key=lambda x: year_races[year_races['raceId'] == x[0]]['date'].iloc[0])
    race_names = [race[1][0] for race in sorted_races]
    countries = [race[1][1] for race in sorted_races]
    flag_paths = [f"F1_Flags_2021-2025/{country}.png" for country in countries]

    driver_scores = {}
    for _, row in year_results.iterrows():
        driver_name = driver_mapping.get(row['driverId'])
        race_id = row['raceId']
        race_name = race_mapping.get(race_id, [None])[0]  # 获取 raceName
        points = row['points']
        if driver_name and race_name:
            if driver_name not in driver_scores:
                driver_scores[driver_name] = {race: 0 for race in race_names}
            driver_scores[driver_name][race_name] = driver_scores[driver_name].get(race_name, 0) + points

    driver_total_points = {driver: sum(scores.values()) for driver, scores in driver_scores.items()}
    driver_data = [[driver] + [scores.get(race, 0) for race in race_names] + [driver_total_points[driver]]
                   for driver, scores in driver_scores.items()]
    driver_table = pd.DataFrame(driver_data, columns=['Driver'] + race_names + ['Total Points'])
    return driver_table.sort_values('Total Points', ascending=False), dict(zip(race_names, flag_paths))

# 车队积分榜
def get_constructor_standings(year):
    year_races = races[races['year'] == year]['raceId'].tolist()
    year_results = results[results['raceId'].isin(year_races)].groupby('constructorId')['points'].sum().reset_index()
    year_sprint = sprint_results[sprint_results['raceId'].isin(year_races)].groupby('constructorId')['points'].sum().reset_index()
    year_total = year_results.merge(year_sprint, on='constructorId', how='outer', suffixes=('_race', '_sprint')).fillna(0)
    year_total['points'] = year_total['points_race'] + year_total['points_sprint']
    constructor_mapping = dict(zip(constructors['constructorId'], constructors['name']))
    year_total['Constructor'] = year_total['constructorId'].map(constructor_mapping)
    return year_total[['Constructor', 'points']].sort_values('points', ascending=False)

# 比赛详情
def get_race_details(race_id):
    race_info = races[races['raceId'] == race_id].merge(circuits[['circuitId', 'country']], on='circuitId').iloc[0]
    # 合并正赛和冲刺赛结果，直接使用 results 的 position 和 grid
    race_results = results[results['raceId'] == race_id].merge(drivers[['driverId', 'forename', 'surname', 'code']], on='driverId')
    race_sprint = sprint_results[sprint_results['raceId'] == race_id].merge(drivers[['driverId', 'forename', 'surname', 'code']], on='driverId')
    race_results = pd.concat([race_results, race_sprint]).fillna({'number': '\\N', 'grid': '\\N', 'position': '\\N'})
    race_results['driver_name'] = race_results['forename'] + ' ' + race_results['surname']
    # 获取该场比赛的圈次数据和进站数据
    print(lap_times)
    lap_data = lap_times[lap_times['raceId'] == race_id][['driverId', 'lap', 'position']]
    pit_data = pit_stops[pit_stops['raceId'] == race_id][['driverId', 'lap']].drop_duplicates()
    return race_info, race_results, lap_data, pit_data

# 车手积分趋势，包括冲刺赛积分
def get_driver_points_trend(year):
    year_races = races[races['year'] == year].sort_values('date')
    race_ids = year_races['raceId'].tolist()
    # 合并正赛和冲刺赛结果，保留 constructorId
    results_year = results[results['raceId'].isin(race_ids)].merge(drivers[['driverId', 'forename', 'surname', 'code']], on='driverId')
    sprint_year = sprint_results[sprint_results['raceId'].isin(race_ids)].merge(drivers[['driverId', 'forename', 'surname', 'code']], on='driverId')
    results_year = pd.concat([results_year, sprint_year])  # 先合并完整数据
    # 保留 constructorId 并计算总积分
    results_year = results_year.groupby(['raceId', 'driverId', 'forename', 'surname', 'constructorId', 'code'])['points'].sum().reset_index()
    print("results_year columns:", results_year.columns)  # 调试：打印列名

    # 车队颜色映射（基于 constructorId）
    team_colors = {
        1: '#fb8713',    # McLaren
        6: '#f21c24',    # Ferrari
        131: '#91f2d5',  # Alpine
        9: '#667cf0',    # Red Bull
        3: '#062bef',    # Mercedes
        214: '#fb51e5',  # AlphaTauri
        210: '#c7c6c7',  # Haas
        117: '#1f420a',  # Aston Martin
        215: '#a1ace8',  # Williams
        15: '#20f33d',   # Alfa Romeo
        51: '#990614',   # Sauber
        213: '#a1ace8'   # Another Williams
    }

    # 按 raceId 排序，确保积分累积按比赛顺序计算
    results_year = results_year.sort_values(['driverId', 'raceId'])
    # 使用 transform 计算每个车手的积分累积，保留所有列
    results_year['cumulative_points'] = results_year.groupby(['driverId'])['points'].cumsum()
    # 重置索引以确保所有列可用
    driver_points = results_year.reset_index(drop=True)
    print("driver_points before merge columns:", driver_points.columns)  # 调试：打印合并前的列名
    # 合并 constructorId，确保 driverId 存在
    if 'driverId' in driver_points.columns:
        driver_points = driver_points.merge(results_year[['driverId', 'constructorId', 'code']].drop_duplicates(), on='driverId', how='left', suffixes=('_original', '_merged'))
        # 统一 constructorId 和 code
        driver_points['constructorId'] = driver_points['constructorId_original'].fillna(driver_points['constructorId_merged'])
        driver_points['code'] = driver_points['code_original'].fillna(driver_points['code_merged'])
        driver_points.drop(columns=['constructorId_original', 'constructorId_merged', 'code_original', 'code_merged'], inplace=True)
    else:
        print("Error: 'driverId' not found in driver_points before merge")
        driver_points = driver_points.merge(results_year[['driverId', 'constructorId', 'code', 'raceId']].drop_duplicates(), on=['driverId', 'raceId'], how='left')
    # 填充缺失的 cumulative_points 为 0（初始值）
    driver_points['cumulative_points'] = driver_points.groupby(['driverId', 'forename', 'surname', 'code'])['cumulative_points'].transform(lambda x: x.fillna(0))
    print("driver_points after merge columns:", driver_points.columns)  # 调试：打印合并后的列名
    if 'constructorId' not in driver_points.columns:
        print("Warning: 'constructorId' not found in driver_points. Using default color.")
        driver_points['team_color'] = '#888888'  # 默认灰色
    else:
        driver_points['team_color'] = driver_points['constructorId'].map(lambda x: team_colors.get(x, '#888888'))  # 映射颜色

    # 每场比赛加分数据（包括冲刺赛）
    race_points = results_year.groupby(['raceId', 'forename', 'surname', 'code'])['points'].sum().reset_index()
    race_points = race_points.merge(year_races[['raceId', 'name']], on='raceId')

    return driver_points.to_dict(orient='records'), year_races['name'].tolist(), race_points.to_dict(orient='records')

# 获取构造函数积分趋势
def get_constructor_points_trend(year):
    # 获取该年的比赛
    year_races = races[races['year'] == year]
    race_results = results[results['raceId'].isin(year_races['raceId'])]

    # 计算每个车队每场比赛的积分
    constructor_points = race_results.groupby(['raceId', 'constructorId'])['points'].sum().reset_index()

    # 关联车队名称和比赛名称
    constructor_points = constructor_points.merge(constructors[['constructorId', 'name']], on='constructorId')
    constructor_points = constructor_points.merge(year_races[['raceId', 'name']], on='raceId')

    # 重命名列并计算累计积分
    constructor_points = constructor_points.rename(columns={'name_x': 'name', 'name_y': 'raceName'})
    constructor_points['cumulative_points'] = constructor_points.groupby('constructorId')['points'].cumsum()

    # 添加车队颜色
    constructor_points['team_color'] = constructor_points['constructorId'].map({
        1: '#fb8713',  # McLaren
        3: '#062bef',  # Williams
        6: '#f21c24',  # Ferrari
        9: '#667cf0',  # Red Bull
        15: '#20f33d', # Sauber
        117: '#1f420a',# Aston Martin
        131: '#91f2d5',# Mercedes
        210: '#c7c6c7',# Haas
        214: '#fb51e5',# Alpine
        215: '#a1ace8' # RB
    }).fillna('#888888')

    # 获取比赛名称列表
    race_names = year_races['name'].tolist()

    return constructor_points[['constructorId', 'name', 'raceName', 'cumulative_points', 'team_color']], race_names




# 添加处理历史圈速差距的函数
def get_history_lap_graph(race_id):
    global lap_times
    print("Before - lap_times columns:", lap_times.columns.tolist())
    # 获取比赛结果、单圈数据和进站数据
    race_results = results[results['raceId'] == race_id].merge(
        drivers[['driverId', 'forename', 'surname', 'code']], on='driverId'
    )
    lap_data = lap_times[lap_times['raceId'] == race_id].merge(
        drivers[['driverId', 'forename', 'surname', 'code']], on='driverId'
    )
    pit_data = pit_stops[pit_stops['raceId'] == race_id][['driverId', 'lap', 'milliseconds']].sort_values(['driverId', 'lap'])

    # 确定最大圈数（基于所有车手的最大圈数）
    max_lap = lap_data['lap'].max() or 1

    # 为每个车手确定最大完成圈数（包括退赛处理）
    driver_max_laps = {}
    for driver_id in lap_data['driverId'].unique():
        driver_result = race_results[race_results['driverId'] == driver_id]
        if not driver_result.empty:
            laps = driver_result['laps'].iloc[0]
            status_id = driver_result['statusId'].iloc[0]
            # 如果状态为非正常完成（例如 DNF）或圈数小于最大圈数，则为退赛
            if pd.isna(laps) or laps < max_lap or status_id not in [1]:  # 假设 1 表示正常完成
                # 查找该车手在 lap_times.csv 中的最大圈数（退赛点）
                max_lap_driver = lap_data[lap_data['driverId'] == driver_id]['lap'].max() or 0
                driver_max_laps[driver_id] = int(max_lap_driver) if max_lap_driver > 0 else 1
            else:
                driver_max_laps[driver_id] = int(laps)  # 正常完成车手的最大圈数
        else:
            # 如果结果中没有该车手，查找 lap_times.csv 中的最大圈数
            max_lap_driver = lap_data[lap_data['driverId'] == driver_id]['lap'].max() or 0
            driver_max_laps[driver_id] = int(max_lap_driver) if max_lap_driver > 0 else 1

    # 车队颜色映射
    team_colors = {
        1: '#fb8713',    # McLaren
        6: '#f21c24',    # Ferrari
        131: '#91f2d5',  # Alpine
        9: '#667cf0',    # Red Bull
        3: '#062bef',    # Mercedes
        214: '#fb51e5',  # AlphaTauri
        210: '#c7c6c7',  # Haas
        117: '#1f420a',  # Aston Martin
        215: '#a1ace8',  # Williams
        15: '#20f33d',   # Alfa Romeo
        51: '#990614',   # Sauber
        213: '#a1ace8'   # Another Williams
    }

    # 为每个车手计算每圈的累计用时（包括进站）
    driver_cumulative_times = {}
    for driver_id in lap_data['driverId'].unique():
        driver_laps = lap_data[lap_data['driverId'] == driver_id].sort_values('lap')
        driver_pit_stops = pit_data[pit_data['driverId'] == driver_id]
        cumulative_times = []  # 存储当前车手的每圈累计用时（包括进站）
        cumulative_time = 0
        driver_max_lap = driver_max_laps[driver_id]

        for lap in range(1, max_lap + 1):
            if lap > driver_max_lap:
                cumulative_times.append(None)  # 退赛后不再计算，保持为 None
                continue

            # 添加单圈用时
            lap_row = driver_laps[driver_laps['lap'] == lap]
            if not lap_row.empty:
                time_ms = lap_row['milliseconds'].iloc[0]
                try:
                    time_ms = float(time_ms)
                    if 50 <= time_ms <= 200000:  # 限制单圈用时在 50 毫秒至 200 秒之间
                        cumulative_time += time_ms / 1000  # 转换为秒
                    else:
                        cumulative_time += 0  # 跳过异常值
                except (ValueError, TypeError):
                    cumulative_time += 0  # 跳过无效数据

            # 添加进站用时（如果有进站）
            pit_stop_row = driver_pit_stops[driver_pit_stops['lap'] == lap]
            if not pit_stop_row.empty:
                pit_time_ms = pit_stop_row['milliseconds'].iloc[0]
                try:
                    pit_time_ms = float(pit_time_ms)
                    if 0 <= pit_time_ms <= 1000000:  # 限制进站用时在 0 至 1000 秒之间
                        cumulative_time += pit_time_ms / 1000  # 转换为秒
                    else:
                        cumulative_time += 0  # 跳过异常值
                except (ValueError, TypeError):
                    cumulative_time += 0  # 跳过无效数据

            cumulative_times.append(cumulative_time)

        driver_cumulative_times[driver_id] = cumulative_times

    # 为每一圈动态确定当前排名第一的车手（比赛用时最少）
    first_cumulative_times_per_lap = []
    for lap in range(1, max_lap + 1):
        lap_times_dict = {driver_id: times[lap - 1] for driver_id, times in driver_cumulative_times.items() if
                          times[lap - 1] is not None}
        if lap_times_dict:
            first_driver_id = min(lap_times_dict, key=lambda x: lap_times_dict[x] or float('inf'))
            first_cumulative_times_per_lap.append(lap_times_dict[first_driver_id])
        else:
            first_cumulative_times_per_lap.append(0)

    # 为每个车手计算每圈的差距（正值表示落后）
    driver_data = []
    for driver_id in lap_data['driverId'].unique():
        cumulative_times = driver_cumulative_times[driver_id]
        gaps = []

        driver_max_lap = driver_max_laps[driver_id]  # 每个车手的最大完成圈数

        for lap in range(1, max_lap + 1):
            if lap > driver_max_lap:
                gaps.append(None)  # 退赛后不再计算，保持为 None（Chart.js 会跳过）
                continue

            current_cumulative_time = cumulative_times[lap - 1]
            if current_cumulative_time is not None and lap <= len(first_cumulative_times_per_lap):
                gap = max(0, current_cumulative_time - first_cumulative_times_per_lap[lap - 1]) if first_cumulative_times_per_lap[lap - 1] > 0 else 0
            else:
                gap = None  # 如果数据缺失或圈数超过第一名数据，使用 None
            gaps.append(gap)

        driver_info = drivers[drivers['driverId'] == driver_id].iloc[0]
        constructor_id = race_results[race_results['driverId'] == driver_id]['constructorId'].iloc[0] if driver_id in race_results['driverId'].values else None
        driver_data.append({
            'driver_id': int(driver_id),  # 确保为 Python int
            'name': f"{driver_info['forename']} {driver_info['surname']} ({driver_info['code']})",
            'gaps': gaps,
            'team_color': team_colors.get(constructor_id, '#888888')  # 默认灰色
        })

    # 转换为 DataFrame 并使用 to_dict
    driver_lap_gaps_df = pd.DataFrame(driver_data)
    driver_lap_gaps = driver_lap_gaps_df.to_dict(orient='records')

    # 准备进站注解，使用 DataFrame 和 to_dict
    pit_annotations_df = pit_data.merge(drivers[['driverId', 'forename', 'surname', 'code']], on='driverId')
    pit_annotations = pit_annotations_df.apply(lambda row: {
        'driverId': int(row['driverId']),  # 转换为 Python int
        'lap': row['lap'],
        'name': f"{row['forename']} {row['surname']} ({row['code']})"
    }, axis=1).tolist()

    # 动态计算最大差距，用于前端调整 Y 轴范围
    all_gaps = []
    for driver in driver_lap_gaps:
        all_gaps.extend([g for g in driver['gaps'] if g is not None])
    max_gap = max(all_gaps) if all_gaps else 0  # 最大差距（正值表示落后）

    print("After - lap_times columns:", lap_times.columns.tolist())
    return driver_lap_gaps, max_lap, pit_annotations, max_gap  # 返回 4 个值

# 添加处理发车网格的函数（框架）
def get_race_grid(race_id):
    # Load base data from results.csv, keeping its 'number' column
    race_results = results[results['raceId'] == race_id].merge(
        drivers[['driverId', 'forename', 'surname', 'code']],  # Exclude 'number' from drivers.csv
        on='driverId'
    ).merge(
        constructors[['constructorId', 'name']],
        on='constructorId'
    )

    # Merge qualifying times (Q1, Q2, Q3) from qualifying.csv
    quali_data = qualifying[qualifying['raceId'] == race_id][['driverId', 'q1', 'q2', 'q3']]
    race_results = race_results.merge(quali_data, on='driverId', how='left')

    # Combine forename and surname into a full name
    race_results['driver_name'] = race_results['forename'] + ' ' + race_results['surname']

    # Define team colors for visualization
    team_colors = {
        1: '#fb8713',    # McLaren
        6: '#f21c24',    # Ferrari
        131: '#91f2d5',  # Alpine
        9: '#667cf0',    # Red Bull
        3: '#062bef',    # Mercedes
        214: '#fb51e5',  # AlphaTauri
        210: '#c7c6c7',  # Haas
        117: '#1f420a',  # Aston Martin
        215: '#a1ace8',  # Williams
        15: '#20f33d',   # Alfa Romeo
        51: '#990614',   # Sauber
        213: '#a1ace8'   # Another Williams
    }
    race_results['team_color'] = race_results['constructorId'].map(team_colors)

    # Select the best qualifying time (Q3 > Q2 > Q1)
    def select_best_time(row):
        if pd.notna(row['q3']) and row['q3'] != '\\N':
            return row['q3']
        elif pd.notna(row['q2']) and row['q2'] != '\\N':
            return row['q2']
        elif pd.notna(row['q1']) and row['q1'] != '\\N':
            return row['q1']
        return 'N/A'
    race_results['best_time'] = race_results.apply(select_best_time, axis=1)

    # Select and rename columns, using 'number' from results.csv
    grid_data = race_results[['grid', 'code', 'number', 'name', 'best_time', 'team_color']].sort_values('grid').rename(columns={'name': 'team'})

    # Filter out invalid grid positions
    grid_data = grid_data[grid_data['grid'].notna() & (grid_data['grid'] != 0) & (grid_data['grid'] != '\\N')]

    # Format the best time into mm:ss.sss
    def format_time(time_str):
        if time_str == 'N/A' or pd.isna(time_str) or time_str == '\\N':
            return 'N/A'
        try:
            # Convert milliseconds to mm:ss.sss if numeric
            ms = float(time_str)
            minutes = int(ms // 60000)
            seconds = int((ms % 60000) // 1000)
            milliseconds = int(ms % 1000)
            return f"{minutes}:{seconds:02d}.{milliseconds:03d}"
        except ValueError:
            # Return as-is if already in mm:ss.sss format
            return time_str
    grid_data['best_time'] = grid_data['best_time'].apply(format_time)

    return grid_data

# 辅助函数：将毫秒格式化为 MM:SS.SSS
def format_time(ms):
    if pd.isna(ms) or ms is None:
        return 'N/A'
    minutes = int(ms // 60000)
    seconds = int((ms % 60000) // 1000)
    milliseconds = int(ms % 1000)
    return f"{minutes}:{seconds:02d}.{milliseconds:03d}"

# 比较两名车手的圈速
def get_lap_time_comparison(race_id, driver_a_id, driver_b_id):
    # 过滤指定比赛和车手的圈速数据
    lap_data = lap_times[lap_times['raceId'] == race_id]
    lap_data_a = lap_data[lap_data['driverId'] == driver_a_id].sort_values('lap')
    lap_data_b = lap_data[lap_data['driverId'] == driver_b_id].sort_values('lap')
    pit_data = pit_stops[pit_stops['raceId'] == race_id]

    max_lap_a = lap_data_a['lap'].max() if not lap_data_a.empty else 0
    max_lap_b = lap_data_b['lap'].max() if not lap_data_b.empty else 0
    max_lap = max(max_lap_a, max_lap_b)

    comparison_table = []
    cumulative_gap = 0
    cumulative_time_a = 0
    cumulative_time_b = 0
    for lap in range(1, int(max_lap) + 1):
        time_a = lap_data_a[lap_data_a['lap'] == lap]['milliseconds'].values[0] if lap <= max_lap_a else None
        time_b = lap_data_b[lap_data_b['lap'] == lap]['milliseconds'].values[0] if lap <= max_lap_b else None
        if time_a:
            cumulative_time_a += time_a
        if time_b:
            cumulative_time_b += time_b
        gap = (time_a - time_b) / 1000.0 if time_a and time_b else None
        if gap:
            cumulative_gap += gap
        comparison_table.append({
            'lap': lap,
            'time_a': format_time(time_a),
            'time_b': format_time(time_b),
            'gap': f"{gap:.3f}" if gap else 'N/A',
            'cumulative_gap': f"{cumulative_gap:.3f}" if cumulative_gap != 0 else '0.000',
            'driver_a_faster': time_a < time_b if time_a and time_b else None,
            'driver_b_faster': time_b < time_a if time_a and time_b else None
        })

    # 获取车手姓名和车队 ID
    driver_a_row = drivers[drivers['driverId'] == driver_a_id]
    driver_b_row = drivers[drivers['driverId'] == driver_b_id]
    driver_a_name = driver_a_row['forename'].values[0] + ' ' + driver_a_row['surname'].values[0]
    driver_b_name = driver_b_row['forename'].values[0] + ' ' + driver_b_row['surname'].values[0]
    constructor_a_id = results[(results['raceId'] == race_id) & (results['driverId'] == driver_a_id)]['constructorId'].values[0]
    constructor_b_id = results[(results['raceId'] == race_id) & (results['driverId'] == driver_b_id)]['constructorId'].values[0]

    # 获取车队颜色
    team_colors = {
        1: '#fb8713',    # McLaren
        6: '#f21c24',    # Ferrari
        131: '#91f2d5',  # Alpine
        9: '#667cf0',    # Red Bull
        3: '#062bef',    # Mercedes
        214: '#fb51e5',  # AlphaTauri
        210: '#c7c6c7',  # Haas
        117: '#1f420a',  # Aston Martin
        215: '#a1ace8',  # Williams
        15: '#20f33d',   # Alfa Romeo
        51: '#990614',   # Sauber
        213: '#a1ace8'   # Another Williams
    }
    team_color_a = team_colors.get(constructor_a_id, '#888888')
    team_color_b = team_colors.get(constructor_b_id, '#888888')

    # 准备总结统计
    summary = {
        'driver_a': driver_a_name,
        'driver_b': driver_b_name,
        'total_time_a': format_time(cumulative_time_a),
        'total_time_b': format_time(cumulative_time_b),
        'fastest_lap_a': format_time(lap_data_a['milliseconds'].min()) if not lap_data_a.empty else 'N/A',
        'fastest_lap_b': format_time(lap_data_b['milliseconds'].min()) if not lap_data_b.empty else 'N/A',
        'slowest_lap_a': format_time(lap_data_a['milliseconds'].max()) if not lap_data_a.empty else 'N/A',
        'slowest_lap_b': format_time(lap_data_b['milliseconds'].max()) if not lap_data_b.empty else 'N/A',
        'pit_stops_a': len(pit_data[pit_data['driverId'] == driver_a_id]),
        'pit_stops_b': len(pit_data[pit_data['driverId'] == driver_b_id]),
        'total_pit_time_a': format_time(pit_data[pit_data['driverId'] == driver_a_id]['milliseconds'].sum()),
        'total_pit_time_b': format_time(pit_data[pit_data['driverId'] == driver_b_id]['milliseconds'].sum()),
        'team_color_a': team_color_a,
        'team_color_b': team_color_b
    }

    # 准备图表数据
    chart_data = {
        'labels': [str(row['lap']) for row in comparison_table],
        'driver_a': [0] * len(comparison_table),  # 第一车手的参考线（零）
        'driver_b': [-float(row['cumulative_gap']) for row in comparison_table]  # 第二车手的相对差距
    }

    return comparison_table, summary, chart_data

# 添加处理进站总结的函数（框架）
def get_pit_stop_summary(race_id):
    pit_data = pit_stops[pit_stops['raceId'] == race_id].merge(
        drivers[['driverId', 'forename', 'surname', 'code']], on='driverId'
    )
    pit_data['driver_name'] = pit_data['forename'] + ' ' + pit_data['surname']
    pit_summary = pit_data.groupby(['driverId', 'driver_name', 'code']).agg({
        'stop': 'count',
        'lap': lambda x: list(x),
        'duration': ['sum', 'mean']
    }).reset_index()
    pit_summary.columns = ['driverId', 'driver_name', 'code', 'pit_stops', 'lap_numbers', 'total_duration', 'avg_duration']
    return pit_summary

# 路由
@app.route('/')
def index():
    years = sorted(races['year'].unique(), reverse=True)
    return render_template('index.html', years=years)

@app.route('/year/<int:year>')
def year_page(year):
    calendar = get_calendar(year)
    driver_table, race_flag_pairs = prepare_driver_table(year)
    driver_points, race_names, race_points = get_driver_points_trend(year)
    return render_template('year_page.html', year=year, races=calendar, driver_table=driver_table,
                           race_flag_pairs=race_flag_pairs, driver_points=driver_points, race_names=race_names,
                           race_points=race_points)

@app.route('/year/<int:year>/drivers')
def driver_standings_page(year):
    calendar = get_calendar(year)
    driver_table, race_flag_pairs = prepare_driver_table(year)
    driver_points, race_names, race_points = get_driver_points_trend(year)
    return render_template('driver_standings.html', year=year, races=calendar, driver_table=driver_table,
                           race_flag_pairs=race_flag_pairs, driver_points=driver_points, race_names=race_names,
                           race_points=race_points)


@app.route('/year/<int:year>/constructors')
def constructor_standings_page(year):
    calendar = get_calendar(year)
    constructors = get_constructor_standings(year)
    constructor_points, race_names = get_constructor_points_trend(year)  # 正确解包返回值

    # 将 constructor_points 从 DataFrame 转换为字典列表
    constructor_points_json = constructor_points.to_dict(orient='records')

    return render_template('constructor_standings.html', year=year, races=calendar, constructors=constructors,
                           constructor_points=constructor_points_json, race_names=race_names)

@app.route('/year/<int:year>/calendar')
def calendar(year):
    calendar = get_calendar(year)  # 用于 base.html
    detailed_calendar = get_detailed_calendar(year)  # 用于 calendar.html
    return render_template('calendar.html', year=year, races=calendar, detailed_races=detailed_calendar)


@app.route('/year/<int:year>/race/<int:race_id>')
def race_detail(year, race_id):
    calendar = get_calendar(year)
    race_info, race_results, lap_data, pit_data = get_race_details(race_id)
    if race_info is None:
        return "比赛未找到", 404

    # 获取 circuitRef
    circuit_ref = circuits[circuits['circuitId'] == race_info['circuitId']]['circuitRef'].iloc[0]

    # 获取该赛道最近 5 场比赛的获胜者
    circuit_id = race_info['circuitId']
    recent_races = races[races['circuitId'] == circuit_id].sort_values(by=['year', 'round'], ascending=False).head(5)

    recent_winners = []
    for race in recent_races.itertuples():
        winner_result = results[(results['raceId'] == race.raceId) & (results['positionOrder'] == 1)]
        if not winner_result.empty:
            driver_id = winner_result['driverId'].iloc[0]
            constructor_id = winner_result['constructorId'].iloc[0]
            driver = drivers[drivers['driverId'] == driver_id]
            constructor = constructors[constructors['constructorId'] == constructor_id]
            recent_winners.append({
                'year': race.year,
                'driver_name': f"{driver['forename'].iloc[0]} {driver['surname'].iloc[0]}" if not driver.empty else "未知",
                'team_name': constructor['name'].iloc[0] if not constructor.empty else "未知"
            })

    return render_template('race_detail.html',
                           year=year,
                           race_id=race_id,
                           race=race_info,
                           race_results=race_results,
                           races=calendar,  # 保持原始逻辑，确保国旗显示
                           constructors=constructors,
                           status=status,
                           lap_data=lap_data,
                           pit_data=pit_data,
                           circuit_ref=circuit_ref,  # 新增赛道图片所需
                           recent_winners=recent_winners)  # 新增最近 5 场比赛获胜者

@app.route('/year/<int:year>/race/<int:race_id>/history_lap')
def history_lap_graph(year, race_id):
    calendar = get_calendar(year)
    race_info = races[races['raceId'] == race_id].merge(circuits[['circuitId', 'country']], on='circuitId').iloc[0]
    driver_lap_gaps, max_lap, pit_annotations, max_gap = get_history_lap_graph(race_id)  # 更新为解包 4 个值
    return render_template('history_lap_graph.html', year=year, race_id=race_id, race=race_info,
                          races=calendar, driver_lap_gaps=driver_lap_gaps, max_lap=max_lap,
                          pit_annotations=pit_annotations, max_gap=max_gap)  # 传递 max_gap

@app.route('/year/<int:year>/race/<int:race_id>/grid')
def race_grid(year, race_id):
    calendar = get_calendar(year)
    race_info = races[races['raceId'] == race_id].merge(circuits[['circuitId', 'country']], on='circuitId').iloc[0]
    grid_data = get_race_grid(race_id)
    return render_template('race_grid.html',
                           year=year,
                           race_id=race_id,
                           race=race_info,
                           races=calendar,
                           grid_data=grid_data)


@app.route('/year/<int:year>/race/<int:race_id>/lap_time', methods=['GET', 'POST'])
def race_lap_time(year, race_id):
    # 获取比赛日历（给 base.html 使用）
    calendar = get_calendar(year)

    # 获取当前比赛信息
    race_info = races[races['raceId'] == race_id].merge(circuits[['circuitId', 'country']], on='circuitId').iloc[0]
    race_results = results[results['raceId'] == race_id].merge(drivers[['driverId', 'forename', 'surname', 'code']],
                                                               on='driverId').merge(
        constructors[['constructorId', 'name']],
        on='constructorId')

    # 添加车队颜色
    team_colors = {
        1: '#fb8713',  # McLaren
        6: '#f21c24',  # Ferrari
        131: '#91f2d5',  # Alpine
        9: '#667cf0',  # Red Bull
        3: '#062bef',  # Mercedes
        214: '#fb51e5',  # AlphaTauri
        210: '#c7c6c7',  # Haas
        117: '#1f420a',  # Aston Martin
        215: '#a1ace8',  # Williams
        15: '#20f33d',  # Alfa Romeo
        51: '#990614',  # Sauber
        213: '#a1ace8'  # Another Williams
    }
    race_results['team_color'] = race_results['constructorId'].map(team_colors).fillna('#FFFFFF')
    race_results = race_results.rename(columns={'name': 'team'})

    drivers_list = race_results[['driverId', 'forename', 'surname', 'code', 'number', 'team', 'team_color']].to_dict(
        'records')

    # 处理车手选择
    if request.method == 'POST' and 'driver_a' in request.form and 'driver_b' in request.form:
        driver_a_id = int(request.form['driver_a'])
        driver_b_id = int(request.form['driver_b'])
    else:
        driver_a_id = drivers_list[0]['driverId']
        driver_b_id = drivers_list[1]['driverId']
    selected_drivers = [driver_a_id, driver_b_id]

    # 获取圈速对比数据
    comparison_table, summary, chart_data = get_lap_time_comparison(race_id, driver_a_id, driver_b_id)

    # 调试：打印当前选择的车手
    print(f"Selected Drivers: {driver_a_id}, {driver_b_id}")

    return render_template('race_lap_time.html',
                           year=year,
                           race_id=race_id,
                           race=race_info,
                           races=calendar,
                           drivers_list=drivers_list,
                           selected_drivers=selected_drivers,
                           comparison_table=comparison_table,
                           summary=summary,
                           chart_data=chart_data)

@app.route('/year/<int:year>/race/<int:race_id>/pit_stop')
def race_pit_stop(year, race_id):
    calendar = get_calendar(year)
    race_info = races[races['raceId'] == race_id].merge(circuits[['circuitId', 'country']], on='circuitId').iloc[0]
    pit_summary = get_pit_stop_summary(race_id)
    return render_template('race_pit_stop.html', year=year, race_id=race_id, race=race_info,
                          races=calendar, pit_summary=pit_summary)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)