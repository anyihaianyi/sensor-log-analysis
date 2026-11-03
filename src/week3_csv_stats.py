import csv

def read_csv(path):
    try:
        with open(path, "r", encoding="UTF-8", newline='') as f:
            return list(csv.reader(f))
    except FileNotFoundError:
        return []

def blank_line(row):        return len(row) == 0
def missing_column(row):    return len(row) < 3        # 1 或 2 列
def multiple_column(row):   return len(row) > 3        # 4 列及以上

def value_invalid(row):
    try:
        float(row[1])
        float(row[2])
        return False
    except ValueError:
        return True

def clean(rows):
    error_count = 0
    right_count = 0
    good = []
    for i,row in enumerate(rows, start=2):
        if blank_line(row):
            print(f"跳过第{i}行，错误类型是空行")
            error_count += 1
            continue

        if missing_column(row):
            print(f"跳过第{i}行，错误类型是某行缺列")
            error_count += 1
            continue

        if multiple_column(row):
            print(f"跳过第{i}行，错误类型是某行多列")
            error_count += 1
            continue            
        
        if value_invalid(row) == True:
            print(f"跳过第{i}行，错误类型是某行值错误")
            error_count += 1
            continue

        good.append((i, float(row[1]), float(row[2]))) # append只能是加一个元素，这里相当于一个元组
        right_count += 1
    return good,right_count,error_count

"""
def stats(values):
    total = sum(values)

    values_count = sum(1 for x in values)
    mean = total / values_count

    m_a_x = max(values)
    m_i_n = min(values)

    max_index = 0
    min_index = 0

    return mean, m_a_x, max_index, m_i_n, min_index
"""

def stats(pairs):               # pairs = [(值, 行号), (值, 行号), ...]
    
    if not pairs:               # 判断是否为空列表
        return None, None, None, None, None 
                 
    total = 0
    for v, line in pairs:      # 解包
        total += v
    mean = total / len(pairs)

    max_v, max_line = max(pairs)      # 一次拿到两个
    min_v, min_line = min(pairs)

    """ 找出所有最大值所在的行号
    第一遍：先求最大值      m = max(pairs)[0]
    第二遍：再找所有等于它的  lines = [line for v, line in pairs if v == m]
    """

    return mean, max_v, max_line, min_v, min_line

def main(file):
    rows = read_csv(file)
    if not rows:
        return
    header = rows[0]
    rows = rows[1:]

    good, right_count, error_count = clean(rows)
    total = right_count + error_count
  
    column_temp = []
    column_humdity = []

    """
    for row in rows:
        raw_column_temp.append(row[1])
        raw_column_humdity.append(row[2])
    """

    for row in good:
        column_temp.append((row[1], row[0]))
        column_humdity.append((row[2], row[0]))

    """ 简便写法
    column_temp = [(t, line) for line, t, h in good]     # 温度在前，行号在后
    column_humdity  = [(h, line) for line, t, h in good]
    """

    t_mean, t_max, t_max_index, t_min, t_min_index = stats(column_temp)
    h_mean, h_max, h_max_index, h_min, h_min_index = stats(column_humdity)

    print(f"读取{total}，有效{right_count}行，跳过{error_count}行")
    print(f"温度：均值{round(t_mean, 2)} 最高{t_max}（第{t_max_index}行） 最低{t_min}（第{t_min_index}行）")
    print(f"湿度：均值{round(h_mean, 2)} 最高{h_max}（第{h_max_index}行） 最低{h_min}（第{h_min_index}行）")

if __name__ == "__main__":
    file = "C:/Users/lenovo/Desktop/具身智能/code/data/sensor_log.csv"
    main(file)