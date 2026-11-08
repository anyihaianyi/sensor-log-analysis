import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
from week3_csv_stats import read_csv, clean

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "sensor_log.csv")
FIG_PATH  = os.path.join(BASE_DIR, "figs", "temp_humidity.png")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]   # 中文别变方块
plt.rcParams["axes.unicode_minus"] = False              # 负号别变方块

# 读取数据
rows = read_csv(DATA_PATH)
header = rows[0]
rows = rows[1:]
good,right_count,error_count = clean(rows)
arr = np.array(good)

# 获取时间
t = []
for element in arr[:, 0]:
    t.append(rows[int(element)-2][0])

t = pd.to_datetime(t)

# 准备数据
y_tem = arr[:,1]
y_hum = arr[:,2]

tem_mean = np.mean(arr.mean(axis=0)[1])
hum_mean = np.mean(arr.mean(axis=0)[2])

# 创建画布
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(11, 6), sharex=True)

# 画出两个折线
ax[0].plot(t, y_tem, color="blue", label="温度")
ax[1].plot(t, y_hum, color="m", linestyle="-.", label="湿度")

ax[0].axhline(tem_mean, color="red", linestyle="--", label="温度均值")
ax[1].axhline(hum_mean, color="red", linestyle="--", label="湿度均值")

# 创建x轴的刻度，及x轴、y轴的刻度范围
y_ticks_tem = range(15, 35)
y_ticks_hum = range(35, 85)

ax[0].xaxis.set_major_locator(mdates.MinuteLocator(interval=20))
ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax[0].set_yticks(y_ticks_tem[::2])

ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax[1].set_yticks(y_ticks_hum[::5])    # 时间轴不用手动设刻度。​ Locator 管“多久一个刻度”，Formatter 管“刻度长什么样”，这两个配好就够了。

# 标题，及x轴、y轴标签
ax[0].set_title("传感器温度变化（2026-10-20）", fontsize=20)
ax[0].set_ylabel("温度 / ℃", fontsize=12) 

ax[1].set_title("传感器湿度变化（2026-10-20）", fontsize=20)
ax[1].set_xlabel("时间", fontsize=12)
ax[1].set_ylabel("湿度 / %", fontsize=12) 

fig.tight_layout()
# 旋转时间标签
plt.xticks(rotation=45, ha='right')

# 添加网格
ax[0].grid(linestyle="--", alpha=0.5)
ax[1].grid(linestyle="--", alpha=0.5)

# 添加图例
ax[0].legend(loc="best")
ax[1].legend(loc="best")

# 保存图像
fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight")

# 完成绘图
plt.show()

mat = arr[:, 1:3]                 # 只取温湿度两列 → (40, 2)
print(mat.mean(axis=0))           # [24.54  61.0875]
print(mat.mean(axis=0).shape)     # (2,)   ← 每列一个
print(mat.mean(axis=1).shape)     # (40,)  ← 每行一个
