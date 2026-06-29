import numpy as np
import matplotlib.pyplot as plt

# 读取 CSV（跳过表头）：列顺序 time, m, p+PV+PC
data1 = np.loadtxt('monitor_left_bottom.csv', delimiter=',', skiprows=1)
data2 = np.loadtxt('monitor_center.csv',      delimiter=',', skiprows=1)
data3 = np.loadtxt('monitor_top.csv',          delimiter=',', skiprows=1)

t1, m1, ptot1 = data1[:, 0], data1[:, 1], data1[:, 2]
t2, m2, ptot2 = data2[:, 0], data2[:, 1], data2[:, 2]
t3, m3, ptot3 = data3[:, 0], data3[:, 1], data3[:, 2]

# ---- 图 1：流体质量 m ---- 
plt.figure(figsize=(12, 7))
plt.plot(t1, m1, 'b-',  linewidth=2, label='left_bottom')
plt.plot(t2, m2, 'r--', linewidth=2, label='center')
plt.plot(t3, m3, 'g-.', linewidth=2, label='top')
plt.xlabel('Time (s)')
plt.ylabel('Fluid mass m')
plt.title('Fluid mass at monitoring points')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('comparison_m.png', dpi=300)

# ---- 图 2：总压力 p + PV + PC ----
plt.figure(figsize=(12, 7))
plt.plot(t1, ptot1, 'b-',  linewidth=2, label='left_bottom')
plt.plot(t2, ptot2, 'r--', linewidth=2, label='center')
plt.plot(t3, ptot3, 'g-.', linewidth=2, label='top')
plt.xlabel('Time (s)')
plt.ylabel('p + PV + PC')
plt.title('Total pressure (p + PV + PC) at monitoring points')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('comparison_p_total.png', dpi=300)

plt.show()