import numpy as np
import matplotlib.pyplot as plt

# 读取 CSV（跳过表头）
data1 = np.loadtxt('results/BDF2_uy_v2_p1_BDF2(noStab).csv', 
                   delimiter=',', skiprows=1)
# data2 = np.loadtxt('results/uy_u2_p1_noStab.csv', 
#                    delimiter=',', skiprows=1)
data3 = np.loadtxt('results/BDF2_uy_v1_p1_BDF2(Stab).csv', 
                   delimiter=',', skiprows=1)
data4 = np.loadtxt('results/uy_v1_p1_GLS_consistentno.csv', 
                   delimiter=',', skiprows=1)
data5 = np.loadtxt('results/BDF2_uy_v1_p1_GLS_BDF2.csv',
                   delimiter=',', skiprows=1)

time1, uy1 = data1[:, 0], data1[:, 1]
# time2, uy2 = data2[:, 0], data2[:, 1]
time3, uy3 = data3[:, 0], data3[:, 1]
time4, uy4 = data4[:, 0], data4[:, 1]
time5, uy5 = data5[:, 0], data5[:, 1]

plt.figure(figsize=(12, 7))
plt.plot(time1, uy1, 'b-', linewidth=2, label='BDF2(noStab)')
# plt.plot(time2, uy2, 'r--', linewidth=2, label='noStab')
plt.plot(time3, uy3, 'g-.', linewidth=2, label='BDF2(Stab)')
# plt.plot(time4, uy4, 'm-', linewidth=2, label='GLS_consistentno')
plt.plot(time5, uy5, 'c-', linewidth=2, label='GLS_BDF2')
plt.xlabel('Time (s)')
plt.ylabel('Vertical Displacement (m)')
plt.title('Vertical Displacement Comparison')
plt.grid(True)
plt.legend()
plt.savefig('comparison.png', dpi=300)
plt.show()