import numpy as np
import matplotlib.pyplot as plt

# 读取 CSV（跳过表头）
data1 = np.loadtxt('results/uy_u2_p1_noStab.csv', 
                   delimiter=',', skiprows=1)
data2 = np.loadtxt('results/uy_u1_p1_GLS.csv', 
                   delimiter=',', skiprows=1)
data3 = np.loadtxt('results/uy_u1_p1_GLS_space.csv', 
                   delimiter=',', skiprows=1)

time1, uy1 = data1[:, 0], data1[:, 1]
time2, uy2 = data2[:, 0], data2[:, 1]
time3, uy3 = data3[:, 0], data3[:, 1]

plt.figure(figsize=(12, 7))
plt.plot(time1, uy1, 'b-', linewidth=2, label='no_stab (P2/P1)')
plt.plot(time2, uy2, 'r--', linewidth=2, label='GLS (P1/P1)')
plt.plot(time3, uy3, 'g-.', linewidth=2, label='GLS (P1/P1) - Space')
plt.xlabel('Time (s)')
plt.ylabel('Vertical Displacement (m)')
plt.title('Vertical Displacement Comparison')
plt.grid(True)
plt.legend()
plt.savefig('comparison.png', dpi=300)
plt.show()