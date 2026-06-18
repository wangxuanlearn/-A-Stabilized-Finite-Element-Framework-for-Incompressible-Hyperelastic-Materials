import re
with open('stabilized_hyperelasticity.py', 'r') as f:
    text = f.read()

# Modify NeoHookean
text = re.sub(
r"class NeoHookean:\n    def __init__\(self, mu, kappa\):\n        self.mu = mu\n        self.kappa = kappa\n\n    def strain_energy\(self, F\):\n        C = F\.T\*F\n        J = det\(F\)\n        return self\.mu/2\*\((tr\(C\) - 3\)|J\*\*\(-2/3\)\*tr\(C\) - 3\) \+ self\.kappa/2\*\(\J-1\)\*\*2",
r"class NeoHookean:\n    def __init__(self, mu):\n        self.mu = mu\n\n    def strain_energy(self, F):\n        C = F.T*F\n        J = det(F)\n        # 对于完全不可压，我们使用修正不变量保证等体积\n        return self.mu/2*(J**(-2/3)*tr(C) - 3)",
text
)

text = text.replace("""class NeoHookean:
    def __init__(self, mu, kappa):
        self.mu = mu
        self.kappa = kappa

    def strain_energy(self, F):
        C = F.T*F
        J = det(F)
        return self.mu/2*(tr(C) - 3) + self.kappa/2*(J-1)**2""", """class NeoHookean:
    def __init__(self, mu):
        self.mu = mu

    def strain_energy(self, F):
        C = F.T*F
        J = det(F)
        return self.mu/2*(J**(-2/3)*tr(C) - 3)""")

text = text.replace("""class MooneyRivlin:
    def __init__(self, C10, C01, kappa):
        self.C10 = C10
        self.C01 = C01
        self.kappa = kappa

    def strain_energy(self, F):
        C = F.T*F
        J = det(F)
        I1_bar = J**(-2/3)*tr(C)
        I2_bar = J**(-4/3)*0.5*(tr(C)**2 - tr(C*C))
        return self.C10*(I1_bar - 3) + self.C01*(I2_bar - 3) + self.kappa/2*(J-1)**2""", """class MooneyRivlin:
    def __init__(self, C10, C01):
        self.C10 = C10
        self.C01 = C01

    def strain_energy(self, F):
        C = F.T*F
        J = det(F)
        I1_bar = J**(-2/3)*tr(C)
        I2_bar = J**(-4/3)*0.5*(tr(C)**2 - tr(C*C))
        return self.C10*(I1_bar - 3) + self.C01*(I2_bar - 3)""")


# Modify Solver
text = text.replace("""        # Material stress
        P = diff(self.material.strain_energy(F_v), F_v)
        # 移除原有的体积部分，替换为混合变量 p 的贡献
        # p 在这里定义为正压力 (对应于 -kappa*(J-1))
        P = P - self.material.kappa * (J - 1) * H - self.p * H
    
        # Stabilization parameter
        tau = 0.1 * self.h**2 / self.material.mu
        
        # Residuals (Eq. 11-12)
        R_u = div(P)  # Simplified, no body force
        R_p = J - 1 + self.p / self.material.kappa""", """        # Material stress
        # 完全不可压缩情况：应力 = 偏应力部分 - p * H (由应变能自动求导得到偏应力)
        P = diff(self.material.strain_energy(F_v), F_v) - self.p * H
    
        # Stabilization parameter
        tau = 0.1 * self.h**2 / self.material.mu
        
        # Residuals (Eq. 11-12)
        R_u = div(P)  # Simplified, no body force
        R_p = J - 1  # 完全不可压缩约束""")

with open('stabilized_hyperelasticity.py', 'w') as f:
    f.write(text)
