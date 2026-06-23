import re

with open("stabilized_hyperelasticity.py", "r") as f:
    text = f.read()

# Replace R_u = div(P) with R_u = div(P) - self.rho * a
# And tau = ... with a dynamically consistent tau

old_code = """        # 全应力
        P = diff(self.material.strain_energy(F_v), F_v) - self.p * H
    
        tau = 0.2 * self.h**2 / self.material.mu
        
        R_u = div(P)"""

new_code = """        # 全应力
        P = diff(self.material.strain_energy(F_v), F_v) - self.p * H
    
        c_stat = self.material.mu / (0.2 * self.h**2)
        c_kin = self.rho / dt_val**2
        tau = 1.0 / (c_kin + c_stat)
        
        R_u = div(P) - self.rho * a"""

if old_code in text:
    print("Found old code, replacing...")
    text = text.replace(old_code, new_code)
else:
    print("Old code not found. Trying regex.")
    
with open("stabilized_hyperelasticity.py", "w") as f:
    f.write(text)
