import re
with open('stabilized_hyperelasticity.py', 'r') as f:
    text = f.read()

text = text.replace('# P = P - self.p*H', 'P = P - self.material.kappa * (J - 1) * H - self.p * H')
text = text.replace('R_p = J - 1', 'R_p = J - 1 + self.p / self.material.kappa')
text = text.replace('Res_u = inner(P, grad(self.v))*dx #+ tau*inner(H, grad(self.v))*R_p*dx', 'Res_u = inner(P, grad(self.v))*dx + tau*inner(H, grad(self.v))*R_p*dx')
text = text.replace('Res_p = self.q*(J - 1)*dx #- tau*inner(R_u, H*grad(self.q))*dx', 'Res_p = self.q*R_p*dx - tau*inner(R_u, H*grad(self.q))*dx')

with open('stabilized_hyperelasticity_patched3.py', 'w') as f:
    f.write(text)
