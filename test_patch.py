import re
with open('stabilized_hyperelasticity.py', 'r') as f:
    text = f.read()

text = text.replace('# P = P - self.p*H', 'P = P - self.p*H')
text = text.replace('Res_u = inner(P, grad(self.v))*dx #+ tau*inner(H, grad(self.v))*R_p*dx', 'Res_u = inner(P, grad(self.v))*dx + tau*inner(H, grad(self.v))*R_p*dx')
text = text.replace('Res_p = self.q*(J - 1)*dx #- tau*inner(R_u, H*grad(self.q))*dx', 'Res_p = self.q*(J - 1)*dx - tau*inner(R_u, H*grad(self.q))*dx')

with open('stabilized_hyperelasticity_patched.py', 'w') as f:
    f.write(text)
