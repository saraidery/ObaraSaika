import numpy as np


from obara_saika import OverlapIntegralGTO, NucAttIntegralGTO, KineticIntegralGTO
from obara_saika import OverlapIntegralPWGTO, NucAttIntegralPWGTO

A = np.zeros(3)
B = np.zeros(3)
C = np.zeros(3)
C[0] = 1.0

l_a = 2
l_b = 2

Z = 1.0

A[1] = 1.2
A[2] = 0.7

B[0] = 0.3
B[2] = -0.25

k_a = np.array([0.25, 0.50, 0.75])
k_b = np.array([-0.30, 0.00, -0.60])

alpha = 0.8
beta = 1.1

N = NucAttIntegralGTO(A, alpha, l_a, B, beta, l_b, C, Z)
print(N.integral())

#S = OverlapIntegralGTO(A, alpha, l_a, B, beta, l_b)
#print(S.integral())

#K = KineticIntegralGTO(A, alpha, l_a, B, beta, l_b)
#I = K.integral()
#print(I)

#cS = OverlapIntegralPWGTO(A, alpha, l_a, k_a, B, beta, l_b, k_b)
#print(cS.integral())
#
#cN = NucAttIntegralPWGTO(A, alpha, l_a, k_a, B, beta, l_b, k_b, C, Z)
#print(cN.integral())
#for elm in I.flatten():
#	print(f"{elm:.20}")