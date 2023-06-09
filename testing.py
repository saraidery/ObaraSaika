import numpy as np


from obara_saika import OverlapIntegralGTO, NucAttIntegralGTO
from obara_saika import OverlapIntegralPWGTO, NucAttIntegralPWGTO

A = np.zeros(3)
B = np.zeros(3)
C = np.zeros(3)
C[0] = 1.0

l_a = 2
l_b = 2

alpha = 1.0
beta = 1.0
Z = 1.0

N = NucAttIntegralGTO(A, alpha, l_a, B, beta, l_b, C, Z)
#print(N.integral())

S = OverlapIntegralGTO(A, alpha, l_a, B, beta, l_b)
#print(S.integral())


A[1] = 1.2
A[2] = 0.7

B[0] = 0.3
B[2] = -0.25

k_a = np.array([0.25, 0.50, 0.75])
k_b = np.array([-0.30, 0.00, -0.60]) # OBS complex conj -> check all places

l_a = 1
l_b = 2

alpha = 0.8
beta = 1.1

cS = OverlapIntegralPWGTO(A, alpha, l_a, k_a, B, beta, l_b, k_b)
#print(cS.integral())

cN = NucAttIntegralPWGTO(A, alpha, l_a, k_a, B, beta, l_b, k_b, C, Z)
print(cN.integral())