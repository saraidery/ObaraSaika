import numpy as np
import os

from obara_saika import QchemBasis, PointCharge
from obara_saika import OverlapIntegralGTO, NucAttIntegralGTO, KineticIntegralGTO
from obara_saika import OverlapIntegralPWGTO, NucAttIntegralPWGTO, KineticIntegralPWGTO
from obara_saika import OverlapGTO, KineticGTO, NucAttGTO


file_path = os.getcwd()
print(file_path)
file_name = os.path.join(file_path, "obara_saika/tests/single_atom_sto3g.txt")
b = QchemBasis(file_name)

pc = [PointCharge(np.array(b.centers[0]), 1.0)]

S = NucAttGTO(b, pc)
I = S.get_nuclear_attraction()
print(I)

#A = np.zeros(3)
#B = np.zeros(3)
#C = np.zeros(3)
#C[0] = 1.0
#
#l_a = 2
#l_b = 2
#
#Z = 1.0
#
#A[1] = 1.2
#A[2] = 0.7
#
#B[0] = 0.3
#B[2] = -0.25
#
#k_a = np.array([0.25, 0.50, 0.75])
#k_b = np.array([-0.30, 0.00, -0.60])
#
#alpha = 0.8
#beta = 1.1
#
#N = NucAttIntegralGTO(A, alpha, l_a, B, beta, l_b, C, Z)
#print(N.integral())

#S = OverlapIntegralGTO(A, alpha, l_a, B, beta, l_b)
#print(S.integral())

#K = KineticIntegralGTO(A, alpha, l_a, B, beta, l_b)
#I = K.integral()
#print(I)
#
#cK = KineticIntegralPWGTO(A, alpha, l_a, k_a, B, beta, l_b, k_b)
#I = cK.integral()
#print(I)

#cS = OverlapIntegralPWGTO(A, alpha, l_a, k_a, B, beta, l_b, k_b)
#print(cS.integral())
#
#cN = NucAttIntegralPWGTO(A, alpha, l_a, k_a, B, beta, l_b, k_b, C, Z)
#print(cN.integral())
#for elm in I.flatten():
#	print(f"{elm:.20}")