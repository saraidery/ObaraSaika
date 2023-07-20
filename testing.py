import numpy as np
import os

from numpy import linalg as la
import matplotlib.pyplot as plt

from obara_saika import QchemBasis, PointCharge, QchemBasisPWGTO
from obara_saika import OverlapIntegralGTO, NucAttIntegralGTO, KineticIntegralGTO
from obara_saika import OverlapIntegralPWGTO, NucAttIntegralPWGTO, KineticIntegralPWGTO
from obara_saika import OverlapGTO, KineticGTO, NucAttGTO
from obara_saika import OverlapPWGTO, KineticPWGTO, NucAttPWGTO
from obara_saika import ERIGTO
from scipy.special import factorial2
from scipy.linalg.lapack import zggev, zhegv, zheev


A = np.array([0.00,  1.200,  0.700])
B = np.array([0.10,  0.300,  0.25])
C = np.array([0.30,  0.000,  -0.25])
D = np.array([0.40,  0.200,  -0.700])
alpha = 0.8
beta = 1.2
gamma = 1.1
delta = 0.5

l_a = 1
l_b = 1
l_c = 1
l_d = 1

eris = ERIGTO(A, alpha, l_a, B, beta, l_b, C, gamma, l_c, D, delta, l_d)
print(eris.integral())




