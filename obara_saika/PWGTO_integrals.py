import numpy as np

from obara_saika.angular_momentum import get_n_cartesian, get_cartesians, get_n_cartesian_accumulated, get_cartesian_index_accumulated, get_cartesians_accumulated
from obara_saika.GTO import GTO, PWGTO, ShellPWGTO
from obara_saika.math import boys_kummer
from obara_saika.GTO_integrals import BaseIntegralGTO, OverlapIntegralGTO, NucAttIntegralGTO, KineticIntegralGTO

class BaseIntegralPWGTO:

    @property
    def k_a(self):
        return self.sh_A.k

    @property
    def k_b(self):
        return self.sh_B.k

    def make_xi(self, PX, k, x):

        return PX + 1j * k/(2.0*x)

class OverlapIntegralPWGTO(OverlapIntegralGTO, BaseIntegralPWGTO):

    def __init__(self, A, alpha, l_a, k_a, B, beta, l_b, k_b):

        self.sh_A = ShellPWGTO(A, alpha, l_a, k_a)
        self.sh_B = ShellPWGTO(B, beta, l_b, -k_b)

        [self.p, self.P, self.K, self.Ad, self.Bd] = self.sh_A * self.sh_B

        self.PA = self.P - self.Ad
        self.PB = self.P - self.Bd

    def integral(self):

        dim_a = get_n_cartesian_accumulated(self.l_a)
        dim_b = get_n_cartesian_accumulated(self.l_b)

        xi_A = self.make_xi(self.PA, self.k_a, self.alpha)
        xi_B = self.make_xi(self.PB, self.k_b, self.beta)

        I = np.zeros([dim_a, dim_b], dtype=complex)

        self.integral_accumulated(I, xi_A, xi_B)

        extract_a = dim_a - get_n_cartesian(self.l_a)
        extract_b = dim_b - get_n_cartesian(self.l_b)

        S_shp = I[extract_a:,extract_b:]

        normalization = self.normalization_array()

        return np.multiply(S_shp, normalization)

class NucAttIntegralPWGTO(NucAttIntegralGTO, BaseIntegralPWGTO):

    def __init__(self, A, alpha, l_a, k_a, B, beta, l_b, k_b, C, Z):

        self.sh_A = ShellPWGTO(A, alpha, l_a, k_a)
        self.sh_B = ShellPWGTO(B, beta, l_b, -k_b)
        self.C = C
        self.Z = Z

        [self.p, self.P, self.K, self.Ad, self.Bd] = self.sh_A * self.sh_B

        self.PA = self.P - self.Ad
        self.PB = self.P - self.Bd
        self.PC = self.P - self.C

    def integral(self):

        dim_a = get_n_cartesian_accumulated(self.l_a)
        dim_b = get_n_cartesian_accumulated(self.l_b)

        aux = np.zeros([dim_a, dim_b, dim_a + dim_b + 1], dtype=complex)

        xi_A = self.make_xi(self.PA, self.k_a, self.alpha)
        xi_B = self.make_xi(self.PB, self.k_b, self.beta)

        self.integral_accumulated(aux, xi_A, xi_B)

        extract_a = dim_a - get_n_cartesian(self.l_a)
        extract_b = dim_b - get_n_cartesian(self.l_b)

        V_shp = -self.Z*aux[extract_a:,extract_b:, 0]

        normalization = self.normalization_array()

        return np.multiply(V_shp, normalization)


class KineticIntegralPWGTO(KineticIntegralGTO, BaseIntegralPWGTO):

    def __init__(self, A, alpha, l_a, k_a, B, beta, l_b, k_b):

        self.sh_A = ShellPWGTO(A, alpha, l_a, k_a)
        self.sh_B = ShellPWGTO(B, beta, l_b, -k_b)

        [self.p, self.P, self.K, self.Ad, self.Bd] = self.sh_A * self.sh_B

        self.PA = self.P - self.Ad
        self.PB = self.P - self.Bd

    def get_integral_over_s(self, S_00):

        xi_A = self.make_xi(self.PA, self.k_a, self.alpha)
        xi_B = self.make_xi(self.PB, self.k_b, self.beta)

        tmp = (2.0 * self.alpha * self.beta * np.dot(xi_A, xi_B)
             + 3.0 * self.alpha * self.beta /(self.p)
             - np.dot(self.k_a, self.k_b)/2.0
             -1j * self.alpha * np.dot(xi_A, self.k_b)
             -1j * self.beta * np.dot(xi_B, self.k_a))

        return  tmp * S_00

    def integral(self):

        dim_a = get_n_cartesian_accumulated(self.l_a)
        dim_b = get_n_cartesian_accumulated(self.l_b)

        I = np.zeros([dim_a, dim_b], dtype=complex)

        xi_A = self.make_xi(self.PA, self.k_a, self.alpha)
        xi_B = self.make_xi(self.PB, self.k_b, self.beta)

        overlap = OverlapIntegralPWGTO(self.A, self.alpha, self.l_a + 1, self.k_a, self.B, self.beta, self.l_b + 1, -self.k_b)

        S = np.zeros([get_n_cartesian_accumulated(self.l_a+1), get_n_cartesian_accumulated(self.l_b+1)], dtype=complex)
        overlap.integral_accumulated(S, xi_A, xi_B)

        self.integral_accumulated(I, xi_A, xi_B, S)


        extract_a = dim_a - get_n_cartesian(self.l_a)
        extract_b = dim_b - get_n_cartesian(self.l_b)

        I_shp = I[extract_a:,extract_b:]

        normalization = self.normalization_array()

        return np.multiply(S_shp, normalization)

