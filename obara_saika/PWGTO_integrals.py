import numpy as np

from obara_saika.angular_momentum import get_n_cartesian, get_cartesians, get_n_cartesian_accumulated, get_cartesian_index_accumulated, get_cartesians_accumulated
from obara_saika.GTO import GTO, PWGTO, ShellPWGTO
from obara_saika.math import boys_kummer
from obara_saika.GTO_integrals import BaseIntegralGTO

class BaseIntegralPWGTO(BaseIntegralGTO):

    @property
    def k_a(self):
        return self.sh_A.k

    @property
    def k_b(self):
        return self.sh_B.k

class OverlapIntegralPWGTO(BaseIntegralPWGTO):

    def __init__(self, A, alpha, l_a, k_a, B, beta, l_b, k_b):

        self.sh_A = ShellPWGTO(A, alpha, l_a, k_a)
        self.sh_B = ShellPWGTO(B, beta, l_b, -k_b)

        [self.p, self.P, self.K, self.Ad, self.Bd] = self.sh_A * self.sh_B

        self.PA = self.P - self.Ad
        self.PB = self.P - self.Bd

    def do_recurrence(self, a, b, cart, PX, I, k, alpha):

        idx_cart = np.argmax(cart)
        a_q = a[idx_cart]
        b_q = b[idx_cart]

        c_a = get_cartesian_index_accumulated(a)
        c_b = get_cartesian_index_accumulated(b)

        value = PX[idx_cart] * I[c_a, c_b] + 1j * k[idx_cart]/(2.0*alpha) * I[c_a, c_b]

        if (a_q > 0):
           c_a_m = get_cartesian_index_accumulated(a-cart)
           value += 1.0/(2.0 * self.p)*a_q*(I[c_a_m, c_b])
        if (b_q > 0):
           c_b_m = get_cartesian_index_accumulated(b-cart)
           value += 1.0/(2.0 * self.p)*b_q*(I[c_a, c_b_m])

        return value

    def integral(self):

        dim_a = get_n_cartesian_accumulated(self.l_a)
        dim_b = get_n_cartesian_accumulated(self.l_b)

        I = np.zeros([dim_a, dim_b], dtype=complex)

        basis_s_P = GTO(self.p, np.real(self.P), np.array([0, 0, 0], dtype=int))

        I[0, 0] = self.K*basis_s_P.GTO_s_overlap_3d()

        incr = [np.array([0, 0, 0], dtype=int),
                np.array([1, 0, 0], dtype=int),
                np.array([0, 1, 0], dtype=int),
                np.array([0, 0, 1], dtype=int)]

        if (self.l_a == 0):
            for b in get_cartesians_accumulated(self.l_b):
                for j in incr:
                    if (sum(j) == 0):
                        continue
                    a = np.array([0, 0, 0], dtype=int)
                    c_a = get_cartesian_index_accumulated(a)
                    c_b = get_cartesian_index_accumulated(b + j)

                    I[c_a, c_b] = self.do_recurrence(a, b, j, self.PB, I, self.k_b, self.beta)

        if (self.l_b == 0):
            for a in get_cartesians_accumulated(self.l_a):
                for i in incr:
                    if (sum(i) == 0):
                        continue

                    b = np.array([0, 0, 0], dtype=int)
                    c_a = get_cartesian_index_accumulated(a + i)
                    c_b = get_cartesian_index_accumulated(b)

                    I[c_a, c_b] = self.do_recurrence(a, b, i, self.PA, I, self.k_a, self.alpha)

        for a in get_cartesians_accumulated(self.l_a):
            for b in get_cartesians_accumulated(self.l_b):

                for i in incr:
                    for j in incr:
                        if (np.sum(i) + np.sum(j) == 0):
                            continue

                        c_a = get_cartesian_index_accumulated(a + i)
                        c_b = get_cartesian_index_accumulated(b + j)

                        if (np.sum(i) == 0):
                            I[c_a, c_b] = self.do_recurrence(a, b, j, self.PB, I, self.k_b, self.beta)
                        if (np.sum(j) == 0):
                            I[c_a, c_b] = self.do_recurrence(a, b, i, self.PA, I, self.k_a, self.alpha)
                        if (np.sum(i) == np.sum(j)):
                            I[c_a, c_b] = self.do_recurrence(a, b+j, i, self.PA, I, self.k_a, self.alpha)

        extract_a = dim_a - get_n_cartesian(self.l_a)
        extract_b = dim_b - get_n_cartesian(self.l_b)

        S_shp = I[extract_a:,extract_b:]

        normalization = self.normalization_array()

        return np.multiply(S_shp, normalization)

class NucAttIntegralPWGTO(BaseIntegralPWGTO):

    def __init__(self, A, alpha, l_a, k_a, B, beta, l_b, k_b, C, Z):

        self.sh_A = ShellPWGTO(A, alpha, l_a, k_a)
        self.sh_B = ShellPWGTO(B, beta, l_b, -k_b)
        self.C = C
        self.Z = Z

        [self.p, self.P, self.K, self.Ad, self.Bd] = self.sh_A * self.sh_B

        self.PA = self.P - self.Ad
        self.PB = self.P - self.Bd
        self.PC = self.P - self.C


    def do_recurrence(self, a, b, cart, PX, aux, m, k, alpha):

        idx_cart = np.argmax(cart)
        a_q = a[idx_cart]
        b_q = b[idx_cart]

        c_a = get_cartesian_index_accumulated(a)
        c_b = get_cartesian_index_accumulated(b)

        value = (
                 + PX[idx_cart] * aux[c_a, c_b, m]
                 - self.PC[idx_cart] * aux[c_a, c_b, m + 1]
                 + 1j * k[idx_cart]/(2.0*alpha) * aux[c_a, c_b, m])

        if (a_q > 0):
           c_a_m = get_cartesian_index_accumulated(a-cart)
           value += 1.0/(2.0 * self.p)*a_q*(aux[c_a_m, c_b, m] - aux[c_a_m, c_b, m + 1])
        if (b_q > 0):
           c_b_m = get_cartesian_index_accumulated(b-cart)
           value += 1.0/(2.0 * self.p)*b_q*(aux[c_a, c_b_m, m] - aux[c_a, c_b_m, m + 1])


        return value

    def auxiliary_integral_s(self, m):

        U = self.p * np.dot(self.PC, self.PC)
        gto_s_P = GTO(self.p, np.real(self.P), np.array([0, 0, 0], dtype=int))

        return self.K * 2.0 * pow(self.p / np.pi, 0.5) * gto_s_P.GTO_s_overlap_3d() * boys_kummer(m, U)

    def integral(self):

        dim_a = get_n_cartesian_accumulated(self.l_a)
        dim_b = get_n_cartesian_accumulated(self.l_b)

        aux = np.zeros([dim_a, dim_b, dim_a + dim_b + 1], dtype=complex)

        for i in np.arange(dim_a + dim_b + 1):
            aux[0, 0, i] = self.auxiliary_integral_s(i)

        incr = [np.array([0, 0, 0], dtype=int),
                np.array([1, 0, 0], dtype=int),
                np.array([0, 1, 0], dtype=int),
                np.array([0, 0, 1], dtype=int)]

        if (self.l_a == 0):
            for b in get_cartesians_accumulated(self.l_b):
                for j in incr:
                    if (sum(j) == 0):
                        continue

                    a = np.array([0, 0, 0], dtype=int)
                    c_a = get_cartesian_index_accumulated(a)
                    c_b = get_cartesian_index_accumulated(b + j)

                    for m in np.arange(self.l_b + self.l_a + 1 - sum(a) - sum(b)):
                        aux[c_a, c_b, m] = self.do_recurrence(a, b, j, self.PB, aux, m, self.k_b, self.beta)

        if (self.l_b == 0):
            for a in get_cartesians_accumulated(self.l_a):
                for i in incr:
                    if (sum(i) == 0):
                        continue

                    b = np.array([0, 0, 0], dtype=int)
                    c_a = get_cartesian_index_accumulated(a + i)
                    c_b = get_cartesian_index_accumulated(b)

                    for m in np.arange(self.l_b + self.l_a + 1 - sum(a) - sum(b)):
                        aux[c_a, c_b, m] = self.do_recurrence(a, b, i, self.PA, aux, m, self.k_a, self.alpha)

        for a in get_cartesians_accumulated(self.l_a):
            for b in get_cartesians_accumulated(self.l_b):

                for i in incr:
                    for j in incr:
                        if (np.sum(i) + np.sum(j) == 0):
                            continue

                        c_a = get_cartesian_index_accumulated(a + i)
                        c_b = get_cartesian_index_accumulated(b + j)

                        for m in np.arange(self.l_b + self.l_a + 1 - sum(a) - sum(b)):
                            if (np.sum(i) == 0):
                                aux[c_a, c_b, m] = self.do_recurrence(a, b, j, self.PB, aux, m, self.k_b, self.beta)
                            if (np.sum(j) == 0):
                                aux[c_a, c_b, m] = self.do_recurrence(a, b, i, self.PA, aux, m, self.k_a, self.alpha)
                            if (np.sum(i) == np.sum(j)):
                                aux[c_a, c_b, m] = self.do_recurrence(a, b+j, i, self.PA, aux, m, self.k_a, self.alpha)

        extract_a = dim_a - get_n_cartesian(self.l_a)
        extract_b = dim_b - get_n_cartesian(self.l_b)

        V_shp = -self.Z*aux[extract_a:,extract_b:, 0]

        normalization = self.normalization_array()

        return np.multiply(V_shp, normalization)
