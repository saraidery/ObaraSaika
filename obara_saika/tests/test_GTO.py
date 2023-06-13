import numpy as np
import pytest
import os

from obara_saika import OverlapIntegralGTO, NucAttIntegralGTO, KineticIntegralGTO

class TestGTO:
    def __overlap__(self, l_a, l_b, S_ref):

        A = np.array([0.0, 1.2, 0.7])
        B = np.array([0.3, 0.0, -0.25])

        alpha = 0.8
        beta = 1.1

        S = OverlapIntegralGTO(A, alpha, l_a, B, beta, l_b)

        assert np.allclose(S.integral().flatten(), S_ref.flatten())

    def test_overlap_ss(self):
        S_ref = np.array([0.68913535])
        self.__overlap__(0, 0, S_ref)

    def test_overlap_ps(self):
        S_ref = np.array([0.11969193,  -0.47876772, -0.37902444])
        self.__overlap__(1, 0, S_ref)


    def test_overlap_pp(self):
        S_ref = np.array([[ 0.16623243,  0.06047592,  0.04787677],
                 [ 0.06047592, -0.06055228, -0.19150709],
                 [ 0.04787677, -0.19150709,  0.02974163]])
        self.__overlap__(1, 1, S_ref)


    def test_overlap_sd(self):
        S_ref = np.array([0.19234703, -0.07617991, -0.06030909,  0.35728136,  0.24123637,  0.29161306])
        self.__overlap__(0, 2, S_ref)


    def test_overlap_dp(self):
        S_ref = np.array([[ 0.03746228,  0.1021339,   0.080856  ],
                 [-0.20003072, -0.01821594, -0.05761105],
                 [-0.15835766, -0.05761105,  0.00894717],
                 [-0.0649224 ,  0.00770658,  0.20558759],
                 [-0.05761105,  0.05768379, -0.03578868],
                 [-0.04923977,  0.19695908, -0.04356061]])
        self.__overlap__(2, 1, S_ref)


    def test_overlap_dd(self):
        S_ref = np.array([[ 0.13595341,  0.03278479,  0.02595463,  0.10479923,  0.07076044,  0.08553714],
                 [ 0.01493013, -0.04381905, -0.13858535, -0.01954075, -0.01262037, -0.06094643],
                 [ 0.01181968, -0.13858535,  0.02152272, -0.0591145,   0.00783004, -0.00460452],
                 [ 0.14345572, -0.00168609, -0.04497956,  0.10727923,  0.00533928,  0.21749003],
                 [ 0.12730021, -0.01262037,  0.00783004,  0.06187905, -0.00783993,  0.01841808],
                 [ 0.10880262, -0.04309177,  0.00953042,  0.20209902, -0.03812169,  0.1008121 ]])
        self.__overlap__(2, 2, S_ref)


    def __nuclear_attraction__(self, l_a, l_b, V_ref):

        A = np.array([0.0, 1.2, 0.7])
        B = np.array([0.3, 0.0, -0.25])
        C = np.array([1.0, 0.0, 0.0])

        Z = 1.0
        alpha = 0.8
        beta = 1.1

        V = NucAttIntegralGTO(A, alpha, l_a, B, beta, l_b, C, Z)

        assert np.allclose(V.integral().flatten(), V_ref.flatten())

    def __kinetic__(self, l_a, l_b, T_ref):

        A = np.array([0.0, 1.2, 0.7])
        B = np.array([0.3, 0.0, -0.25])

        alpha = 0.8
        beta = 1.1

        T = KineticIntegralGTO(A, alpha, l_a, B, beta, l_b)

        assert np.allclose(T.integral().flatten(), T_ref.flatten())

    def test_kinetic_ss(self):
        S_ref = np.array([[0.23834233]])
        self.__kinetic__(0, 0, S_ref)

    def test_kinetic_ps(self):
        S_ref = np.array([[ 0.15226882],
                          [-0.60907529],
                          [-0.48218461],])
        self.__kinetic__(1, 0, S_ref)


    def test_kinetic_pp(self):
        S_ref = np.array([[ 0.19747143,  0.13295563,  0.10525654],
                          [ 0.13295563, -0.30111217, -0.42102616],
                          [ 0.10525654, -0.42102616, -0.10260204],])
        self.__kinetic__(1, 1, S_ref)


    def test_kinetic_sd(self):
        S_ref = np.array([[-0.03527816, -0.16748066, -0.13258885,  0.3273281,   0.53035541,  0.18295708]])
        self.__kinetic__(0, 2, S_ref)


    def test_kinetic_dp(self):
        S_ref = np.array([[ 0.13032456,  0.03268356,  0.02587449],
                          [-0.42291284, -0.10745725, -0.18002328],
                          [-0.334806,   -0.18002328, -0.02257785],
                          [-0.13125357, -0.0289675,   0.41563631],
                          [-0.18002328,  0.34028128,  0.09031141],
                          [-0.08224843,  0.32899372, -0.17811555],])
        self.__kinetic__(2, 1, S_ref)


    def test_kinetic_dd(self):
        S_ref = np.array([[ 0.09604206,  0.14442149,  0.11433368, -0.00377303,  0.08819035, -0.02777982],
                          [ 0.10516821, -0.25480035, -0.42137641, -0.03447746, -0.08613901, -0.15114887],
                          [ 0.08325817, -0.42137641, -0.05612506, -0.16367593, -0.01250572,  0.01672175],
                          [ 0.08121281,  0.00477582, -0.13260042, -0.14199375, -0.01512341,  0.29946709],
                          [ 0.2124924 , -0.08613901, -0.01250572,  0.10917864, -0.0092286,  -0.06688698],
                          [ 0.02336906, -0.11189559,  0.04779726,  0.26563013, -0.19118903, -0.11966924],])
        self.__kinetic__(2, 2, S_ref)
