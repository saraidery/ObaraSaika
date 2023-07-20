import numpy as np
import pytest
import os

from obara_saika import ERIGTO

class TestERIGTO:
    def __eri__(self, l_a, l_b, l_c, l_d, eri_ref):

        A = np.array([0.00,  1.200,  0.700])
        B = np.array([0.10,  0.300,  0.25])
        C = np.array([0.30,  0.000,  -0.25])
        D = np.array([0.40,  0.200,  -0.700])
        alpha = 0.8
        beta = 1.2
        gamma = 1.1
        delta = 0.5

        eri = ERIGTO(A, alpha, l_a, B, beta, l_b, C, gamma, l_c, D, delta, l_d)

        assert np.allclose(eri.integral().flatten(), eri_ref.flatten())

    def test_eri_ssss(self):
        eri_ref = np.array([2.4268106])
        self.__eri__(0, 0, 0, 0, eri_ref)

    def test_eri_psss(self):
        eri_ref = np.array([ 0.22021627, -1.47482081, -0.88095275])
        self.__eri__(1, 0, 0, 0, eri_ref)

    def test_eri_spss(self):
        eri_ref = np.array([-0.02246479, 0.70930872, 0.21111202])
        self.__eri__(0, 1, 0, 0, eri_ref)

    def test_eri_ssps(self):
        eri_ref = np.array([-0.01742172, 0.35710453, -0.05912788])
        self.__eri__(0, 0, 1, 0, eri_ref)

    def test_eri_sssp(self):
        eri_ref = np.array([-0.26010278, -0.12825759,  1.03293689])
        self.__eri__(0, 0, 0, 1, eri_ref)

    def test_eri_spps(self):
        eri_ref = np.array([ 0.08301695, 0.00351809, 0.00991933,
                            0.00173175, 0.17529691,-0.03792619,
                            0.00785644, 0.0104208 , 0.05245638])
        self.__eri__(0, 1, 1, 0, eri_ref)

    def test_eri_pssp(self):
        eri_ref = np.array([0.05925315, -0.00481471, 0.10310386,
                            0.16489338,  0.148867, -0.64838046,
                            0.10379149,  0.02591431, -0.31736481])
        self.__eri__(1, 0, 0, 1, eri_ref)

    def test_eri_dsss(self):
        eri_ref = np.array([ 0.56040121,-0.24125529,-0.15144698, 1.44624257, 0.95589778, 0.88041593])
        self.__eri__(2, 0, 0, 0, eri_ref)

    def test_eri_sdss(self):
        eri_ref = np.array([ 0.54062606, -0.02082797, -0.01637106,  0.75728169,  0.13547978,  0.5789876 ])
        self.__eri__(0, 2, 0, 0, eri_ref)

    def test_eri_ssds(self):
        eri_ref = np.array([ 0.65493378,-0.01921419,-0.01955574, 0.72227324, 0.0296262 , 0.68781892])
        self.__eri__(0, 0, 2, 0, eri_ref)

    def test_eri_sssd(self):
        eri_ref = np.array([ 0.68268623,  0.00903573, -0.21204454,  0.67650385, -0.0498582, 1.12603297])
        self.__eri__(0, 0, 0, 2, eri_ref)

