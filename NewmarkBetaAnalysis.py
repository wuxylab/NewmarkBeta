# -*- coding: utf-8 -*-

import numpy as np
from numba import jit
from scipy.linalg import eigh
from SeismicIsolationFloor import *


class NewmarkBetaAnalysis(object):
    """
    This class is used to solve Time History Response Analysis by NewmarkBeta.
    """

    def __init__(self, data, div, m, k):
        self.period = 1
        self.beta = 1 / 6
        self.gamma = 0.5
        self.dt = 0.02 / div  # 時刻刻み (増分後)
        self.h = 0.03  # 上部構造の粘性減衰
        self.gcc = data  # 入力地震波（加速度）
        self.m = m  # 質量ベクトル
        self.k = k  # 剛性ベクトル
        self.deg = len(self.m) + 1  # 自由度 (=22 in present case)

        self.M = self.make_M()
        self.K = self.make_K()
        self.steps = len(self.gcc)
        self.get_mode()
        self.C = None

    def make_K(self):
        """剛性行列 Stiffness Matrix を生成する関数"""
        self.k = np.append(self.k, [0]) * 1000  # Gal -> kN/m
        K_upper = np.diag(self.k[1:]) - np.diag(self.k[1:-1], k=1)
        K_lower = np.diag(self.k[:-1]) - np.diag(self.k[1:-1], k=-1)
        return K_upper + K_lower

    def make_M(self):
        """質量行列 Mass Matrix を生成する関数"""
        return np.diag(self.m)

    def get_C(self):
        C = 2 * self.h / self.omega[0] * self.K
        return C

    def get_Kbase(self):
        """Maybe another function name?"""
        pass

    def get_mode(self):
        """ a @ v = w @ b @ v
            W: (Eigenvalue) 固有振動数^2
            V: (Eigenvector) 固有モード"""
        self.W, self.V = eigh(self.K[1:, 1:], self.M[1:, 1:])  # 免震層以外の層
        self.omega = np.sqrt(self.W)

    @jit    # JITによる加速
    def newmark(self):
        # init acc, vel, disp = m*1 vector
        disp = np.zeros(self.deg, self.steps)  # 変位 displacement
        vel = np.zeros(self.deg, self.steps)  # 速度 velocity
        acc = np.zeros(self.deg, self.steps)  # 加速度 acceleration
        A = (1 / (2 * self.beta * self.dt),  # A0
             1 / (self.beta * self.dt ** 2),  # A1
             1 / (2 * self.beta),  # A2
             (self.beta / 4 - 1) * self.dt,  # A3
             1 / (self.beta * self.dt))  # A4
        self.C = self.get_C()

        for i in range(self.steps):
            kb = get_kb(dispb=disp[0][i])   # disp_base
            # Maxwell Modelによる修正
            cd_mx = 0
            if isRF is True:
                for i in range(OD['k']):
                    if abs(v_02) < OD['vy'][i]:
                        cd_mx += OD['n'][i] * OD['c1'][i]
                    else:
                        cd_mx += OD['n'][i] * OD['c2'][i]
            else:
                cd_mx = OD['n'] @ OD['c1']
            k_mx = OD['n'] @ OD['sk']
            keq_mx = 0 if abs(v_02) < e-20 else (k_mx * cd_mx / self.dt) / (k_mx + cd_mx / self.dt)
            kb += keq_mx

            K_xx = self.K
            K_xx = K_xx[0][0] + kb
            C_ss = 2*self.h / .W[0] * self.K

            K_star = K_xx + A[0] * C_ss + A[1] * self.M
            delta_F = -self.m * self.gcc[i] + \
                      self.M @ (A[0] * vel[:, i] + A[2] * acc[:, i]) + \
                      C_ss @ (A[2] * vel[:, i] + A[3] * acc[:, i])

            delta_disp = np.invert(K_star).dot(delta_F)  # 変位の変化
            delta_vel = A[0] * delta_disp - A[2] * vel[:, i] - A[3] * acc[:, i]  # 速度の変化
            delta_acc = A[1] * delta_disp - A[4] * vel[:, i] - A[2] * acc[:, i]  # 加速度の変化

            disp[:, i + 1] = disp[:, i] + delta_disp
            vel[:, i + 1] = vel[:, i] + delta_vel
            acc[:, i + 1] = acc[:, i] + delta_acc
        return disp, vel, acc

    def newmark_loop(self):
        pass


# class NewmarkBetaMaxwell(NewmarkBetaAnalysis):
#     """MAXWELLモデルの修正機能を追加したプログラム"""
#
#     def __init__(self, period, delta_t, h, data):
#         super(NewmarkBetaAnalysis, self).__init__()
#         self.cd_maxwell = 0
#         self.vel_02 = 0  # 節点0-2間の相対速度
#         self.delta_d0 = 0  # 内部節点0の変位増分
#         self.Device = SeismicIsolationDevice()
#
#     def maxwell_correction(self):
#         """この関数はMAXWELLモデルの修正を行う"""
#         for i in range(Device.OD['k']):  # cd_maxwell
#             c_OD = Device.OD['c1'][i] if abs(self.vel_02) < Device.OD['vy'] else Device.OD['c2'][i]
#             self.cd_maxwell += Device.OD['n'][i]
#
#         for i in range(Device.OD['k']):
#             self.cd_maxwell += Device.OD['n'].dot(Device.OD['c1'])


def drift_max(deg, disp, temp_drift_max):
    # 層間変位
    drift = np.zero((deg - 1, 1))
    drift[0] = disp[0]
    for i in range(deg - 1):
        drift[i + 1] = disp[i + 1] - disp[i]
    drift = np.abs(drift) * 1000  # mmに単位変換
    drift_max = np.maximum(drift, temp_drift_max)
    return drift_max


def shearforce_max(deg, q, q_old, bk1, dltx, dltx_old, df0, temp_q_max):
    q = q_old + bk1 * (dltx - dltx_old)
    q[0] = q_old[0] + kb * dltx[0] - df0
    q_max = np.maximum(np.abs(q) * 1000, temp_q_max)  # mmに単位変換
