# -*- coding: utf-8 -*-
import numpy as np
from DataProcessing import SeismicDataProcessing
from NewmarkBetaAnalysis import NewmarkBetaAnalysis

# To-DO List:
# 1. 免震層剛性の計算の変数の流れを明白してやり直す    ※　Working　On
# 2. 最後のグラフの部分を完成する
# 3. 結果(途中も)を保存する、次は前回から続く


def outputs():
    pass


def main():
    div = 100
    # 入力地震波の処理  (div：増分数 amp_to:増幅係数)
    gcc_zz = SeismicDataProcessing(div=div, amp_to='L3').gcc_increment
    # 質量ベクトル（下層から）
    m = np.array([1102, 650, 542, 542, 596, 537, 537, 538, 539, 539, 539,
                  538, 537, 537, 537, 537, 537, 537, 537, 537, 542, 712])
    # 剛性ベクトル（下層から）
    k = np.array([0, 838, 966, 899, 878, 872, 864, 855, 830, 828, 822,
                  815, 805, 805, 801, 791, 780, 756, 719, 667, 625, 411])
    instance1 = NewmarkBetaAnalysis(gcc_zz, m, k)
    instance1.newmark()


if __name__ == "__main__":
    main()


