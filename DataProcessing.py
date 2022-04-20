# -*- coding: utf-8 -*-

import json
import os

import numpy as np
import pandas as pd


# 　データを増分するより、許容値以内に収まるように繰り返し収束計算

class SeismicDataProcessing(object):
    """ このクラスは入力地震波の読み取りと処理を行う。
        処理したデータを保存すれば、次回は読み込んで解析が速くなる
            data[Index]: ElCentroNS, HachiNS, TaftEW """

    def __init__(self, div=10, amp_to='L2'):
        self.div = div  # 増分数 default:10
        self.dt = 0.02 / self.div  # 増分後の時刻刻み
        self.gcc = self.raw_data = self.diff = self.data = self.steps = None
        self.gcc_increment = [[], [], []]  # [ElCentroNS, HachiNS, TaftEW]
        self.amp_to = amp_to  # default: L2
        self.awave = self.__choose_awave()  # awave: 波の増幅係数
        self.file = 'SeismicData\\SeismicData.csv'
        self.save = 'ProcessedData\\GndAccInc_' + self.amp_to + '_' + str(self.div) + '.json'
        self.make_or_load()

    def __choose_awave(self):
        """ ElCNS      75/33.5 (50kine L2に合わせた場合)
            TaftEW     50kine/17.7kine = 2.825
            八戸NS      50kine/34.4kine = 1.453 """
        if self.amp_to == 'L3':
            return 75 / np.array([33.5, 34.4, 17.7])
        elif self.amp_to == 'L2':
            return 50 / np.array([33.5, 34.4, 17.7])

    def ground_acceleration(self):
        """地震のデータを読む取ると増幅係数を乗ずる"""
        self.raw_data = pd.read_csv(self.file, index_col=False) / 100  # 単位変換 Gal -> m/s2
        self.gcc = {  # Ground Acceleration
            0: self.raw_data['ElCentroNS'].dropna(how='all') * self.awave[0],  # .dropna(): delete all Nan
            1: self.raw_data['HachiNS'].dropna(how='all') * self.awave[1],
            2: self.raw_data['TaftEW'].dropna(how='all') * self.awave[2]
        }
        self.steps = [len(self.gcc[i]) for i in range(3)]

    def increment(self):
        for case in range(3):  # 3つの地震増分を計算をする
            for i in range(self.steps[case] - 1):
                self.diff = (self.gcc[case][i + 1] - self.gcc[case][i]) / self.div
                for j in range(self.div):
                    self.gcc_increment[case].append(self.diff)

    def make_or_load(self):
        """既存のデータがあれば、そのままロードして使う。なければ新しく作る"""
        if os.path.exists(self.save):
            print('Loading Exist Data...')
            with open(self.save, 'r') as fo:
                self.gcc_increment = json.load(fo)
        else:
            print('Making New File ...')
            self.ground_acceleration()
            self.increment()
            data = json.dumps(self.gcc_increment, indent=4)
            with open(self.save, 'w') as fw:
                fw.write(data)


if __name__ == '__main__()':
    gcc_inc = SeismicDataProcessing(div=60, amp_to='L3').gcc_increment
