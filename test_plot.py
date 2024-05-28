import numpy as np
import heartpy as hp
import csv
import warnings
import CNN_model
import glob
import filter
import read
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

warnings.filterwarnings(action='ignore')
load_model=tf.keras.models.load_model("best_model.h5")

g1=glob.glob("C:/Users/user/PycharmProjects/Emotion Classification 2/plotting_dataset/g1.txt")
class Plot:
    def __init__(self, data, tot, overlap, chunk_size, gmm_n, gmm_p, lab0, lab1, m, n):
        self.data=data
        self.chunk_size=chunk_size
        self.tot=tot
        self.overlap=overlap
        self.gmm_n = gmm_n
        self.gmm_p = gmm_p
        self.lab0 = lab0
        self.lab1 = lab1
        self.m = m
        self.n = n

    def dividing_and_extracting(self):
        #test_data = read.read(self.data, 1, chunk_size=self.chunk_size, overlap=self.overlap)
        #test_data_list = test_data.read_txt_files_with_skip()
        test_filtered = filter.preprocessing(data=self.data, chunk_size=self.chunk_size, train_or_test="test",
                                             overlap=self.overlap)
        x, y = test_filtered.chunk_data_hp()

        pk_list = []
        fake_index = []

        index = np.where(np.max(x, axis=1) >= 14000)[0]  # >= 14000, 알아서 자르는 수 조정
        new_data = np.delete(x, index, axis=0)
        new_data_y = np.delete(y, index, axis=0)

        fake_index.extend(index)
        # l = len(index)
        for i in range(len(new_data)):
            temp = new_data[i, :]
            temp_y = new_data_y[i]
            wd, m = hp.process(temp, sample_rate=25)
            temp_pk_array = np.full((1, 300), np.nan)
            temp_pk_array[0, wd['removed_beats']] = -1
            peaks = wd['peaklist']
            fake_peaks = wd['removed_beats']
            fake_index.extend(fake_peaks)
            real_peaks = [item for item in peaks if item not in fake_peaks]
            for index in real_peaks:
                if not ((index - 13 < 0) or (index + 14 >= new_data.shape[1])):
                    peak_shape = temp[index - 13:index + 14]
                    #peak_shape = np.concatenate((np.array([temp_y]), peak_shape))
                    d = peak_shape.reshape(1, -1)
                    lb1 = self.gmm_n.predict(d)
                    lb2 = self.gmm_p.predict(d)

                    for i in range(len(lb1)):
                        if lb1[i] != self.lab1 and lb2[i] != self.lab0:
                            pk_gmm = -1
                        else:
                            pk_gmm = peak_shape

                    if isinstance(pk_gmm, np.ndarray):
                        normalized = []
                        for value in pk_gmm:
                            normalized_num = (value - self.n) / (self.m - self.n)
                            normalized.append(normalized_num)
                        pk_nor = np.array(normalized)
                        peak_reshaped = pk_nor.reshape(1, 27)
                        prediction = load_model.predict(peak_reshaped)
                        temp_pk_array[0, index] = prediction
                    else:
                        temp_pk_array[0, index] = -1

            condition = np.logical_or(temp_pk_array >= 0, temp_pk_array == -1)
            for value in temp_pk_array[condition]:
                pk_list.append(value)

        np_peak = np.array(pk_list)
        print(np_peak.shape)
        return np_peak

    def plot_data(self):
        data = self.dividing_and_extracting()
        negative_one_indices = [i for i, value in enumerate(data) if value == -1]
        other_indices = [i for i, value in enumerate(data) if value != -1]
        other_values = [value for value in data if value != -1]
        f = interp1d(other_indices, other_values, kind='previous', fill_value="extrapolate")
        x_new = np.linspace(min(other_indices), max(other_indices), num=300, endpoint=True)
        y_new = f(x_new)
        plt.plot(x_new, y_new, color='blue')
        plt.plot(negative_one_indices, [-1] * len(negative_one_indices), 'ro', markersize=3)
        plt.show()