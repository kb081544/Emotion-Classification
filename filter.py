import numpy as np
import heartpy as hp
import csv
import warnings
warnings.filterwarnings(action='ignore')
import read
import glob
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture

window_size_green = 300
overlap = 0


class preprocessing:
    def __init__(self, data, train_or_test, overlap, chunk_size):
        self.data=data
        self.chunk_size=chunk_size
        self.tot=train_or_test
        self.overlap=overlap

    def chunk_data_hp(self):
        #global x_result
        data_y=[y[0] for y in self.data]
        data_x=[x[1:] for x in self.data]
        sum_removed=0
        y_result = []
        x_new_result=[]
        y_new_result=[]
        pk_list = []
        sum = 0
        cnt = 0
        exc = 0
        for sublist, label in zip(data_x, data_y):
            for i in range(0, len(sublist) - self.chunk_size + 1, self.chunk_size - self.overlap):
                x_chunk = sublist[i:i + self.chunk_size]
                filtered = hp.filter_signal(x_chunk, [0.5, 8], sample_rate=25, order=3,
                                            filtertype='bandpass')
                try:
                    wd, m = hp.process(filtered, sample_rate=25)
                    if (len(wd['peaklist']) != 0):
                        sum += (len(wd['peaklist']) - len(wd['removed_beats']))
                        sum_removed += len(wd['removed_beats'])
                        temp = wd['hr']
                        # print(temp)
                        temp_pk = (len(wd['peaklist']) - len(wd['removed_beats']))
                        if (cnt == 0):
                            x_result = temp
                        else:
                            x_result = np.vstack([x_result, temp])
                    else:
                        exc += 1
                        temp_pk = 0
                        temp = wd['hr']
                        if (cnt == 0):
                            x_result = temp
                        else:
                            x_result = np.concatenate((x_result, temp))
                    cnt += 1
                    pk_list.append(temp_pk)
                    y_result.append(label)
                except:
                    print("예외처리")
                    continue
        pk_np = np.array(pk_list)
        # print(x_result)
        print(x_result.shape)
        # print(pk_np)
        print(pk_np.shape)
        print("cnt: ", cnt)
        print('exc: ', exc)
        print("y result: ", len(y_result))
        new_temp = 0
        new_cnt = 0
        if(self.tot=="train"):
            for j in range(cnt):
                # available_signal.append(pk_list[j])
                x_new_result.append(x_result[j])
                y_new_result.append(y_result[j])
                new_temp += m['bpm']
                new_cnt += 1
            new_avg = new_temp / new_cnt
            # 개수 평균이 아닌 bpm 평균 계산해야함
            print("heartpy 평균: ", new_avg)
            # print(new_avg.quantile(q=))
            print("cnt: ", cnt)
            print('exc: ', exc)
            print("필터링 후 데이터셋 개수: ", len(x_new_result))
            print("y result: ", len(y_new_result))
            print(sum_removed)
            return x_new_result, y_new_result

        elif(self.tot=="test"):
            return x_result, y_result
    def dividing_and_extracting(self):
        x,y=self.chunk_data_hp()
        peak_shapes = []
        fake_index = []

        index = np.where(np.max(x, axis=1) >= 14000)[0]  # >= 14000, 알아서 자르는 수 조정
        new_data = np.delete(x, index, axis=0)
        new_data_y=np.delete(y, index, axis=0)

        fake_index.extend(index)
        # l = len(index)
        for i in range(len(new_data)):
            temp = new_data[i, :]
            temp_y=new_data_y[i]
            wd, m = hp.process(temp, sample_rate=25)
            peaks = wd['peaklist']
            fake_peaks = wd['removed_beats']
            fake_index.extend(fake_peaks)
            real_peaks = [item for item in peaks if item not in fake_peaks]
            for index in real_peaks:
                if not ((index - 13 < 0) or (index + 14 >= new_data.shape[1])):
                    peak_shape = temp[index - 13:index + 14]
                    peak_shape = np.concatenate((np.array([temp_y]), peak_shape))
                    peak_shapes.append(peak_shape)

        np_peak = np.array(peak_shapes)
        print(np_peak.shape)
        return np_peak

    def GMM_model(self, tot, gmm_p=None, gmm_n=None):
        if tot == "train":
            data = self.dividing_and_extracting()
            print(np.shape(data))
            data0 = data[data[:, 0] == 0]
            data0 = data0[:, 1:]
            data1 = data[data[:, 0] == 1]
            data1 = data1[:, 1:]

            n_components = 2
            gmm_p = GaussianMixture(n_components=n_components, covariance_type='full')
            gmm_p.fit(data0)

            # 긍정 gmm model
            labels = gmm_p.predict(data0)
            outliers = data0[labels == 1]
            normals = data0[labels == 0]

            # 부정 gmm model
            gmm_n = GaussianMixture(n_components=n_components, covariance_type='full')
            gmm_n.fit(data1)
            labels = gmm_n.predict(data1)
            outliers_n = data1[labels == 1]
            normals_n = data1[labels == 0]

            global lab1
            global lab0

            if np.mean(normals_n) > np.mean(outliers_n):
                spp1 = normals_n
                lab1 = 0
            else:
                spp1 = outliers_n
                lab1 = 1
            if np.mean(normals) < np.mean(outliers):
                spp0 = normals
                lab0 = 0
            else:
                spp0 = outliers
                lab0 = 1

            global m
            global n
            m = np.max(spp1)
            n = np.min(spp1)
            normalized_train = []
            for value in spp0:
                normalized_num = (value - n) / (m - n)
                normalized_train.append(normalized_num)
            normalized_train_n = []
            for value in spp1:
                normalized_num = (value - n) / (m - n)
                normalized_train_n.append(normalized_num)

            normalized_train = np.array(normalized_train)
            normalized_train_n = np.array(normalized_train_n)

            normals_y = np.zeros((normalized_train.shape[0], 1))
            g_x_p = np.concatenate((normals_y, normalized_train), axis=1)
            normals_n_y = np.ones((normalized_train_n.shape[0], 1))
            g_x_n = np.concatenate((normals_n_y, normalized_train_n), axis=1)

            data = np.concatenate((g_x_p, g_x_n))
            np.random.shuffle(data)
            data_x = data[:, 1:]
            data_y = data[:, 0]

            x_train_g, x_test_g, y_train_g, y_test_g = train_test_split(data_x, data_y, test_size=0.2)
            return x_train_g, x_test_g, y_train_g, y_test_g, gmm_p, gmm_n

        elif tot == "test":
            if gmm_p is None or gmm_n is None:
                raise ValueError("GMM models must be provided for test data")

            data = self.dividing_and_extracting()
            d = np.array(data)
            dy = d[:, 0]
            d = d[:, 1:]
            tst = []

            lb1 = gmm_n.predict(d)
            lb2 = gmm_p.predict(d)

            for i in range(len(lb1)):
                if lb1[i] != lab1 and lb2[i] != lab0:
                    pass
                else:
                    tst.append(d[i])

            normalized = []
            for value in tst:
                normalized_num = (value - n) / (m - n)
                normalized.append(normalized_num)

            data = np.array(normalized)
            data_x = data
            data_y = dy
            return data_x, data_y
