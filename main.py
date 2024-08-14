import CNN_model
import glob
import filter
import read
import test_plot
import joblib
import pickle

chunk_size=300
overlap=0

file_positive_green = glob.glob(r"P\green\*.txt")
file_negative_green=glob.glob(r"N\green\*.txt")

h1=glob.glob("C:/Users/user/PycharmProjects/Emotion Classification 2/plotting_dataset/h1.txt")
g1=glob.glob("C:/Users/user/PycharmProjects/Emotion Classification 2/plotting_dataset/g1.txt")

train_positive_reader=read.read(data=file_positive_green, y=0, chunk_size=chunk_size, overlap=overlap)
train_negative_reader=read.read(data=file_negative_green, y=1, chunk_size=chunk_size, overlap=overlap)
p_train_list=train_positive_reader.read_txt_files_with_skip()
n_train_list=train_negative_reader.read_txt_files_with_skip()
train_list=p_train_list+n_train_list

gmm_n_from_joblib = joblib.load('gmm_n_v2.pkl')
gmm_p_from_joblib = joblib.load('gmm_p_v2.pkl')

with open("list_v2.pickle","rb") as fi:
    test = pickle.load(fi)
lab0_f = test[0]
lab1_f = test[1]
m_f = test[2]
n_f = test[3]

#train_filtered=filter.preprocessing(data=train_list, chunk_size=chunk_size,train_or_test="train", overlap=overlap)
#x_train, x_test, y_train, y_test, gmm_p, gmm_n, lab0, lab1, m, n = train_filtered.GMM_model(tot="train")
#training=CNN_model.model(x_train, x_test, y_train, y_test)
#history, predictions, score=training.build_model()

test_data=read.read(h1, 1, chunk_size=chunk_size, overlap=overlap)
test_data_list=test_data.read_txt_files_with_skip()
test_filtered=filter.preprocessing(data=test_data_list, chunk_size=chunk_size, train_or_test="test",overlap=overlap)
x_test, y_test = test_filtered.GMM_model(tot="test", gmm_p=gmm_p_from_joblib, gmm_n=gmm_n_from_joblib)
print(x_test)
print(y_test)

#plot
test_plotter = test_plot.Plot(data = test_data_list, tot="test", chunk_size=chunk_size, overlap=overlap, gmm_p=gmm_p_from_joblib, gmm_n=gmm_n_from_joblib, lab0=lab0_f, lab1=lab1_f, m=m_f, n=n_f)
t_plot = test_plotter.plot_data()

