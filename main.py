import CNN_model
import glob
import filter
import read

chunk_size=300
overlap=0

file_positive_green = glob.glob(r"C:\Users\user\PycharmProjects\Emotion Classification 2\P\green\*.txt")
file_negative_green=glob.glob(r"C:\Users\user\PycharmProjects\Emotion Classification 2\N\green\*.txt")

h1=glob.glob("C:/Users/user/PycharmProjects/Emotion Classification 2/plotting_dataset/h1.txt")
g1=glob.glob("C:/Users/user/PycharmProjects/Emotion Classification 2/plotting_dataset/g1.txt")

train_positive_reader=read.read(data=file_positive_green, y=0, chunk_size=chunk_size, overlap=overlap)
train_negative_reader=read.read(data=file_negative_green, y=1, chunk_size=chunk_size, overlap=overlap)
p_train_list=train_positive_reader.read_txt_files_with_skip()
n_train_list=train_negative_reader.read_txt_files_with_skip()
train_list=p_train_list+n_train_list

train_filtered=filter.preprocessing(data=train_list, chunk_size=chunk_size,train_or_test="train", overlap=overlap)
x_train, x_test, y_train, y_test=train_filtered.GMM_model(data=train_filtered, tot="train")
training=CNN_model.model(x_train, x_test, y_train, y_test)
history, predictions, score=training.build_model()

test_data=read.read(h1, 1, chunk_size=chunk_size, overlap=overlap)
test_data_list=test_data.read_txt_files_with_skip()
test_filtered=filter.preprocessing(data=test_data_list, chunk_size=chunk_size, train_or_test="test",overlap=overlap)
x_test, y_test=test_filtered.GMM_model(data=test_filtered, tot="test")
print(x_test)
print(y_test)

