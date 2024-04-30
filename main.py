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
x_train, x_test, y_train, y_test=train_filtered.GMM_model()
training=CNN_model.model(x_train, x_test, y_train, y_test)
history, predictions, score=training.build_model()

# 대문자로 놓으면 CONSTANT로 됨.
# 데이터를 이제 많이 얻을 예정
# 알고리즘은 고정되었으니 실험 계획을 정말 제대로 얻어야 함.
# 논문에 들어갈 실험 데이터를 얻어야 함.
# 현재 test가 많이 빈약한 상태. 많이 채워넣어야 함.
# 다양한 상황의 데이터를 많이 얻어야 함.

test_data=read.read(h1, 1, chunk_size=chunk_size, overlap=overlap)
test_data_list=test_data.read_txt_files_with_skip()
test_filtered=filter.preprocessing(data=test_data_list, chunk_size=chunk_size, train_or_test="test",overlap=overlap)
x_test, y_test=test_filtered.GMM_model()
print(x_test)
print(y_test)

