import csv
import pickle
from utils import GeneSeg
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Định nghĩa các đường dẫn và tên file
vec_dir = "file\\word2vec.pickle"
data_check_file = "data\\data_check.csv"
process_data_dir = "file\\processed_data2.pickle"

# Hàm tiền xử lý dữ liệu
def pre_process():
    # Đọc dữ liệu từ tệp word2vec.pickle
    with open(vec_dir, 'rb') as f:
        word2vec = pickle.load(f)
    dictionary, reverse_dictionary, embeddings = word2vec['dictionary'], word2vec['reverse_dictionary'], word2vec['embeddings']

    # Đọc dữ liệu từ tệp data_check.csv
    data_check = []
    with open(data_check_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            data_check.append(row[0])

    # Tách từ
    gs = GeneSeg()
    data_check = [gs.cut(sentence) for sentence in data_check]

    # Chuyển đổi từ văn bản thành dạng số
    data_check_index = []
    for sentence in data_check:
        sentence_index = []
        for word in sentence:
            if word in dictionary:
                sentence_index.append(dictionary[word])
            else:
                sentence_index.append(0)  # Sử dụng chỉ số 0 cho các từ không có trong từ điển
        data_check_index.append(sentence_index)

    # Chuẩn hóa độ dài của mỗi mẫu dữ liệu
    max_length = max(len(sentence) for sentence in data_check_index)
    data_check_index = pad_sequences(data_check_index, maxlen=max_length, padding='post')

    # Lưu trữ thông tin về kích thước dữ liệu và độ dài mẫu
    data_info = {
        'data_size': len(data_check_index),
        'max_length': max_length,
        'embedding_dim': embeddings.shape[1]
    }

    # Ghi dữ liệu đã tiền xử lý vào tệp process_data.pickle
    with open(process_data_dir, 'wb') as f:
        pickle.dump({'data': data_check_index, 'info': data_info}, f)

    print("Tiền xử lý dữ liệu hoàn thành.")

# Gọi hàm pre_process() để tiền xử lý dữ liệu
pre_process()