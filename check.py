import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import GeneSeg
from keras.models import load_model
import numpy as np

# Đường dẫn và tên tệp chứa mô hình và từ điển
model_file = "G:/file/Conv_model.keras"
#model_file = "G:/file/MLP_model.h5"
#model_file = "G:/file/LSTM_model.h5"
vec_file = "G:/file/word2vec.pickle"

# Load mô hình
model = load_model(model_file)

# Load từ điển từ tệp word2vec.pickle
with open(vec_file, "rb") as f:
    word2vec = pickle.load(f)
    dictionary = word2vec["dictionary"]
    reverse_dictionary = word2vec["reverse_dictionary"]
    embeddings = word2vec["embeddings"]
# Câu đầu vào
sentence = '<img/src/onerror=prompt(8)>'
def check_xss(payload):
    # Tiền xử lý câu bằng hàm GeneSeg()
    processed_sentence = GeneSeg(sentence)
    print(processed_sentence)

    max_length = 532  
    embedding_size = 128  

    # Khởi tạo input_matrix
    input_matrix = np.full((1, max_length, embedding_size),-1)

    # Chuyển đổi từng từ trong câu thành vector nhúng
    for j in range(len(processed_sentence)):
        word = processed_sentence[j]
        if word in dictionary:
            word_index = dictionary[word]
            input_matrix[0, j, :] = embeddings[word_index]
        else:
            input_matrix[0, j, :] = embeddings[dictionary["UNK"]]  

    print(input_matrix)
    # Sử dụng input_matrix cho model.predict()
    prediction = model.predict(input_matrix)
    print(prediction[0][1])
        # Lớp 1 (index 1) thể hiện tấn công XSS, lớp 0 (index 0) thể hiện không tấn công XSS
    if prediction[0][1] > 0.5:
        print ("Payload chứa tấn công XSS")
        return 1
    else:
        print ("Payload không chứa tấn công XSS")
        return 0
    
payload = '&#14&#14>!-->script>>script>>script>>script>top[/al/.source+/ert/`1`>/script>/drfv/>/script>/drfv/>/script>/drfv/>/script>/drfv/-->'
check_xss(payload)