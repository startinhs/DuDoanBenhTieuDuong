"""
# Tensorflow và Keras là gì? (opensource) 
Dùng để xây dựng và huấn luyện mô hình học sâu.
    KR: high-level (frontend), đơn giản
    TF: low-level  (backend), ít đơn giản hơn KR
    chạy cpu, gpu (gpu trước, k có thì chạy cpu)
//Cac khai niem: Keras:
Sequential Layers: xdung mô hình đơn giản, không có sự chia nhánh hoặc hợp nhất các lớp.
Activation Function: giới thiệu phi tuyến tính vào mô hình, chọn relu cho các lớp ẩn và softmax cho lớp đầu ra phân loại.
Fit, Loss: huấn luyện mô hình, xác định hàm mất mát cần tối thiểu hóa.
Optimizer:
SGD: kiểm soát tốt tốc độ học và có thời gian huấn luyện dài.
Adam: cần tốc độ học tự động điều chỉnh và huấn luyện nhanh hơn.
Evaluation: đánh giá hiệu suất mô hình trên dữ liệu kiểm tra sau khi huấn luyện.
//Build:
Load dữ liệu: Nạp dữ liệu cần thiết vào chương trình.
Xây dựng model: Thiết kế cấu trúc mô hình mạng nơ-ron.
Compile model: Biên dịch mô hình, xác định hàm mất mát và bộ tối ưu hóa.
Train model: Huấn luyện mô hình với dữ liệu đã cho.
Đánh giá model: Đánh giá hiệu suất mô hình trên dữ liệu kiểm tra.
Dự đoán dữ liệu mới: Sử dụng mô hình để dự đoán trên dữ liệu mới.
Lưu model: Lưu mô hình đã huấn luyện vào file.
Load model: Nạp mô hình đã lưu để sử dụng hoặc đánh giá lại.

# Mạng Neural là gì? (quan trọng, dl opensource, tf,..)
    - Nhiều layer liên tiếp nhau
    //ex:
    in layer         out layer
    x1 -
    x2 - - (w1,w2,w3) - x1'  => x1' = x1*w1+x2*w2+x3*w3
    x3 -
    (3 nơ ron)          (1)

    bao nhieu thong so input, output thì có bao nhiêu nơ ron


# Các khái niệm trong Keras: model, layers, loss, optimizer..
# Các bước để xây dựng một mạng NN cơ bản bằng Keras


Nạp dữ liệu:
    DATASET: Train, Validation (Test lại xem Train ok không), Test (so với data thực tế với data mô hình)
"""
from keras import Sequential
from keras.src.layers import Dense
from keras.src.saving import load_model
import numpy
#1. Load dữ liệu và chia Train, Val, Test (dung sklearn de chia)
from numpy import loadtxt
from sklearn.model_selection import train_test_split

dataset = loadtxt('pima-indians-diabetes.data.csv', delimiter=',')
print(dataset)
z
x = dataset[:,0:8] #0->7: 8 Phan tu input
y = dataset[:,8] # 1 phan tu output(true/false)

x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.2) #20% test, 80% train,val
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.2)

###Train model:
# #input 8: 16 neural -> 8 neural -> 1 neural(out)
# model = Sequential()
# model.add(Dense(16, input_dim=8, activation='relu')) #relu: tính chất mạng nơ ron, nếu k chỉ là hàm tuyến tính thường
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='sigmoid')) #sigmoid: hàm đồ thị. >0.5 là có, <0.5 là không
# model.summary()
#
# #Complile model: Loss,Optimizer, Metric
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# #Train model: Epoch(so lan duyet cac ptu qua tap train), Batch_size(so luong ptu da chon trong train va dua vao model), Validation
# model.fit(x_train,y_train,epochs=100,batch_size=8,validation_data=(x_val,y_val))
#
# model.save("myModel.h5")

#cacs tham số của epochs,.. là các siêu tham số


model = load_model("myModel.h5")

loss, acc = model.evaluate(x_test, y_test)
print("Loss = ",loss)
print("Acc = ",acc)

x_new = x_test[10] #ptu thu 10 cua input
y_new = y_test[10]
x_new = numpy.expand_dims(x_new, axis=0) #them chieu
y_predict = model.predict(x_new)
result = "Co tieu duong (1)"
if y_predict <= 0.5:
    result = "Khong tieu duong (0)"
print("Gia tri du doan = ",y_predict," | ", result) #>0,5 la co, <0,5 la khong
print("Gia tri thuc = ",y_new) #out



# vấn đề khác tự tìm hiểu: checkpoint, đánh giá model Precision, recall, f1 score,..., loss, accuracy