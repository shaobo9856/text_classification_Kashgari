import tqdm
import jieba

def read_data_file(path):
    lines = open(path, 'r', encoding='utf-8').read().splitlines()
    x_list = []
    y_list = []
    for line in tqdm.tqdm(lines):
        rows = line.split('\t')
        if len(rows) >= 2:
            y_list.append(rows[0])
            x_list.append(list(jieba.cut('\t'.join(rows[1:]))))
        else:
            print(rows)
    return x_list, y_list

test_x, test_y = read_data_file('cnews/cnews.test.txt')
train_x, train_y = read_data_file('cnews/cnews.train.txt')
val_x, val_y = read_data_file('cnews/cnews.val.txt')


from tensorflow.keras.callbacks import TensorBoard
from kashgari.tasks.classification import CNN_Model

# Using TensorBoard record training process
tf_board = TensorBoard(log_dir='tf_dir/cnn_model',
                       histogram_freq=5,
                       update_freq='batch')

model = CNN_Model()
model.fit(train_x, train_y, val_x, val_y,
          batch_size=128,
          callbacks=[tf_board])