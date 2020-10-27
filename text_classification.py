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

model.evaluate(test_x, test_y)

model.save('./model')

import random
# 加载模型
loaded_model = model.load_model('cnn_classification_model')
loaded_model.predict(random.sample(train_x, 10))

# 预测指定样本
news_sample = """「DeepMind 击败人类职业玩家的方式与他们声称的 AI 使命，以及所声称的『正确』方式完全相反。」
DeepMind 的人工智能 AlphaStar 一战成名，击败两名人类职业选手。掌声和欢呼之余，它也引起了一些质疑。在前天 DeepMind 举办的 AMA 中，AlphaStar 项目领导者 Oriol Vinyals 和 David Silver、职业玩家 LiquidTLO 与 LiquidMaNa 回答了一些疑问。不过困惑依然存在……
"""
x = list(jieba.cut(news_sample))
y = loaded_model.predict([x])
print(y[0]) # 输出游戏


