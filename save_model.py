

import random
import kashgari
# 加载模型
loaded_model = kashgari.utils.load_model('cnn_classification_model')
loaded_model.predict(random.sample(train_x, 10))

# 预测指定样本
news_sample = """「DeepMind 击败人类职业玩家的方式与他们声称的 AI 使命，以及所声称的『正确』方式完全相反。」
DeepMind 的人工智能 AlphaStar 一战成名，击败两名人类职业选手。掌声和欢呼之余，它也引起了一些质疑。在前天 DeepMind 举办的 AMA 中，AlphaStar 项目领导者 Oriol Vinyals 和 David Silver、职业玩家 LiquidTLO 与 LiquidMaNa 回答了一些疑问。不过困惑依然存在……
"""
x = list(jieba.cut(news_sample))
y = loaded_model.predict([x])
print(y[0]) # 输出游戏