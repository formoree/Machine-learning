import tensorflow as tf
from tensorflow.keras import datasets,layers,optimizers,Sequential
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def preprocess(x,y):
    """进行数据处理 转换参数类型"""
    x = tf.cast(x,dtype = tf.float32)/255.
    y = tf.cast(y,dtype = tf.int32)
    return x,y

#数据获取
(x_train,y_train),(x_test,y_test) = datasets.fashion_mnist.load_data()
# x:[60000,28,28] y:[60000,]
print(x_train.shape,y_train.shape)

#处理数据
batch_size = 128
x_train = tf.reshape(x_train,[-1,28*28])
db = tf.data.Dataset.from_tensor_slices((x_train,y_train))
db = db.map(preprocess).shuffle(10000).batch(batch_size).repeat()

x_test = tf.reshape(x_test,[-1,28*28])
db_test = tf.data.Dataset.from_tensor_slices((x_test,y_test))
db_test = db_test.map(preprocess).batch(batch_size)

#网络设置
model = Sequential([
    layers.Dense(256,activation=tf.nn.relu),
    layers.Dense(128,activation=tf.nn.relu),
    layers.Dense(64,activation=tf.nn.relu),
    layers.Dense(32,activation=tf.nn.relu),
    layers.Dense(10),
])

model.build(input_shape=[None,28*28])
model.summary() #调试

optimizer = optimizers.Adam(lr=1e-3)

def main():
    """运行主函数"""
    model.compile(optimizer=optimizers.Adam(lr=0.01),
                    loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                    metircs=['accuracy'])

    # validation_freq=2表示2个epochs做一次验证
    model.fit(db, epochs=2, validation_data=db_test, validation_freq=2,steps_per_epoch=x_train.shape[0]//batch_size)

    # # 测试
    # model.evaluate(db_test)

if __name__ == "__main__":
    main()