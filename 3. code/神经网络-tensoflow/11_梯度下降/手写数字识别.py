import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets,layers,optimizers,Sequential,metrics

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
db = tf.data.Dataset.from_tensor_slices((x_train,y_train))
db = db.map(preprocess).shuffle(60000).batch(batch_size)

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
    for epoch in range(10):
        for step,(x,y) in enumerate(db):

            x = tf.reshape(x,[-1,28*28])
            #triain
            with tf.GradientTape() as tape:

                logits = model(x)
                y_onehot = tf.one_hot(y,depth=10)

                loss_mse = tf.reduce_mean(tf.losses.MSE(y_onehot,logits))
                loss_cs = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot,
                                                            logits,from_logits=True))

            grads = tape.gradient(loss_cs,model.trainable_variables)
            optimizer.apply_gradients(zip(grads,model.trainable_variables))

            #### 留一个疑问 为什么这里的step可以到达500？
            if step % 100 == 0:
                print(epoch,step,f'loss:{float(loss_cs),float(loss_mse)}')

        # test
        total_correct = 0
        total_num = 0
        for x, y in db_test:
            x = tf.reshape(x,[-1,28*28])
            logits = model(x)

            prob = tf.nn.softmax(logits,axis=1)
            pred = tf.argmax(prob,axis=1)
            pred = tf.cast(pred,dtype=tf.int32)

            correct = tf.equal(pred,y)
            correct = tf.reduce_sum(tf.cast(correct,dtype=tf.int32))

            total_correct += int(correct)
            total_num += x.shape[0]

        acc = total_correct / total_num
        print(epoch,f'test acc:{acc}')

if __name__ == "__main__":
    main()