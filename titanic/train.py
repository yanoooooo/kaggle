import chainer
from chainer import optimizers
from chainer import serializers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import Model
from sklearn.model_selection import train_test_split

# ハイパーパラメータの設定
hidden_size = 300
batch_size = 128
max_epoch = 300

def training(x_train, t_train, x_test, t_test):
    model = Model(hidden_size, 2)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train_loss, train_acc, train_acc_list = [], [], []
    test_loss, test_acc, test_acc_list = [], [], []
    for epoch in range(max_epoch):
        print('Epoch: %d' % (epoch+1))

        train_sum_accuracy, train_sum_loss = 0, 0
        # データをバッチサイズごとに使って学習
        for i in range(0, len(x_train), batch_size):
            x_batch = x_train[i:i+batch_size]
            t_batch = t_train[i:i+batch_size]
            x_batch = x_batch.values.astype(np.float32)
            t_batch = t_batch.values.astype(np.int32)

            # 勾配を初期化
            model.zerograds()
            # 順伝播させて誤差と精度を算出
            loss, acc = model.forward(x_batch, t_batch)
            # 誤差逆伝播で勾配を計算
            loss.backward()
            optimizer.update()

            train_loss.append(loss.data)
            train_acc.append(acc.data)
            train_sum_loss     += float(loss.data) * len(x_batch)
            train_sum_accuracy += float(acc.data) * len(x_batch)

        # 訓練データの誤差と、正解精度を表示
        print('train mean loss={}, accuracy={}'.format(train_sum_loss / len(x_train), train_sum_accuracy / len(x_train)))
        train_acc_list.append(train_sum_accuracy / len(x_train))

        # テストデータでの確認
        test_sum_accuracy, test_sum_loss = 0, 0
        for i in range(0, len(x_test), batch_size):
            x_batch = x_test[i:i+batch_size]
            t_batch = t_test[i:i+batch_size]
            x_batch = x_batch.values.astype(np.float32)
            t_batch = t_batch.values.astype(np.int32)

            # 順伝播させて誤差と精度を算出
            loss, acc = model.forward(x_batch, t_batch)

            test_loss.append(loss.data)
            test_acc.append(acc.data)
            test_sum_loss     += float(loss.data) * len(x_batch)
            test_sum_accuracy += float(acc.data) * len(x_batch)

        # テストデータでの誤差と、正解精度を表示
        print('test  mean loss={}, accuracy={}'.format(test_sum_loss / len(x_test), test_sum_accuracy / len(x_test)))
        test_acc_list.append(test_sum_accuracy / len(x_test))

    serializers.save_hdf5("./models/model_"+str(max_epoch), model)
    return train_acc_list, test_acc_list

if __name__ == '__main__':
    data = pd.read_csv('./data/treated_train.csv')
    train, test = train_test_split(data, test_size=0.2)
    train = train.drop('PassengerId', axis=1)
    test = test.drop('PassengerId', axis=1)
    train = train.drop('Fare', axis=1)
    test = test.drop('Fare', axis=1)

    print("-- data size --")
    print(len(train))
    print(len(test))

    t_train = train['Survived']
    x_train = train.drop('Survived', axis=1)
    t_test = test['Survived']
    x_test = test.drop('Survived', axis=1)

    train_acc_list, test_acc_list = training(x_train, t_train, x_test, t_test)

    # グラフの描画
    x = np.arange(len(train_acc_list))
    plt.plot(x, train_acc_list, marker='o', label='train')
    t = np.arange(len(test_acc_list))
    plt.plot(t, test_acc_list, marker='+', label='test')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.ylim(-0.05, 1.0)
    plt.show()
