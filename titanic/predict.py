import chainer
from chainer import serializers
import pandas as pd
import numpy as np
from model import Model

hidden_size = 300
modelname = 'models/model_300'

def predict(data):
    model = Model(hidden_size, 2)
    serializers.load_hdf5(modelname, model)

    predict_list = []
    for i in range(0, len(data)):
        x = data[i:i+1]
        passenger_id = x['PassengerId'].tolist()
        x = x.drop('PassengerId', axis=1)
        x = x.values.astype(np.float32)

        prd = model.forward(x, np.array([]), False)
        predict_list.append((passenger_id[0], prd.argmax()))

    return predict_list


if __name__ == '__main__':
    data = pd.read_csv('./data/treated_test.csv')
    # train = train.drop('PassengerId', axis=1)
    # test = test.drop('PassengerId', axis=1)
    data = data.drop('Fare', axis=1)

    print("-- data size --")
    print(len(data))

    predict_list = predict(data)

    filename = 'predict.txt'
    file = open(filename, 'w')
    file.write('PassengerId,Survived\n')
    for index, line in enumerate(predict_list):
        if index+1 != len(predict_list):
            file.write(str(line[0]) +','+ str(line[1]) +'\n')
        else:
            file.write(str(line[0]) +','+ str(line[1]))
    file.close()
