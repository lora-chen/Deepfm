#coding=utf-8
import pandas as pd
from  sklearn.metrics import log_loss, roc_auc_score
from  sklearn .model_selection import train_test_split
from  sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder,LabelBinarizer
import warnings
from deepctr.models import DeepFM,DeepFMMTL
from deepctr.inputs import  SparseFeat, DenseFeat, get_feature_names
import scipy as sp
from keras.utils import np_utils
import numpy as np

import matplotlib.pyplot as plt

def llfun(act, pred,idx):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred[idx])
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act[idx]*sp.log(pred) + sp.subtract(1,act[idx])*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act[idx])
    return ll


def training_vis(hist,file_name):
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    # acc = np.sum([hist.history['output1_acc'],hist.history['output2_acc']], axis = 0)
    # val_acc = np.sum([hist.history['val_output1_acc'],hist.history['val_output2_acc']], axis=0)
    acc = hist.history['output2_acc']
    val_acc =  hist.history['val_output2_acc']


    # output1_loss = hist.history['output1_loss']
    output2_loss = hist.history['output2_loss']
    # val_output1_loss = hist.history['val_output1_loss']
    val_output2_loss = hist.history['val_output2_loss']

    # output1_acc = hist.history['output1_acc']
    output2_acc = hist.history['output2_acc']
    # val_output1_acc = hist.history['val_output1_acc']
    val_output2_acc = hist.history['val_output2_acc']

    # make a figure
    fig = plt.figure(figsize=(12,6))
    # subplot loss
    ax1 = fig.add_subplot(121)
    ax1.plot(loss,label='train_loss')
    ax1.plot(val_loss,label='val_loss')

    # ax1.plot(output1_loss,label='output1.loss')
    ax1.plot(output2_loss, label='output2.loss')
    # ax1.plot(val_output1_loss, label='val_output1.loss')
    ax1.plot(val_output2_loss, label='val_output2.loss')

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss on Training and Validation Data')
    ax1.legend()
    # subplot acc
    ax2 = fig.add_subplot(122)
    ax2.plot(acc,label='train_acc')
    ax2.plot(val_acc,label='val_acc')

    # ax2.plot(val_output1_acc,label = 'val_output1_acc')
    ax2.plot(val_output2_acc, label='val_output2_acc')
    # ax2.plot(output1_acc,label='output1_acc')
    ax2.plot(output2_acc, label='output2_acc')

    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy  on Training and Validation Data')
    ax2.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    use_plot = True

    warnings.filterwarnings("ignore")
    data = pd.read_csv('./test1.csv')

    sparse_features = ['Sparse' + str(i) for i in range(1, 348)]
    chinese_features = ['year', 'provname', 'urbcode', 'birthyear', 'marriage', 'houseowner', 'bathroom', 'familyincm',
                        'expenditure', 'worktype', 'industry']
    sparse_features = sparse_features + chinese_features

    dense_features1 = ['sex',	'eduyr','income',	'earnlvl',	'hauslvl',	'birthyear_1',	'urbcode_1',	'sex_1',	'marriage_1',	'houseowner_1',	'bathroom_1',	'education_1'	,'familyincm_1'	,'expenditure_1',	'worktype_1',	'industry_1',	'townincm',	'dispincm',	'workavgincm',	'villmean'
                       ]
    mms = MinMaxScaler(feature_range=(0, 1))
    # lb = LabelBinarizer()

    dense_features = dense_features1
    data[dense_features] = mms.fit_transform(data[dense_features])


    target_list = ['education','faminlvl']
    for ii in range( 1,len(target_list)+1):
        name = 'target' + str(ii)
        locals() ['target' + str(ii) ]= target_list[ii - 1]

    target1 = 'education'
    target2 = 'faminlvl'
    label_feature_number1 = len(data[target1].unique())
    label_feature_number2 = len(data[target2].unique())

    label_number = []
    label_number.append(label_feature_number1)
    label_number.append(label_feature_number2)

    Y1 = data[target1].values
    Y2 = data[target2].values
    encoder = LabelEncoder()
    encoded_Y1 = encoder.fit_transform(Y1)
    encoded_Y2 = encoder.fit_transform(Y2)
    # convert integers to dummy variables (one hot encoding)
    dummy_target1 = np_utils.to_categorical(encoded_Y1)
    dummy_target2 = np_utils.to_categorical(encoded_Y2)

    # data[target] = lb.fit_transform(data[target])

    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                           for feat in sparse_features] + [DenseFeat(feat, 1,)
                          for feat in dense_features]


    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model
    train, test, target1_train, target1_test,target2_train,target2_test = train_test_split(data, dummy_target1,dummy_target2, test_size=0.4, random_state=0)
    # train, test = train_test_split(data, test_size=0.2)



    train_model_input = {name:train[name] for name in feature_names}
    test_model_input = {name:test[name] for name in feature_names}
    # print(type(train_model_input))

    # 4.Define Model,train,predict and evaluate
    model = DeepFMMTL(linear_feature_columns, dnn_feature_columns,label_number = label_number)
    # compile(self, optimizer, loss, metrics=None, loss_weights=None, sample_weight_mode=None, weighted_metrics=None,
    #         target_tensors=None)
    # model.compile(optimizer='adam',loss={'output1': 'sparse_categorical_crossentropy', 'output2': 'sparse_categorical_crossentropy'}, \
    #               loss_weights={'output1': 1, 'output2': 1},metrics=['acc'])
    model.compile(optimizer='adam',
                  loss={'output1': 'categorical_crossentropy', 'output2': 'categorical_crossentropy'}, \
                  loss_weights={'output1': 0, 'output2': 1}, metrics=['accuracy'])

    # model.compile("adam", "binary_crossentropy",
    #               metrics=['binary_crossentropy'], )
    # record = LossHistory()
    history = model.fit(train_model_input,
                        y=[target1_train, target2_train],
                        batch_size = 512 , epochs=300, steps_per_epoch=4,validation_split=0.2)

    # pred_ans = model.predict(test_model_input, batch_size=256)
    score = model.evaluate(x=test_model_input,
                           y=[target1_test, target2_test], steps=64)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    training_vis(history,file_name='img_output')
    # record.loss_plot('epoch')

    # print("test LogLoss", round(log_loss(test[target1].values, pred_ans), 4))
    # acc_score1 = []
    # acc_score2 = []
    # acc_scores = []
    # for index in range(0, len(pred_ans)):
    #     result1 = llfun(target1_test, pred_ans[0], index)
    #     result2 = llfun(target2_test, pred_ans[1], index)
    #     acc_score1.append(result1)
    #     acc_score2.append(result2)
    #     acc_scores.append(result1)
    #     acc_scores.append(result2)
    # print("task1 loss:", int(sum(acc_score1) / len(acc_score1)))
    # print("task2 loss",int(sum(acc_score2) / len(acc_score2)))
    # print("Total loss", int(sum(acc_scores) / len(acc_scores)))
    # print("test LogLoss", round(log_loss(test[target1].values, pred_ans), 4))
