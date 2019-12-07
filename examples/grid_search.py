#coding=utf-8
import pandas as pd
from  sklearn.metrics import log_loss, roc_auc_score
from  sklearn .model_selection import train_test_split
from  sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder,LabelBinarizer
import warnings
import tensorflow as tf
from collections import OrderedDict
from deepctr.models import DeepFM,DeepFMMTL
from deepctr.inputs import  SparseFeat, DenseFeat, get_feature_names
from keras.utils import np_utils
import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras.metrics import Recall,Precision
import matplotlib.pyplot as plt


'''
目录已经设好
测试版本：每个先跑300个epochs
'''

def training_vis(hist,file_name,label):
    loss = hist.history['loss']
    total_recall =  np.sum([hist.history['output1_recall'],hist.history['output2_recall']], axis = 0)
    total_precision = np.sum([hist.history['output1_precision'],hist.history['output2_precision']], axis = 0)
    output1_loss = hist.history['output1_loss']
    output2_loss = hist.history['output2_loss']
    output1_precision = hist.history['output1_precision']
    output2_precision = hist.history['output2_precision']

    val_output1_recall = hist.history['val_output1_recall']
    val_output2_recall = hist.history['val_output2_recall']
    val_output1_precision = hist.history['val_output1_precision']
    val_output2_precision = hist.history['val_output2_precision']

    # make a figure
    fig = plt.figure(figsize=(12,6))
    # subplot loss
    ax1 = fig.add_subplot(131)
    ax1.plot(loss,label='train_loss')
    ax1.plot(total_recall,label='total_recall')
    ax1.plot(total_precision,label='total_precision')


    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss, precision and recall on total Data')
    ax1.legend()
    # subplot acc
    ax2 = fig.add_subplot(132)
    ax2.plot(output1_loss,label='output1_loss')
    ax2.plot(val_output1_recall,label='val_output1_recall')
    ax2.plot(output1_precision, label='output1_precision')
    ax2.plot(val_output1_precision, label='val_output1_precision')

    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Output1 loss')
    ax2.set_title('Loss , precision and recall on output1')
    ax2.legend()
    # subplot acc
    ax3 = fig.add_subplot(133)
    ax3.plot(output2_loss, label='output2_loss')
    ax3.plot(val_output2_recall, label='val_output2_recall')
    ax3.plot(output2_precision, label='output2_precision')
    ax3.plot(val_output2_precision, label='val_output2_precision')

    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Output2 loss')
    ax3.set_title('Loss, precision and recall on output2')
    ax3.legend()
    plt.tight_layout()
    plt.savefig( './history_img/history_%s.png' % str(label))


def create_model(linear_feature_columns,dnn_feature_columns,label_number,embedding_size, dnn_hidden_units,dnn_dropout,optimizer):
    model = DeepFMMTL(linear_feature_columns, dnn_feature_columns, label_number=label_number,
                      embedding_size=embedding_size,dnn_hidden_units=dnn_hidden_units,
                      dnn_dropout= dnn_dropout)
    model.compile(optimizer=optimizer,
                  loss={'output1': 'categorical_crossentropy', 'output2': 'categorical_crossentropy'}, \
                  loss_weights={'output1': 1, 'output2': 1}, metrics=['accuracy',Recall(),Precision()])
    return model


def order_dict(dicts, n):
    result = []
    result1 = []
    p = sorted([(k, v) for k, v in dicts.items()], reverse=True)
    s = set()
    for i in p:
        s.add(i[1])
    for i in sorted(s, reverse=True)[:n]:
        for j in p:
            if j[1] == i:
                result.append(j)
    for r in result:
        result1.append(r[0])
    return result1


def return_N_Max(list, N ):
    if (N <= len(list)and N>0):
        res = sorted(range(len(list)), key=lambda sub: list[sub])[-N:]
    else:
        print('Error!')
    return res



if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    data = pd.read_csv('./test_big.csv')
    data = shuffle(data.iloc[3:])

    sparse_features = ['Sparse' + str(i) for i in range(1, 348)]
    chinese_features = ['year', 'provname', 'urbcode', 'birthyear', 'marriage', 'houseowner', 'bathroom', 'familyincm',
                        'expenditure', 'worktype', 'industry']
    sparse_features = sparse_features + chinese_features

    dense_features1 = ['sex',	'eduyr','income',	'earnlvl',	'hauslvl',	'birthyear_1',	'urbcode_1',	'sex_1',	'marriage_1',	'houseowner_1',	'bathroom_1',	'education_1'	,'familyincm_1'	,'expenditure_1',	'worktype_1',	'industry_1',	'townincm',	'dispincm',	'workavgincm',	'villmean'
                       ]
    mms = MinMaxScaler(feature_range=(0, 1))

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

    dummy_target1 = np_utils.to_categorical(encoded_Y1)
    dummy_target2 = np_utils.to_categorical(encoded_Y2)

    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                           for feat in sparse_features] + [DenseFeat(feat, 1,)
                          for feat in dense_features]


    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model
    train, test, target1_train, target1_test,target2_train,target2_test = train_test_split(data, dummy_target1,dummy_target2, test_size=0.4, random_state=0)




    train_model_input = {name:train[name] for name in feature_names}
    test_model_input = {name:test[name] for name in feature_names}


    # def DeepFMMTL(linear_feature_columns, dnn_feature_columns, embedding_size=8, use_fm=True,
    #               dnn_hidden_units=(128, 128),
    #               l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024,
    #               dnn_dropout=0,
    #               dnn_activation='relu', dnn_use_bn=False, label_number=[]):

    embedding_size = [8,16]
    dnn_hidden_units = [(128, 128)]
    optimizer = ['Adam']
    # dnn_dropout = [ 0,0.1,0.2]
    dnn_dropout = [0.3]
    # dnn_dropout = [0]
    # embedding_size = [4]
    # dnn_hidden_units = [(64, 64), (128, 128)]
    # optimizer = ['SGD']
    # dnn_dropout = [0]

    log_values= []
    log_parameter= []
    train_loss = []

    len= len(embedding_size)* len(dnn_hidden_units)* len(optimizer) * len(dnn_dropout)
    index=0
    for idx_em in embedding_size:
        for idx_hidden in dnn_hidden_units:
            for idx_opti in optimizer:
                for idx_drop in dnn_dropout:
                    print("Doing %d task. Total %d tasks"  %(index, len))
                    tf.compat.v1.keras.backend.clear_session
                    tf.compat.v2.keras.backend.clear_session
                    model= create_model(linear_feature_columns, dnn_feature_columns, label_number, embedding_size=idx_em, dnn_hidden_units=idx_hidden,
                                            dnn_dropout=idx_drop, optimizer=idx_opti)
                    history = model.fit(train_model_input,
                                        y=[target1_train, target2_train],
                                        batch_size=512, epochs=300, steps_per_epoch=4, validation_split=0.2)
                    # score = model.evaluate(x=test_model_input,
                    #                        y=[target1_test, target2_test], steps=64)
                    hist_df = pd.DataFrame(history.history)
                    # save to csv:
                    label= str('embedding_'+ str(idx_em)+ ' hidden_units_'+ str(idx_hidden)+' optimizer_'+str(idx_opti) + ' dropout_' + str(idx_drop))
                    hist_csv_file = './history_log/history_%s.csv' %str(label)
                    with open(hist_csv_file, mode='w') as f:
                        hist_df.to_csv(f)
                    training_vis(history, file_name='img_output',label=label)

                    # train_loss.append(history.history['loss'])
                    # log_values.append(round(score[0],5))
                    # log_parameter.append(str('embedding ='+ str(idx_em)+ ' hidden_units='+ str(idx_hidden)+' optimizer:'+str(idx_opti) + ' dropout:' + str(idx_drop)))


                    del model
    # np.savetxt("train_loss.txt", train_loss[-1])
    # np.savetxt("train_loss11.txt", train_loss)
    # np.savetxt("test_loss.txt", log_values)
    # ['loss', 'output1_loss', 'output2_loss', 'output1_acc', 'output2_acc']
    top_N = 3


    # temp = return_N_Max(log_values,top_N)
    # for ii in range(len(temp)):
    #     print('The top'+ str(top_N - ii) + 'parameter is: '+ str(log_parameter[temp[ii]]) + ', and the final loss is: '+ str(log_values[temp[ii]])  )

