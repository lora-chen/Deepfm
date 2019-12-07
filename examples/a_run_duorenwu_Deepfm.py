#coding=utf-8
import pandas as pd
from  sklearn.metrics import log_loss, roc_auc_score
from  sklearn .model_selection import train_test_split
from  sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder,LabelBinarizer
import warnings
from deepctr.models import DeepFM,DeepFMMTL
from deepctr.inputs import  SparseFeat, DenseFeat, get_feature_names
from keras.utils import np_utils
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import keras_metrics as km
import keras
import tensorflow as tf
import pickle


def training_vis(hist,file_name):
    loss = hist.history['loss']
    # val_loss = hist.history['val_loss']
    # acc = np.sum([hist.history['output1_acc'],hist.history['output2_acc']], axis = 0)
    # val_acc = np.sum([hist.history['val_output1_acc'],hist.history['val_output2_acc']], axis=0)
    # acc = hist.history['output2_acc']
    # val_acc =  hist.history['val_output2_acc']

    total_recall =  np.sum([hist.history['output1_recall'],hist.history['output2_recall']], axis = 0)
    output1_loss = hist.history['output1_loss']
    output2_loss = hist.history['output2_loss']
    val_output1_recall = hist.history['val_output1_recall']
    val_output2_recall = hist.history['val_output2_recall']
    # val_output1_loss = hist.history['val_output1_loss']
    # val_output2_loss = hist.history['val_output2_loss']
    #
    # output1_acc = hist.history['output1_acc']
    # output2_acc = hist.history['output2_acc']
    # val_output1_acc = hist.history['val_output1_acc']
    # val_output2_acc = hist.history['val_output2_acc']

    # make a figure
    fig = plt.figure(figsize=(12,6))
    # subplot loss
    ax1 = fig.add_subplot(131)
    ax1.plot(loss,label='train_loss')
    ax1.plot(total_recall,label='total_recall')

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss and recall on total Data')
    ax1.legend()
    # subplot acc
    ax2 = fig.add_subplot(132)
    ax2.plot(output1_loss,label='output1_loss')
    ax2.plot(val_output1_recall,label='val_output1_recall')


    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Output1 loss')
    ax2.set_title('Loss and recall on output1')
    ax2.legend()
    # subplot acc
    ax3 = fig.add_subplot(133)
    ax3.plot(output2_loss, label='output2_loss')
    ax3.plot(val_output2_recall, label='val_output2_recall')

    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Output2 loss')
    ax3.set_title('Loss and recall on output2')
    ax3.legend()

    plt.tight_layout()
    plt.savefig( file_name+'.png')
    plt.show()

if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    data = pd.read_csv('./test_small.csv')
    data = shuffle(data.iloc[3:])

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
    train, test, target1_train, target1_test,target2_train,target2_test = train_test_split(data, dummy_target1,dummy_target2, test_size=0.1, random_state=0)
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
                  loss_weights={'output1': 1, 'output2': 1}, metrics=['acc',tf.keras.metrics.Recall()])
    # keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    history = model.fit(train_model_input,
                        y=[target1_train, target2_train],
                        batch_size = 512 , epochs=1, steps_per_epoch=4,validation_split=0.2)
    # convert the history.history dict to a pandas DataFrame:
    hist_df = pd.DataFrame(history.history)
    # or save to csv:
    hist_csv_file = './history_log/history.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

    # pred_ans = model.predict(test_model_input, batch_size=256)
    # score = model.evaluate(x=test_model_input,
    #                        y=[target1_test, target2_test], steps=64)

    # ['loss', 'output1_loss', 'output2_loss', 'output1_acc', 'output1_recall', 'output2_acc', 'output2_recall']
    # print('Test score:', score[0])
    # print('Test accuracy:', score[0]+score[1])
    training_vis(history,file_name='img_output')

