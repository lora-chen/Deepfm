# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import confusion_matrix
from imblearn.metrics import specificity_score, sensitivity_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.python.keras.utils.np_utils import to_categorical
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from deepctr.models import DeepFM, xDeepFM, CCPM, FNN, PNN, WDL, NFM, AFM, DCN, AutoInt, NFFM, FGCNN, FiBiNET
from deepctr.inputs import SparseFeat, DenseFeat,get_feature_names
import warnings
import matplotlib.ticker as ticker
warnings.filterwarnings("ignore")


def draw(list, ziel, name, logloss, auc):
    clr = ['r','g','b','orange']
    clr = ['#29A2C6', 'r', '#73B66B', '#FF6D31']
    label = ['PI-2', 'FI-2', 'FC-2', 'EL-2']
    markers = ['o','*','.','+']
    markers = ['o', 'o', 'o', 'o']
    x = list
    y = logloss
    for i in range(len(ziel)):
        plt.figure(figsize=(6, 4))  # 创建绘图对象
        plt.plot(x, y[i], clr[i],marker= markers[i], ms = 10, linewidth=4)  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
        #plt.xlabel(name)  # X轴标签
        #plt.ylabel('Log_Loss')  # Y轴标签
        #plt.title(ziel[i])  # 图标题

        plt.grid(axis="y")
        plt.tick_params(labelsize=16)
        plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
        plt.savefig("../pic/%s.png" % (name + '_'+ziel[i]))
        #plt.show()

    '''
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    l=[0]*len(logloss)
    for i in range(len(logloss)):
        y = logloss[i]
        l[i] = ax1.plot(x, y, clr[i], marker= markers[i],ms = 10,linewidth=4)[0]
        ax1.set_title("LogLoss",loc='center')
    ax1.grid(axis="y")

    for i in range(len(auc)):
        y = auc[i]
        ax2.plot(x, y, clr[i], marker= markers[i], ms = 10, linewidth=4)
        ax2.set_title("AUC",loc='center')
    ax2.grid(axis="y")

    ax2.legend(l,  # The line objects
               labels=label,  # The labels for each line
               loc="center right",  # Position of legend
               borderaxespad=0.1,  # Small spacing around legend box
               title="",  # Title for the legend
               bbox_to_anchor = (1.22,0.5),#1.25 0.5 -- 10*4
               fontsize = 15)
    fig.tight_layout()
    fig.suptitle(name, y=0.98,x = 0.48,size=15)
    #plt.savefig("../pic/%s.png"%(name+''))  # 保存图
    #plt.subplots_adjust(right=100)
    plt.subplots_adjust(left=0.1, right=0.9, top=1, bottom=0.1)
    '''
    plt.show()


# 不平衡处理 不能用
def over_sampling(X_train0, y_train0, over_samp0):
    if over_samp0 == 1:
        # SMOTEENN / SMOTETomek 过采样+欠采样
        sm = SMOTETomek()
        X_resampled, y_resampled = sm.fit_sample(X_train0, y_train0)
    if over_samp0 == 2:
        # Smote 过采样
        sm = SMOTE() #kind='svm'
        X_resampled, y_resampled = sm.fit_sample(X_train0, y_train0)
    if over_samp0 == 3:
        # 随机过采样
        ros=RandomOverSampler(random_state=42) #random_state=42
        X_resampled, y_resampled=ros.fit_sample(X_train0, y_train0)
    if over_samp0 != 0:
        X_train0 = X_resampled
        y_train0 = y_resampled
    return (X_train0, y_train0)


def model_gridsearch(lfc,dfc,grid):
    model_names = []
    models = []
    if grid == 'embedding_size':
        # embedding_size = [2, 4, 6, 8, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120, 160, 200]
        embedding_size = [ 25, 30, 40, 50]
        xlabel = embedding_size
    elif grid == 'dnn_hidden_units':
        # dnn_hidden_units = [4, 5, 6, 7, 8, 9, 10]
        dnn_hidden_units = [ 7,8, 9,10]
        xlabel = dnn_hidden_units
    elif grid == 'dnn_hidden_units_len':
        # dnn_hidden_units_len = [2, 3, 4, 5, 6, 7, 8]
        dnn_hidden_units_len = [2, 3,4,5]
        xlabel = dnn_hidden_units_len
    elif grid == 'dnn_dropout':
        # dnn_dropout = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        dnn_dropout = [0, 0.1, 0.2,0.3]
        xlabel= dnn_dropout
    else:
        models = [DeepFM(lfc, dfc, task='binary')]
        model_names = ['DeepFM']
        xlabel = []
        return (models, model_names,xlabel)
    for a in xlabel:  # dnn_hidden_units dnn_dropout embedding_size
        if grid == 'embedding_size':
            models.append(DeepFM(lfc, dfc, embedding_size=a, task='binary', seed=1024) )
        elif grid == 'dnn_hidden_units':
            models.append(DeepFM(lfc, dfc, dnn_hidden_units=(2**a, 2**a), task='binary', seed=1024) )
        elif grid == 'dnn_hidden_units_len':
            models.append(DeepFM(lfc, dfc, dnn_hidden_units = (128,) * a, task='binary', seed=1024) )
        elif grid == 'dnn_dropout':
            models.append(DeepFM(lfc, dfc, dnn_dropout= a, task='binary', seed=1024) )
        model_names.append(str(a) + ' '+ grid)

    return (models, model_names,xlabel)
    '''
    
    models = [#xDeepFM(linear_feature_columns, dnn_feature_columns, task='binary'),
              DeepFM(linear_feature_columns, dnn_feature_columns, embedding_size=8,task='binary'),
        #            linear_feature_columns, dnn_feature_columns, embedding_size=8, use_fm=True, dnn_hidden_units=(128, 128),
        #            l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024, dnn_dropout=0,
        #            dnn_activation='relu', dnn_use_bn=False, task='binary'

              #CCPM(linear_feature_columns, dnn_feature_columns, task='binary'),
              #FNN(linear_feature_columns, dnn_feature_columns, task='binary'),
              #PNN(dnn_feature_columns, task='binary'),
              #WDL(linear_feature_columns, dnn_feature_columns, task='binary'),
              #NFM(linear_feature_columns, dnn_feature_columns, task='binary'),
              #AFM(linear_feature_columns, dnn_feature_columns, task='binary'),
              #DCN(dnn_feature_columns, task='binary'),
              #AutoInt(dnn_feature_columns, task='binary'),
              #NFFM(linear_feature_columns, dnn_feature_columns, task='binary'),
              #FGCNN(dnn_feature_columns, task='binary'),
              #FiBiNET(linear_feature_columns, dnn_feature_columns, task='binary')
              ]
              '''
    '''
    model_names = [
        #'xDeepFM',
        'DeepFM',
        # 'CCPM',
        #'FNN', 'PNN', 'WDL',
        # #'NFM', 'AFM',
        #'DCN',
        #'AutoInt', 'NFFM', 'FGCNN',
        #'FiBiNET'
        ] '''

def run(data, ziel, line0, grid , loop):
    poi_feature_transfer = []
    print('++++', '\n', grid)
    for a in range(len(poi_feature)):
        poi_feature_transfer.append('poi_feature_%d'%a)
        data = data.rename(columns={poi_feature[a]: 'poi_feature_%d'%a})

    features = ['provname', 'prefname', 'cntyname', 'townname', 'villname','dispincm', 'urbcode_1', 'hauslvl']  + poi_feature_transfer#
    sparse_features = []
    dense_features = []
    for f in features:
        if f not in x_category or x_category[f] == 1:
            dense_features.append(f)
        else:
            sparse_features.append(f)
    data[sparse_features] = data[sparse_features].fillna(-1)
    data[dense_features] = data[dense_features].fillna(0 )

    y=[]
    #ziel =  # villmean, income
    y_limit= [np.min(data[ziel])-1]+ line0 +[np.max(data[ziel])]
    for index, row in data.iterrows():
        for i in range(1, len(y_limit)):
            if y_limit[i - 1] < row[ziel] <= y_limit[i]:
                y.append(i-1)
                break
    data['income_0'] = y
    target = ['income_0']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name
    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique()) for feat in sparse_features] + \
                             [DenseFeat(feat, 1,)for feat in dense_features]
    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns
    fixlen_feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model
    train, test = train_test_split(data, test_size=0.2)

    # try to oversampling
   # (train_x,train_y)=over_sampling(train[features],train[ziel], 3)
   # train = (np.column_stack((train_x, train_y)))
    train_model_input = [train[name] for name in fixlen_feature_names]
    test_model_input = [test[name] for name in fixlen_feature_names]
# 4.Define Model,train,predict and evaluate ##############################################
    (models, model_names,xlabel) = model_gridsearch(linear_feature_columns,dnn_feature_columns,grid)
    logloss, auc1, acc1, pre1, recall1,f11 = [],[],[],[],[],[]
    print(ziel, line0, len(data))
    for name,model in zip(model_names,models):
        ll_avg, auc_avg = [],[]
        for i in range(loop):
            model.compile("adam",'binary_crossentropy',
                          metrics=['binary_crossentropy'])
            history = model.fit(train_model_input, train[target].values,
                                batch_size=256, epochs=10, verbose=0, validation_split=0.2, )
            pred_ans = model.predict(test_model_input, batch_size=256)

            true = test[target].values
            '''
            f = open("pred.csv", 'a', encoding='utf_8_sig')
            f.write('%s\n'%(ziel))
            for i in range(len(pred_ans)):
                f.write('%s, %s\n' % (pred_ans[i],true[i] ))
            f.close()'''


            ll = round(log_loss(test[target].values, pred_ans), 4)
            auc = round(roc_auc_score(test[target].values, pred_ans), 4)
            #acc = round(accuracy_score(test[target].values, pred_ans.round()), 4)
            #pre = round(precision_score(test[target].values, pred_ans.round()), 4)
            #recall = round(recall_score(test[target].values, pred_ans.round()), 4)
            #f1 = round(f1_score(test[target].values, pred_ans.round(), average='weighted'),4)
            #spec = round(specificity_score(test[target].values, pred_ans.round(), average='weighted'),4)
            #sens = round(sensitivity_score(test[target].values, pred_ans.round(), average='weighted'),4)
            print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
            print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
            ll_avg.append(ll), auc_avg.append(auc)
        logloss.append(np.mean(ll_avg)), auc1.append(np.mean(auc_avg))#, acc1.append(acc), pre1.append(pre), recall1.append(recall), f11.append(f1)

        '''
        cm = confusion_matrix(test[target].values, pred_ans.round())
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = []
        for m in range(len(line0)+1):
            cm.append([])
            for n in range(len(line0)+1):
                cm[m].append(round(cm_normalized[m][n],4))
        '''
        '''
        print(name)
        print("LogLoss", ll, end=' ')
        print("AUC", auc, end=' ')
        print("accuracy", acc, end=' ')
        #print("precision" , pre, end=' ')
        #print("recall", recall, end=' ')
        print("f1" , f1, end=' ')
        print("spec", spec, end=' ')
        print("sens" , sens, end=' ')
        print(cm)
        #f = open("DeepFM.csv", 'a', encoding='utf_8_sig')
        #f.write('%s,%s\n'%(ziel,line0))
        #f.write('%s, %s, %s, %s, %s, %s, %s,' % (name, ll, auc, acc, f1, spec, sens))
        #f.write('%s\n' % str(cm).replace(',',';'))
        #f.close()
        '''
    return (logloss, auc1, xlabel)

############################################

# 特征属性（dense_features：1， sparse_features：0， 不在这里的属性 = dense_features）
x_category = {'eduyr': 1,
              'provname': 0,	'prefname': 0,	'cntyname': 0,	'townname': 0,	'villname': 0,
              'worktype': 0, 'industry': 0, 'ses2':0,
              'birthyear': 1, 'sex': 0, 'sex_1': 0, 'marriage': 0,
              'hauslvl': 1, 'houseowner': 0, 'bathroom': 0,
              'urbcode': 0, 'urbcode_1': 0, 'huji':0, 'expenditure': 1,
              'townincm': 1, 'dispincm': 1, 'workavgincm': 1}  # 'education_1', 和 eduyr对应的

poi_feature = ['行政地标', '村庄', '房地产', '政府机构', '住宅区', '购物', '教育培训', '公司企业', '行政单位', '幼儿园'  # > 300
               , '各级政府', '公司', '医疗', '金融', '汽车服务', '其他', '小学', '酒店', '超市', '写字楼', '内部楼栋', '商铺', '银行' # >150
               , '家居建材', '美食', '交通设施', '公检法机构', '厂矿', '中餐厅', '旅游景点', '中学', '出入口', '门' # >100
               , '园区', '汽车维修', '诊所', '乡镇', '桥', '家电数码', '休闲娱乐', '购物中心', '汽车销售', '综合医院', '信用社' #>50
                ]
poi_feature2 = [ '培训机构', '快捷酒店', '市场', '文物古迹', '汽车美容', '自然地物', '山峰', '公园', '四星级' #>20
               , '洗浴按摩', '专科医院', '长途汽车站', '休闲广场', '高等院校', '农林园艺', '宿舍', '运动健身', '生活服务', '教堂'
               , '居民委员会', '路侧停车位', '福利机构', '风景区', '景点', '疗养院', '体育场馆', '电影院', '港口', '投资理财'   #>10
               , '火车站', '文化传媒', '三星级', '农家院', '科研机构', '社会团体', '汽车配件', '游乐园', '充电站', '星级酒店'
               , '图书馆', '剧院', '博物馆', '展览馆', '公寓式酒店', 'ktv', '疾控中心' #>5
               , '邮局', '外国餐厅', '党派团体', '水系', '文化宫', '度假村', '汽车检测场', '咖啡厅', '物流公司', '房产中介机构', '健身中心', '政治教育机构', '汽车租赁'
               , '内部楼号', '急救中心', '服务区', '美术馆', '涉外机构', '殡葬服务', '五星级', '亲子教育', '行政区划', '典当行', '植物园', '公用事业', '医疗保健'
               ]
#poi_feature += poi_feature2
poi_feature_1=[]
for p in poi_feature:
    poi_feature_1.append('p_'+p)
poi_feature += poi_feature_1

df0 = pd.read_csv('./selected_data.csv',delimiter="\t")
df0 = df0.fillna(-1)
df0 = df0[df0['houseowner'] == '自家（包括父母家）自有住房']
#df0 = df0[df0['prefname'] == '哈尔滨市']
# 目前只用了部分POI，全部POI在第100行去掉注释
ziel=['income', 'faminlvl', 'expenditure_1','eduyr']  #  'eduyr',, 'faminlvl', 'expenditure_1', 'eduyr'
line = [[2000], [4], [1], [12]] #, [14],


grids = ['dnn_hidden_units_len','dnn_hidden_units','dnn_dropout','embedding_size']
#metric = 'auc'
loop = 1


for grid in grids:
    logloss, auc = [], []
    for i in range(len(ziel)):
        if ziel[i] == 'eduyr':
            df = df0[(df0['eduyr']>0)]
        elif ziel[i] == 'expenditure_1':
            df = df0[(df0['expenditure_1']!=8) & (df0['expenditure_1']!=9)]
        elif ziel[i] == 'worktype_1':
            df = df0[(df0['worktype_1']!=12) & (df0['worktype_1']!=13)]
        elif ziel[i] == 'income':
            df = df0[(df0['income']>=50)]
        elif ziel[i] == 'faminlvl':
            df = df0[(df0['faminlvl']>0)]
        (ll,a,xlabel) = run(df, ziel[i], line[i],grid ,loop)
        # print("This is  logloss", ll,"\n")
        #
        # print("This is AUC",a)

        logloss.append(ll)
        auc.append(a)
    # print(logloss)
    # print(auc)

    '''
    f = open("zuotu.txt", 'a', encoding='utf_8_sig')
    f.write('%s\n%s\n' % ( logloss, auc))
    f.close()
    print()
    #draw(xlabel, grid, logloss, auc)



embedding_size = [2, 4, 6, 8, 10, 15, 20, 25, 30, 40, 50]#, 60, 80, 100, 120, 160, 200
embedding_size2=['2','4','6','8','10','15','20','25','30','40','50','60','80','100','120','160','200']
dnn_hidden_units = ['16', '32', '64', '128', '256', '512', '1024']
dnn_hidden_units_len = ['2','3','4','5','6','7','8']
dnn_dropout = ['0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8']

xlabel = [embedding_size,dnn_dropout,dnn_hidden_units,dnn_hidden_units_len]
xlabel2=['embedding_size','dnn_dropout','dnn_hidden_units','dnn_layers']

ll=[0,0,0,0]
auc = [0,0,0,0]
(ll[0],auc[0]) = \
    (
[[0.5031,0.5001,0.5015,0.5036,0.5059,0.5080,0.5061,0.5072,0.5055,0.5077,0.5061],#,0.5047,0.5060,0.5090,0.5051,0.5073,0.5102  1《》2
[0.5789,0.5780,0.5762,0.5779,0.5747,0.5830,0.5771,0.5781,0.5764,0.5756,0.5849],#,0.5848,0.5819,0.5743,0.5805,0.5764,0.5783
[0.5821,0.5806,0.5808,0.5811,0.5816,0.5817,0.5813,0.5809,0.5823,0.5812,0.5836],#,0.5838,0.5822,0.5817,0.5842,0.5849,0.5811
[0.4690,0.4682,0.4645,0.4813,0.4674,0.4704,0.4679,0.4677,0.4670,0.4677,0.4751]]#,0.4682,0.4798,0.4723,0.4690,0.4643,0.4680

,[]
   )
(ll[1],auc[1]) = \
    (
[[0.5277,0.5139,0.5078,0.5001,0.5269,0.5238,0.5196,0.5201,0.5199],# 6 7 8 -》 1 2 3
[0.5797,0.5804,0.5786,0.5764,0.5750,0.5743,0.5791,0.5782,0.5871],# 5-》4/ 6 7 8 -》 3 4 5
[0.5880,0.5852,0.5847,0.5832,0.5806,0.5848,0.5852,0.5860,0.5866],#6 7 8-》2 3 4 / yuan4 5 3 2 -》 5 6 7 8
[0.4655,0.4643,0.4690,0.4648,0.4696,0.4655,0.4769,0.4702,0.4765]]

,[]
   )
(ll[2],auc[2]) = \
    (
[[0.5008,0.5001,0.5037,0.5049,0.5042,0.5053,0.5091],
[0.5762,0.5743,0.5758,0.5788,0.5755,0.5807,0.5777], # 1《>2
[0.5824,0.5806,0.5810,0.5831,0.5828,0.5810,0.5830],
[0.4677,0.4643,0.4754,0.4736,0.4778,0.4704,0.4762]] #1《》2

,[]
   )
(ll[3],auc[3]) = \
    (
[[0.5080,0.5001,0.5167,0.5069,0.5099,0.5060,0.5062],
[0.5772,0.5743,0.5751,0.5755,0.5752,0.5751,0.5842], # 原4=3=0.5751 +了一些， 7《》8
[0.5939,0.5858,0.5817,0.5806,0.5834,0.5855,0.5880],#8《》4
[0.4695,0.4695,0.4643,0.4678,0.4741,0.4761,0.4788]] # 8 《-》7 / 7 8 -》 3 4

,[]
   )
for i in range(4):
    draw(xlabel[i], ['PI_2', 'FI_2', 'FC_2','EL_2'],xlabel2[i], ll[i],auc)
# #print('++++++++++++++++++++++++++')

'''
