import sys
import numpy as np
from scipy import stats
from scipy.signal import find_peaks
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.metrics import r2_score
from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib as mpl

mpl.use('TKAgg')
import copy
import random

try:
    import tensorflow.compat.v1 as tf
    tf.compat.v1.disable_eager_execution()
    print("成功加载 TensorFlow GPU 版本")
except ImportError as e:
    print(f"警告: 无法加载 TensorFlow GPU 版本 ({e}). 切换到 CPU 模式运行.")
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')

from sklearn.linear_model import LinearRegression
import csv

def c_index100(y0, bs, NN):
    L = y0.shape[0]
    N2 = round(L / bs)
    y1 = np.sort(y0.ravel())
    indexall = np.linspace(0, len(y1), NN + 1)

    label_index = []
    bindd = []
    for indd in range(NN):
        aindd = [idx for (idx, val) in enumerate(y0) if
                 y0[idx] >= y1[round(indexall[indd])] and y0[idx] < y1[round(indexall[indd + 1]) - 1]]
        random.shuffle(aindd)
        label_index.append(aindd)
        bindd.append(round(len(aindd) / N2))

    tx = []
    if N2 > 1:
        for n in range(N2):
            t_ix = []
            for indd in range(NN):
                t_ix.append(label_index[indd][n * bindd[indd]:(n + 1) * bindd[indd]])
            train_index = [item for sublist in t_ix for item in sublist]
            tx.append(train_index)
    elif N2 == 1:
        nn = int(bs / NN)
        t_ix = []
        for indd in range(NN):
            t_ix.append(label_index[indd][:nn])
        train_index0 = [item for sublist in t_ix for item in sublist]
        tx.append(train_index0)
        train_index1 = np.delete(np.arange(len(y1)), train_index0, axis=0)
        tx.append(train_index1)
    return tx


def w_pls_nonlinear(x1, y, na):
    #########PLS模型
    pls = PLSRegression(n_components=na)
    pls.fit(x1, y)
    w01_raw = pls.x_rotations_.astype(np.float32)
    max_abs_raw = np.max(np.abs(w01_raw))
    w01=w01_raw/(max_abs_raw*100)###100这个系数看情况改目的是别让XW进入tanh的死区但也不能太大太大了WX纯在tanh线性范围内了

    T = pls.x_scores_.astype(np.float32)

    raw_proj = np.matmul(x1, w01)  # N×H
    b01 = np.mean(T - raw_proj, axis=0)  #b的初始化就是0，没啥用
    b01 = b01.astype(np.float32)

    tnew= np.matmul(x1, w01)+b01

    Z= np.tanh(tnew)
    model = LinearRegression()

    model.fit(Z,y)
    w02 = ((model.coef_.T)).astype(np.float32)
    b02= ((model.intercept_)).astype(np.float32)
    return w01, b01,w02,b02

def w_pls_linear(x1, y, na):
    pls = PLSRegression(n_components=na)
    pls.fit(x1, y)
    w01 = pls.x_rotations_.astype(np.float32)
    w02 = pls.y_loadings_.T.astype(np.float32)
    return w01, w02


def build_model(Nfea, Numc, flag_linear):
    input = tf.placeholder(tf.float32, [None, Nfea])
    output = tf.placeholder(tf.float32, [None, 1])
    flag_train = tf.placeholder(tf.bool)
    lr0 = tf.placeholder(tf.float32)

    with tf.variable_scope("PLS_NET"):
        w1 = tf.get_variable("W1", shape=[Nfea, Numc],
                             initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
        b1 = tf.get_variable("B1", shape=[Numc], initializer=tf.zeros_initializer())
        z1 = tf.matmul(input, w1) + b1
        o1 = tf.tanh(z1)
        w2 = tf.get_variable("W2", shape=[Numc, 1], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
        b2 = tf.get_variable("B2", shape=[1], initializer=tf.zeros_initializer())
        if flag_linear == 'linear':
            z2 = tf.matmul(z1, w2) + b2
        else:
            z2 = tf.matmul(o1, w2) + b2
        final_tensor = z2

    loss = tf.reduce_mean(tf.square(final_tensor - output))
    t_step = tf.train.AdamOptimizer(learning_rate=lr0).minimize(loss)
    pred = final_tensor
    return input, output, flag_train, lr0, pred, t_step, w1, b1, w2, b2, o1


scaler2 = StandardScaler()


def PLSaoNET_Modeling_sub(train_iv, train_ov0, valid_iv, valid_ov0, test_data, hidden_neurons, BATCH_train,
                          LEARNING_rate,
                          N_stra=5, STEPS=1000, TrainThresholds=5, ValidThresholds=5, F_linear='nonlinear'):
    tf.reset_default_graph()
    Nfea = train_iv.shape[-1]
    Numc = hidden_neurons

    bottleneck_input, ground_truth_input, flag_training, learning_rate0, y_pred, train_step, weights_1, biases_1, weights_2, biases_2, middleo1 = build_model(
        Nfea, Numc, F_linear)

    train_ov = scaler2.fit_transform(train_ov0.reshape(-1, 1))
    t_rmse = np.zeros((STEPS, 1), dtype=float)
    v_rmse = np.zeros((STEPS, 1), dtype=float)

    test_number = len(test_data)
    p_rmse = np.zeros((test_number, STEPS, 1), dtype=float)
    test_pred = []
    for n1 in range(test_number):
        test_pred_sub = copy.deepcopy(test_data[n1][1])
        test_pred.append(test_pred_sub)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        if F_linear == 'linear':
            w1_pls, w2_pls = w_pls_linear(train_iv, train_ov, Numc)
            sess.run(tf.assign(weights_1, w1_pls))
            sess.run(tf.assign(weights_2, w2_pls))
        else:
            w1_pls, b1_pls, w2_pls, b2_pls = w_pls_nonlinear(train_iv, train_ov, Numc)
            update1 = tf.assign(weights_1, w1_pls)
            sess.run(update1)
            update2 = tf.assign(biases_1, b1_pls)
            sess.run(update2)
            update3 = tf.assign(weights_2, w2_pls)
            sess.run(update3)
            update4 = tf.assign(biases_2, b2_pls)
            sess.run(update4)

        train_pred1 = sess.run(y_pred, feed_dict={bottleneck_input: train_iv, flag_training: False,
                                                  learning_rate0: LEARNING_rate})
        valid_pred1 = sess.run(y_pred, feed_dict={bottleneck_input: valid_iv, flag_training: False,
                                                  learning_rate0: LEARNING_rate})
        train_pred = scaler2.inverse_transform(train_pred1).ravel()
        valid_pred = scaler2.inverse_transform(valid_pred1).ravel()
        t_rmse[0] = mean_squared_error(train_ov0.ravel(), train_pred, squared=False)
        v_rmse[0] = mean_squared_error(valid_ov0.ravel(), valid_pred, squared=False)

        for n1 in range(test_number):
            test_pred1 = sess.run(y_pred, feed_dict={bottleneck_input: test_data[n1][0], flag_training: False,
                                                     learning_rate0: LEARNING_rate})
            test_pred[n1] = scaler2.inverse_transform(test_pred1).ravel()
            p_rmse[n1, 0] = mean_squared_error(test_data[n1][1].ravel(), test_pred[n1], squared=False)

        loss_min1 = 20
        loss_min2 = 20
        s = 0
        t = 0

        for i in range(1, STEPS):
            tx = c_index100(train_ov0, BATCH_train, N_stra)
            N2 = len(tx)
            for n in range(N2):
                train_index_new = tx[n]
                sess.run(train_step, feed_dict={bottleneck_input: train_iv[train_index_new],
                                                ground_truth_input: train_ov[train_index_new],
                                                flag_training: True,
                                                learning_rate0: LEARNING_rate})

            train_pred1 = sess.run(y_pred, feed_dict={bottleneck_input: train_iv, flag_training: False,
                                                      learning_rate0: LEARNING_rate})
            valid_pred1 = sess.run(y_pred, feed_dict={bottleneck_input: valid_iv, flag_training: False,
                                                      learning_rate0: LEARNING_rate})
            train_pred = scaler2.inverse_transform(train_pred1).ravel()
            valid_pred = scaler2.inverse_transform(valid_pred1).ravel()

            t_rmse[i] = mean_squared_error(train_ov0.ravel(), train_pred, squared=False)
            v_rmse[i] = mean_squared_error(valid_ov0.ravel(), valid_pred, squared=False)

            for n1 in range(test_number):
                test_pred1 = sess.run(y_pred, feed_dict={bottleneck_input: test_data[n1][0], flag_training: False,
                                                         learning_rate0: LEARNING_rate})
                test_pred[n1] = scaler2.inverse_transform(test_pred1).ravel()
                p_rmse[n1, i] = mean_squared_error(test_data[n1][1].ravel(), test_pred[n1], squared=False)

            if t_rmse[i] < loss_min1:
                s = 0
                loss_min1 = t_rmse[i]
            else:
                s += 1

            if v_rmse[i] < loss_min2:
                t = 0
                loss_min2 = v_rmse[i]
            else:
                t += 1

            if ((s > TrainThresholds) and (t > ValidThresholds)) or (t > ValidThresholds + 50) or (i >= STEPS - 1):
                w1 = sess.run(weights_1, feed_dict={bottleneck_input: train_iv, flag_training: False,
                                                    learning_rate0: LEARNING_rate})
                w2 = sess.run(weights_2, feed_dict={bottleneck_input: train_iv, flag_training: False,
                                                    learning_rate0: LEARNING_rate})
                b1 = sess.run(biases_1, feed_dict={bottleneck_input: train_iv, flag_training: False,
                                                   learning_rate0: LEARNING_rate})
                b2 = sess.run(biases_2, feed_dict={bottleneck_input: train_iv, flag_training: False,
                                                   learning_rate0: LEARNING_rate})
                o1 = sess.run(middleo1, feed_dict={bottleneck_input: train_iv, flag_training: False,
                                                   learning_rate0: LEARNING_rate})
                break
        else:
            w1 = sess.run(weights_1, feed_dict={bottleneck_input: train_iv, flag_training: False,
                                                learning_rate0: LEARNING_rate})
            w2 = sess.run(weights_2, feed_dict={bottleneck_input: train_iv, flag_training: False,
                                                learning_rate0: LEARNING_rate})
            b1 = sess.run(biases_1, feed_dict={bottleneck_input: train_iv, flag_training: False,
                                               learning_rate0: LEARNING_rate})
            b2 = sess.run(biases_2, feed_dict={bottleneck_input: train_iv, flag_training: False,
                                               learning_rate0: LEARNING_rate})
            o1 = sess.run(middleo1, feed_dict={bottleneck_input: train_iv, flag_training: False,
                                               learning_rate0: LEARNING_rate})

    return train_pred, valid_pred, test_pred, o1, w1, w2, b1, b2, t_rmse, v_rmse, i


def PLSaoNET_Modeling(dir_train,dir_parameter, dir_save, element_name,
                      train_start, train_end, n_components,bestBS, bestlr, flag_linear, flag_nor,
                      S_en_Factor, B_de_Factor,
                      n_stra, steps, train_threshold, valid_threshold, index_flag,flag_multispec):
    try:
        # ================== 1. 加载数据 ==================
        
        print(f"预处理输入光谱形状: {spec_data.shape}")

        # ================== 2. 光谱预处理 ==================
        print("\n[步骤2] 光谱预处理...")
        ## ================== 3. 划分训练集和测试集 ==================
        print("\n[步骤3] 划分训练集和测试集...")
        

        print("步骤 4: 开始执行输入标准化...")
        
        print("步骤 5: 开始训练 PLSaoNET 模型...")

        from sklearn.model_selection import train_test_split
        train_i_remain, valid_i, y_train_remain, y_valid = train_test_split(train_i, y_train,test_size=0.2, random_state=0,shuffle=True)

        y_train_pred, y_valid_pred, y_test_pred, ttt, ww01, ww02, bb01, bb02, train_rmse, valid_rmse, nnn = PLSaoNET_Modeling_sub(
            train_i, y_train, valid_i, y_valid, [[test_i, y_test, '测试集']], n_components, bestBS, bestlr,
            N_stra=n_stra, STEPS=steps, TrainThresholds=train_threshold, ValidThresholds=valid_threshold,
            F_linear=flag_linear)
       
        # ================== 6. 模型评估 ==================

        print("步骤 6: 开始绘制并保存结果图...")

        return 1
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0


if __name__ == "__main__":

    # 默认测试参数
    ########陈彤代码调试
    
    #######新增一个上位机的勾选框，指示是否是多光谱
    flag_multispec = 1

    #######传入参数
    element_name = 'Al'
    dir_train 
    dir_parameter
    dir_save

    train_start = 10
    train_end = 90
    pls_principal_components = 12
    index_flag = 0
    BS=32
    lr=0.0001
    flag_net_linear = 'nonlinear'  #######网络是线性的还是非线性的，可选linear、nonlinear
    flag_normalizer = 'Normal'  #######标准化方式是默认的各维全减均值除方差还是区分信号和背景，可选Normal、Custom
        
    ################下面这四个可以告诉用户，基本不用改
    n_stra = 5  #######训练样本分层抽样的层数
    n_steps = 1000  #######训练最大迭代次数
    t_threshold = 5  #######提前终止时训练集的停止下降次数约束
    v_threshold = 5  #######提前终止时验证集的停止下降次数约束
    ####新增
    Indexflag=0#指示数据集划分方式是从文件读取离散索引，还是直接以 train_start到train_end截取一段

    
    ####调用
    PLSaoNET_Modeling(dir_train, dir_parameter, dir_save, element_name, train_start, train_end,
                      pls_principal_components, BS, lr,
                      flag_net_linear, flag_normalizer, Signal_en_factor, Background_de_factor, n_stra, n_steps,
                      t_threshold, v_threshold, Indexflag,flag_multispec)
