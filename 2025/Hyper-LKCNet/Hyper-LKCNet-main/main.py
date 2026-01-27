from func import *
from CCA_att import *
from FocalLoss import *
from ETFclassifier import *

import keras
from keras.layers import Conv2D, Conv3D, Flatten, Dense, Reshape,BatchNormalization
from keras.layers import Dropout, Input
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from sklearn.metrics import classification_report

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
from collections import defaultdict
import pandas as pd
import spectral
import tensorflow as tf
import keras.backend as BK

name = 'KSC'
data_path = os.path.join(os.getcwd(), 'data')
if name == 'KSC':
    data = sio.loadmat(os.path.join(data_path, 'KSC.mat'))['KSC']
    labels = sio.loadmat(os.path.join(data_path, 'KSC_gt.mat'))['KSC_gt']
elif name == 'PU':
    data = sio.loadmat(os.path.join(data_path, 'PaviaU.mat'))['paviaU']
    labels = sio.loadmat(os.path.join(data_path, 'PaviaU_gt.mat'))['paviaU_gt']
elif name == 'IP':
    data = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
    labels = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']
elif name == 'LK':
    data = sio.loadmat(os.path.join(data_path, 'WHU_Hi_LongKou.mat'))['WHU_Hi_LongKou']
    labels = sio.loadmat(os.path.join(data_path, 'WHU_Hi_LongKou_gt.mat'))['WHU_Hi_LongKou_gt']
    # 将数据转换为 np.int64 类型
    data = data.astype(np.int64)
    labels = labels.astype(np.int64)

    windowSize = 39
    kernelsize = 21
    X, y = data, labels
    print(X.shape, y.shape)

    K = X.shape[2]
    print(K)

    K = 20 if name == 'LK' or name == 'KSC' else (30 if name == 'IP' else 15)
    X, pca = applyPCA(X, numComponents=K)  # 数据主成分降维
    print(X.shape)  # 降维后形状
    X, y = createImageCubes(X, y, windowSize=windowSize)
    print(X.shape, y.shape)

    test_ratio = 0.9
    Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X, y, test_ratio)
    print(Xtrain.shape, ytrain.shape)
    print(Xtest.shape,ytest.shape)

    # 划分训练(train,valid)
    Xtrain, Xvalid, ytrain, yvalid = splitTrainTestSet(Xtrain, ytrain, 0.5)
    print(Xtrain.shape, Xvalid.shape, ytrain.shape, yvalid.shape)
    Xtrain = Xtrain.reshape(-1, windowSize, windowSize, K, 1)
    print(Xtrain.shape)
    ytrain =to_categorical(ytrain)
    print(ytrain.shape)

    Xvalid = Xvalid.reshape(-1, windowSize, windowSize, K, 1)
    print(Xvalid.shape)

    yvalid =to_categorical(yvalid)
    print(yvalid.shape)

    from keras.layers import SeparableConv2D
    S = windowSize
    L = K
    output_units = 9 if name == 'PU' or name == 'LK' else (13 if name == 'KSC' else 16)

    # Shared 3D CNN part
    input_layer = Input((S, S, K, 1))
    conv_layer1 = Conv3D(filters=8, kernel_size=(3, 3, 7), activation='relu')(input_layer)
    # conv_layer1 =BatchNormalization()(conv_layer1)
    conv_layer2 = Conv3D(filters=16, kernel_size=(3, 3, 5), activation='relu')(conv_layer1)
    # conv_layer2 = BatchNormalization()(conv_layer2)
    conv_layer3 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu')(conv_layer2)
    conv3d_shape = conv_layer3._keras_shape
    conv_layer3 = Reshape((conv3d_shape[1], conv3d_shape[2], conv3d_shape[3]*conv3d_shape[4]))(conv_layer3)
    # channel attention
    c = conv_layer3.shape[3].value
    h = conv_layer3.shape[1].value
    w = conv_layer3.shape[2].value
    cca_model = CCA()(conv_layer3)

    # Large Kernel
    conv_layer4_detail = Conv2D(filters=64, kernel_size=(3,3), activation='relu')(cca_model)
    conv_layer5_detail = SeparableConv2D(filters=64, kernel_size=(kernelsize, kernelsize), activation='relu')(conv_layer4_detail)
    # FC1
    flatten_layer_detail = Flatten()(conv_layer5_detail)
    dense_layer1_detail = Dense(units=256, activation='relu')(flatten_layer_detail)
    dense_layer1_detail = Dropout(0.4)(dense_layer1_detail)
    # # ETFHead
    num_classes = output_units
    in_channels = 256
    etf_head = ETFHead(num_classes, in_channels)(dense_layer1_detail)
    # FC2
    dense_layer2_detail = Dense(units=128, activation='relu')(etf_head)
    dense_layer2_detail = Dropout(0.4)(dense_layer2_detail)
    # Output layer
    output_layer = Dense(units=output_units, activation='softmax')(dense_layer2_detail)

    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)
    model.summary()
    # Compile the model
    adam = Adam(lr=0.001, decay=1e-06)
    focal_loss = FocalLoss(gamma=2, weight=None,label_smoothing=0.0)
    # model.compile(loss=focal_loss, optimizer=adam, metrics=['accuracy'])
    model.compile(loss=focal_loss, optimizer=adam, metrics=['accuracy'])
    # Checkpoint
    filepath = f"/public/home//best-model.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    # Train the model
    history = model.fit(x=Xtrain, y=ytrain, batch_size=256, epochs=100, callbacks=callbacks_list)

    # Save the model
    model.save(filepath)

    plt.figure(figsize=(7,7))
    plt.grid()
    plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Training','Validation'], loc='upper right')
    plt.savefig(f"/public/home/loss_curve.pdf")
    # plt.show()

    plt.figure(figsize=(5,5))
    plt.ylim(0,1.1)
    plt.grid()
    plt.plot(history.history['accuracy'])
    #plt.plot(history.history['val_acc'])
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['Training','Validation'])
    plt.savefig(f"/public/home/acc_curve.pdf")

    # Validation
    # load best weights
    model.load_weights(filepath)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    # model.compile(loss=focal_loss, optimizer=adam, metrics=['accuracy'])

    Xtest = Xtest.reshape(-1, windowSize, windowSize, K, 1)
    print(Xtest.shape)

    ytest = to_categorical(ytest)
    print(ytest.shape)

    Y_pred_test = model.predict(Xtest)
    y_pred_test = np.argmax(Y_pred_test, axis=1)
    print(ytest.shape)
    print(y_pred_test.shape)

    classification = classification_report(np.argmax(ytest, axis=1), y_pred_test)
    print(classification)

    classification, confusion, Test_loss, Test_accuracy, oa, each_acc, aa, kappa = reports(model,Xtest, ytest, name)
    classification = str(classification)
    confusion = str(confusion)
    file_name = f"/public/home/classification_report.txt"


    with open(file_name, 'w') as x_file:
        x_file.write('{} Test loss (%)'.format(Test_loss))
        x_file.write('\n')
        x_file.write('{} Test accuracy (%)'.format(Test_accuracy))
        x_file.write('\n')
        x_file.write('\n')
        x_file.write('{} Kappa accuracy (%)'.format(kappa))
        x_file.write('\n')
        x_file.write('{} Overall accuracy (%)'.format(oa))
        x_file.write('\n')
        x_file.write('{} Average accuracy (%)'.format(aa))
        x_file.write('\n')
        x_file.write('\n')
        x_file.write('{}'.format(classification))
        x_file.write('\n')
        x_file.write('{}'.format(confusion))

    # 测试集的真实标签和预测标签
    # 这里假设 ytest 是真实标签，y_pred_test 是预测标签
    # 如果不是，请将其替换为真实的标签和预测的标签
    y_true = np.argmax(ytest, axis=1)

    # 计算混淆矩阵、OA、AA 和 Kappa
    conf_matrix, overall_accuracy, avg_accuracy, kappa = calculate_metrics(y_true, y_pred_test)

    # 计算各类别精度
    class_accuracies = calculate_class_accuracies(conf_matrix)

    # 定义 CSV 文件名
    csv_filename = f"/public/home/test_metrics.csv"

    # 保存到 CSV
    save_to_csv(class_accuracies, overall_accuracy, avg_accuracy, kappa, csv_filename)

    print("Test metrics saved to:", csv_filename)

# load the original image
X, y = data,labels
height = y.shape[0]
width = y.shape[1]
PATCH_SIZE = windowSize
numComponents = K

X, pca = applyPCA(X, numComponents=numComponents)

X = padWithZeros(X, PATCH_SIZE // 2)

# calculate the predicted image
outputs = np.zeros((height, width))
for i in range(height):
    for j in range(width):
        target = int(y[i, j])
        # if target == 0:
        #     continue
        # else:
        image_patch = Patch(X, i, j,PATCH_SIZE)
        X_test_image = image_patch.reshape(1, image_patch.shape[0], image_patch.shape[1], image_patch.shape[2],
                                               1).astype('float32')
        prediction = (model.predict(X_test_image))
        prediction = np.argmax(prediction, axis=1)
        outputs[i][j] = prediction + 1

# 使用DrawResult绘制预测图像
X_result = DrawResult(outputs.astype(int), imageID=7)
plt.imsave('/public/home/LK/' + 'allpre'+ '.png', X_result)
print('结果图保存阶段完成！', '\n')
outputs = np.zeros((height, width))
for i in range(height):
    for j in range(width):
        target = int(y[i, j])
        if target == 0:
            continue
        else:
        image_patch = Patch(X, i, j,PATCH_SIZE)
        X_test_image = image_patch.reshape(1, image_patch.shape[0], image_patch.shape[1], image_patch.shape[2],
                                               1).astype('float32')
        prediction = (model.predict(X_test_image))
        prediction = np.argmax(prediction, axis=1)
        outputs[i][j] = prediction + 1
ground_truth = spectral.imshow(classes=y, figsize=(7, 7))

predict_image = spectral.imshow(classes=outputs.astype(int), figsize=(7, 7))

spectral.save_rgb("predictions.jpg", outputs.astype(int), colors=spectral.spy_colors)

spectral.save_rgb(str(name)+ "_ground_truth.jpg", y, colors=spectral.spy_colors)