import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from operator import truediv
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report,cohen_kappa_score
import pandas as pd
import matplotlib.pyplot as plt

def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test


def custom_train_test_split(labeled_data, flag='IP'):
    train_num = []
    for i in np.unique(labeled_data[:, -1]):
        temp = labeled_data[labeled_data[:, -1] == i, :]
        temp_num = temp[:, 0]  # 假设第一列是ID列
        np.random.shuffle(temp_num)  # 打乱顺序
        if flag == 'indian':
            if i == 1:
                train_num.append(temp_num[0:20])
            elif i == 7:
                train_num.append(temp_num[0:10])
            elif i == 9:
                train_num.append(temp_num[0:5])
            elif i == 15:
                train_num.append(temp_num[0:40])
            elif i == 16:
                train_num.append(temp_num[0:40])
            else:
                train_num.append(temp_num[0:90])

    train_num = np.concatenate(train_num)
    train_data = labeled_data[np.isin(labeled_data[:, 0], train_num)]
    test_data = labeled_data[~np.isin(labeled_data[:, 0], train_num)]

    return train_data, test_data

def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
    return newX, pca

def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

def createImageCubes(X, y, windowSize=5, removeZeroLabels = True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    return patchesData, patchesLabels

def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def reports(model,X_test, y_test, name):
    # start = time.time()
    Y_pred = model.predict(X_test)
    y_pred = np.argmax(Y_pred, axis=1)
    # end = time.time()
    # print(end - start)
    if name == 'IP':
        target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
            , 'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                        'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                        'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                        'Stone-Steel-Towers']
    elif name == 'SA':
        target_names = ['Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 'Fallow_rough_plow',
                        'Fallow_smooth',
                        'Stubble', 'Celery', 'Grapes_untrained', 'Soil_vinyard_develop', 'Corn_senesced_green_weeds',
                        'Lettuce_romaine_4wk', 'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk', 'Lettuce_romaine_7wk',
                        'Vinyard_untrained', 'Vinyard_vertical_trellis']
    elif name == 'PU':
        target_names = ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal sheets', 'Bare Soil', 'Bitumen',
                        'Self-Blocking Bricks', 'Shadows']
    elif name == 'KSC':
        target_names = ['Scrub', 'Willow swamp', 'CP hammock', 'CP/Oak','Slash pine',
                    'Oak/broadleaf', 'Hardwood Swamp', 'Graminoid marsh','Spartina marsh',
                    'Cattail marsh', 'Salt marsh', 'Mud flats', 'Water']
    elif name == 'LK':
        target_names = ['Corn', 'Cotton', 'Sesame', 'Broad-leaf soybean	', 'Narrow-leaf soybean',
                        'Rice', 'Water', 'Roads and houses', 'Mixed weed']

    classification = classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target_names)
    oa = accuracy_score(np.argmax(y_test, axis=1), y_pred)
    confusion = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(np.argmax(y_test, axis=1), y_pred)
    score = model.evaluate(X_test, y_test, batch_size=32)
    Test_Loss = score[0] * 100
    Test_accuracy = score[1] * 100

    return classification, confusion, Test_Loss, Test_accuracy, oa * 100, each_acc * 100, aa * 100, kappa * 100

# 计算混淆矩阵
def calculate_confusion_matrix(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred)
    # 添加缺失的类别
    if cm.shape[0] < num_classes:
        cm = np.pad(cm, ((0, num_classes - cm.shape[0]), (0, 0)), mode='constant', constant_values=0)
    return cm

# 计算各类别精度
def calculate_class_accuracies(confusion_matrix):
    class_accuracies = np.diag(confusion_matrix) / confusion_matrix.sum(axis=1)
    return class_accuracies

# 保存到 CSV
def save_to_csv(accuracies, oa, aa, kappa, filename):
    class_names = [f'Class {i}' for i in range(len(accuracies))]
    df = pd.DataFrame({'Class': class_names, 'Accuracy': accuracies})
    df.loc[len(accuracies)] = ['Overall Accuracy', oa]
    df.loc[len(accuracies) + 1] = ['Average Accuracy', aa]
    df.loc[len(accuracies) + 2] = ['Kappa', kappa]
    df.to_csv(filename, index=False)

# 计算 OA, AA 和 Kappa
def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    oa = accuracy_score(y_true, y_pred)
    aa = np.mean(calculate_class_accuracies(cm))
    kappa = cohen_kappa_score(y_true, y_pred)
    return cm, oa, aa, kappa

def Patch(data, height_index, width_index,PATCH_SIZE):
    height_slice = slice(height_index, height_index + PATCH_SIZE)
    width_slice = slice(width_index, width_index + PATCH_SIZE)
    patch = data[height_slice, width_slice, :]

    return patch

def DrawResult(labels, imageID):
    labels -= 1
    num_class = labels.max() + 1
    if imageID == 1:  # PU
        row = 610
        col = 340
        palette = np.array([[216, 191, 216],
                            [0, 255, 0],
                            [0, 255, 255],
                            [45, 138, 86],
                            [255, 0, 255],
                            [255, 165, 0],
                            [159, 31, 239],
                            [255, 0, 0],
                            [255, 255, 0]])
        palette = palette * 1.0 / 255
    elif imageID == 2:  # IP
        row = 145
        col = 145
        palette = np.array([[255, 0, 0],
                            [0, 255, 0],
                            [0, 0, 255],
                            [255, 255, 0],
                            [0, 255, 255],
                            [255, 0, 255],
                            [176, 48, 96],
                            [46, 139, 87],
                            [160, 32, 240],
                            [255, 127, 80],
                            [127, 255, 212],
                            [218, 112, 214],
                            [160, 82, 45],
                            [127, 255, 0],
                            [216, 191, 216],
                            [238, 0, 0]])
        palette = palette * 1.0 / 255
    elif imageID == 3:  # Botswana
        row = 1476
        col = 256
        palette = np.array([[255, 0, 0],
                            [0, 255, 0],
                            [0, 0, 255],
                            [255, 255, 0],
                            [0, 255, 255],
                            [255, 0, 255],
                            [176, 48, 96],
                            [46, 139, 87],
                            [160, 32, 240],
                            [255, 127, 80],
                            [127, 255, 212],
                            [218, 112, 214],
                            [160, 82, 45],
                            [127, 255, 0]])
        palette = palette * 1.0 / 255
    elif imageID == 4:  # Salinas
        row = 512
        col = 217
        palette = np.array([[37, 58, 150],
                            [47, 78, 161],
                            [56, 87, 166],
                            [56, 116, 186],
                            [51, 181, 232],
                            [112, 204, 216],
                            [119, 201, 168],
                            [148, 204, 120],
                            [188, 215, 78],
                            [238, 234, 63],
                            [246, 187, 31],
                            [244, 127, 33],
                            [239, 71, 34],
                            [238, 33, 35],
                            [180, 31, 35],
                            [123, 18, 20]])
        palette = palette * 1.0 / 255
    elif imageID == 5:  # PaviaC
        row = 1096
        col = 715
        palette = np.array([[37, 97, 163],
                            [44, 153, 60],
                            [122, 182, 41],
                            [219, 36, 22],
                            [227, 156, 47],
                            [227, 221, 223],
                            [108, 35, 127],
                            [130, 67, 142],
                            [229, 225, 74]])
        palette = palette * 1.0 / 255
    elif imageID == 6:  # KSC
        row = 512
        col = 614
        palette = np.array([[94, 203, 55],
                            [255, 0, 255],
                            [217, 115, 0],
                            [179, 30, 0],
                            [0, 52, 0],
                            [72, 0, 0],
                            [255, 255, 255],
                            [145, 132, 135],
                            [255, 255, 172],
                            [255, 197, 80],
                            [60, 201, 255],
                            [11, 63, 124],
                            [0, 0, 255]])
        palette = palette * 1.0 / 255
    elif imageID == 7:  # Longkou
        row = 550
        col = 400
        palette = np.array([[255, 0, 0],
                            [0, 255, 0],
                            [0, 0, 255],
                            [255, 255, 0],
                            [0, 255, 255],
                            [255, 0, 255],
                            [176, 48, 96],
                            [46, 139, 87],
                            [160, 32, 240]])
        palette = palette * 1.0 / 255

    X_result = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    for i in range(0, num_class):
        mask = (labels == i)
        X_result[mask] = (palette[i] * 255).astype(np.uint8)

    # X_result = np.reshape(X_result, (row, col, 3))
    plt.axis("off")
    plt.imshow(X_result)
    return X_result