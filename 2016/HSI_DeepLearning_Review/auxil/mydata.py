import copy
import numpy as np
import os
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def random_unison(a,b, rstate=None):
    """
    同步随机打乱两个数组 a 和 b
    """
    assert len(a) == len(b)
    p = np.random.RandomState(seed=rstate).permutation(len(a))
    return a[p], b[p]

def random_single(a, rstate=None):
    """
    随机打乱单个数组 a
    """
    return a[np.random.RandomState(seed=rstate).permutation(len(a))]

def loadData(name, num_components=None, preprocessing="standard"):
    """
    加载 HSI 数据集并进行预处理
    :param name: 数据集名称 (IP, SV, UP, UH 等)
    :param num_components: PCA 降维后的维数
    :param preprocessing: 预处理方法 (standard, minmax, none)
    """
    data_path = os.path.join(os.getcwd(),'../HSI-datasets')
    if name  in ["IP", "DIP", "DIPr"]:
        # 加载 Indian Pines 数据集
        data = sio.loadmat(os.path.join(data_path, 'indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'indian_pines_gt.mat'))['indian_pines_gt']
    elif name == 'SV':
        # 加载 Salinas 数据集
        data = sio.loadmat(os.path.join(data_path, 'salinas_corrected.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'salinas_gt.mat'))['salinas_gt']
    elif name  in ["UP", "DUP", "DUPr"]:
        # 加载 Pavia University 数据集
        data = sio.loadmat(os.path.join(data_path, 'paviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(data_path, 'paviaU_gt.mat'))['paviaU_gt']
    elif name == 'UH':
        # 加载 Houston 数据集
        data = sio.loadmat(os.path.join(data_path, 'houston.mat'))['houston']
        labels = sio.loadmat(os.path.join(data_path, 'houston_gt.mat'))['houston_gt_tr']
        labels += sio.loadmat(os.path.join(data_path, 'houston_gt.mat'))['houston_gt_te']
        num_class = 15
    else:
        print("NO DATASET")
        exit()
    
    # 设置类别数
    num_class = 15 if name == "UH" else 9 if name in ["UP", "DUP", "DUPr"] else 16
    shapeor = data.shape
    data = data.reshape(-1, data.shape[-1])
    
    # PCA 降维
    if num_components != None:
        data = PCA(n_components=num_components).fit_transform(data)
        shapeor = np.array(shapeor)
        shapeor[-1] = num_components
    
    # 数据标准化/归一化
    if preprocessing == "standard": data = StandardScaler().fit_transform(data)
    elif preprocessing == "minmax": data = MinMaxScaler().fit_transform(data)
    elif preprocessing == "none": pass
    else: print("[WARNING] Not preprocessing method selected")
    
    data = data.reshape(shapeor)
    return data, labels, num_class


def split_data(pixels, labels, value, splitdset="sklearn", rand_state=None):
    """
    划分训练集和测试集
    :param splitdset: 划分方式 (sklearn, custom, custom2)
    """
    if splitdset == "sklearn":
        # 使用 sklearn 的 train_test_split 进行分层抽样
        X_test, X_train, y_test, y_train = \
            train_test_split(pixels, labels, test_size=value, stratify=labels, random_state=rand_state)
    elif "custom" in splitdset:
        labels = labels.reshape(-1)
        X_train = []; X_test = []; y_train = []; y_test = [];
        if "custom" == splitdset: 
            # 自定义划分：基于每个类别的样本数量
            values = np.unique(value, return_counts=1)[1][1:]
            for idi, i in enumerate(values):
                samples = pixels[labels==idi+1]
                samples = random_single(samples, rstate=rand_state)
                for a in samples[:i]: 
                    X_train.append(a); y_train.append(idi)
                for a in samples[i:]:
                    X_test.append(a); y_test.append(idi)
        elif "custom2" == splitdset:
            # 另一种自定义划分
            for idi, i in enumerate(value):
                samples = pixels[labels==idi]
                samples = random_single(samples, rstate=rand_state)
                for a in samples[:i]: 
                    X_train.append(a); y_train.append(idi)
                for a in samples[i:]:
                    X_test.append(a); y_test.append(idi)
        X_train = np.array(X_train); X_test = np.array(X_test)
        y_train = np.array(y_train); y_test = np.array(y_test)
        # 随机打乱训练集
        X_train, y_train = random_unison(X_train,y_train, rstate=rand_state)
    return X_train, X_test, y_train, y_test


def select_samples(pixels, labels, samples):
    return split_data(pixels, labels, samples, splitdset="custom")

def load_split_data_fix(name, pixels, path_dset='../HSI-datasets'):
    """
    加载具有固定训练/测试划分的数据集
    """
    data_path = os.path.join(os.getcwd(), path_dset)
    if name == "UH":
        # Houston 数据集固定划分
        y_train = sio.loadmat(os.path.join(data_path, 'houston_gt.mat'))['houston_gt_tr'].reshape(-1)
        y_test = sio.loadmat(os.path.join(data_path, 'houston_gt.mat'))['houston_gt_te'].reshape(-1)
    elif name in ["DIP", "DIPr"]:
        # Indian Pines 不相交（disjoint）数据集划分
        y_train2 = sio.loadmat(\
                    os.path.join(data_path, 'indianpines_disjoint_dset.mat'))\
                                             ['indianpines_disjoint_dset']
        y_test = sio.loadmat(os.path.join(data_path, 'indian_pines_gt.mat'))['indian_pines_gt']
        y_train = copy.deepcopy(y_train2)
        # 标签映射
        for i, val in enumerate([0,2,3,5,6,8,10,11,12,14,1,4,7,9,13,15,16]): y_train[y_train2==i] = val
        del y_train2
        if name == "DIP": y_test[y_train!=0] = 0
        else: X_train, X_test, y_train, y_test = select_samples(pixels, y_test, y_train)
    elif name in ["DUP", "DUPr"]:
        # Pavia University 固定数据集划分
        y_train = sio.loadmat(os.path.join(data_path, 'TRpavia_fixed.mat'))['TRpavia_fixed'].reshape(-1)
        y_test = sio.loadmat(os.path.join(data_path, 'TSpavia_fixed.mat'))['TSpavia_fixed'].reshape(-1)
        if name == "DUP": pass
        else: X_train, X_test, y_train, y_test = select_samples(pixels, y_test, y_train)
    
    # 提取非零样本
    if name in ["UH", "DIP", "DUP"]:
        y_train = y_train.reshape(-1)
        y_test = y_test.reshape(-1)
        X_train = pixels[y_train!=0,:]
        X_test  = pixels[y_test!=0,:]
        del pixels
        y_train = y_train[y_train!=0] - 1 # 类别索引从 0 开始
        y_test  = y_test[y_test!=0] - 1
        X_train, y_train = random_unison(X_train,y_train, rstate=None)
        #X_test, y_test = random_unison(X_test,y_test, rstate=None)
    return X_train, X_test, y_train, y_test


def padWithZeros(X, margin=2):
    """
    对 HSI 数据进行零填充 (Zero Padding)
    """
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX
    # ALERT: TRY THIS
    #import cv2
    # return cv2.copyMakeBorder(X, margin, margin, margin, margin, cv2.BORDER_REPLICATE)


def createImageCubes(X, y, windowSize=5, removeZeroLabels = True):
    """
    为每个像素提取图像块 (Patch/Cube)
    :param windowSize: 窗口大小 (如 5x5)
    :param removeZeroLabels: 是否移除背景类 (标签为 0 的样本)
    """
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # 分块后的数据 shape: (总像素数, 窗口高, 窗口宽, 通道数)
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    # 遍历图像提取 patch
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    # 过滤掉标签为 0 的背景点
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1 # 类别索引从 0 开始
    return patchesData, patchesLabels.astype("int")
