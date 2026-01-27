import torch
import torch.distributed as dist
import tensorflow as tf
from keras.layers import Layer
import numpy as np

def generate_random_orthogonal_matrix(feat_in, num_classes):
    rand_mat = np.random.random(size=(feat_in, num_classes))
    orth_vec, _ = np.linalg.qr(rand_mat)
    orth_vec = orth_vec.astype(np.float32)
    return orth_vec

def produce_training_rect(label, num_classes):
    rank, world_size = 0, 0  # placeholder for get_dist_info() function
    if world_size > 0:
        recv_list = [None for _ in range(world_size)]
        dist.all_gather_object(recv_list, label.cpu())
        new_label = torch.cat(recv_list).to(device=label.device)
        label = new_label
    uni_label, count = torch.unique(label, return_counts=True)
    batch_size = label.size(0)
    uni_label_num = uni_label.size(0)
    assert batch_size == torch.sum(count)
    gamma = torch.tensor(batch_size / uni_label_num, device=label.device, dtype=torch.float32)
    rect = torch.ones(1, num_classes).to(device=label.device, dtype=torch.float32)
    rect[0, uni_label] = torch.sqrt(gamma / count)
    return rect

class ETFHead(Layer):
    def __init__(self, num_classes, in_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.in_channels = in_channels

        orth_vec = generate_random_orthogonal_matrix(self.in_channels, self.num_classes)
        i_nc_nc = np.eye(self.num_classes)
        one_nc_nc = np.ones((self.num_classes, self.num_classes)) / self.num_classes
        etf_vec = np.matmul(orth_vec, i_nc_nc - one_nc_nc)
        etf_vec = etf_vec * np.sqrt(self.num_classes / (self.num_classes - 1))
        self.etf_vec = self.add_weight(shape=etf_vec.shape, initializer=tf.constant_initializer(etf_vec),
                                        trainable=False)

    def pre_logits(self, x):
        x = x / tf.norm(x, axis=1, keepdims=True)
        return x

    def call(self, x, **kwargs):
        x = self.pre_logits(x)
        cls_score = tf.matmul(x, self.etf_vec)
        return cls_score

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.num_classes