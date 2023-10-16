####################
# load mnist
####################

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

import tensorflow as tf
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from tensorflow.keras.applications import DenseNet201 # 导入DenseNet201模型
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model

# 加载MNIST数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

data = np.concatenate( (x_train, x_test) )
label = np.concatenate( (y_train, y_test) )

# 创建DenseNet-201模型
model = DenseNet201(weights='imagenet', include_top=False)

# 添加全局平均池化层
x = GlobalAveragePooling2D()(model.output)

# 创建新的模型，该模型以DenseNet的输出作为输入，并输出特征向量
feature_extractor = Model(inputs=model.input, outputs=x)

# 提取特征（批处理方式）
batch_size = 1000
features = []

# 特征提取
for i in range(0, len(data), batch_size):
    
    batch = data[i:i + batch_size]

    # 数据预处理
    batch = np.repeat( batch[..., np.newaxis], 3, -1 ) # 转换为RGB格式
    batch = tf.image.resize( batch, (224, 224) ) # 调整大小以适应DenseNet-201的输入尺寸
    batch = tf.keras.applications.densenet.preprocess_input(batch) # 预处理输入数据
    
    batch_features = feature_extractor.predict( batch, verbose=2 )
    features.extend(batch_features)

data = np.array(features)

# 输出特征的形状
print()
print('data.shape =', data.shape)
print('label.shape =', label.shape)

####################
# pca
####################

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
data_2d = pca.fit_transform(data)

plt.figure()
plt.scatter( data_2d[:, 0], data_2d[:, 1], c=label,
             cmap='jet', marker='.', alpha=0.1 )
plt.title('DenseNet-201 & PCA')
plt.pause(0.1)

####################
# tsne
####################

from sklearn.manifold import TSNE

tsne = TSNE( n_components=2,
             learning_rate='auto',
             init='random',
             n_jobs=-1,
             )
data_2d = tsne.fit_transform(data)

plt.figure()
plt.scatter( data_2d[:, 0], data_2d[:, 1], c=label,
             cmap='jet', marker='.', alpha=0.1 )
plt.title('DenseNet-201 & t-SNE')
plt.pause(0.1)

####################
# umap
####################

# pip install umap-learn
from umap import UMAP

umap = UMAP(n_components=2, n_jobs=-1)
data_2d = umap.fit_transform(data)

plt.figure()
plt.scatter( data_2d[:, 0], data_2d[:, 1], c=label,
             cmap='jet', marker='.', alpha=0.1 )
plt.title('DenseNet-201 & UMAP')
plt.show()
