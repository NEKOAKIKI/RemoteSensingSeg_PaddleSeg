import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import random
import paddle
import paddleseg.transforms as T
from paddleseg.datasets import Dataset
from paddleseg.models import HarDNet
from paddleseg.models.losses import CrossEntropyLoss
from paddleseg.core import train, evaluate, predict


datas = []
image_base = 'data/data77571/img_train'   # 训练集原图路径
annos_base = 'data/data77571/lab_train'   # 训练集标签路径

# 读取原图文件名
ids_ = [v.split('.')[0] for v in os.listdir(image_base)]

# 将训练集的图像集和标签路径写入datas中
for id_ in ids_:
    img_pt0 = os.path.join(image_base, '{}.jpg'.format(id_))
    img_pt1 = os.path.join(annos_base, '{}.png'.format(id_))
    datas.append((img_pt0.replace('/home/aistudio', ''), img_pt1.replace('/home/aistudio', '')))
    if os.path.exists(img_pt0) and os.path.exists(img_pt1):
        pass
    else:
        raise "path invalid!"

random.shuffle(datas)

# 打印datas的长度和具体存储例子
print('total:', len(datas))
print(datas[0][0])
print(datas[0][1])
print(datas[10][:])

plt.figure(figsize=(8, 8))
for i in range(len(datas[10][:])):
    plt.subplot(len(datas[10][:]), 2, i + 1)
    plt.title(datas[10][i])
    plt.imshow(cv2.imread(datas[10][i])[:, :, ::-1])
    
plt.show()


# #### **将训练集、测试集图片路径写入txt文件**
# 四类标签，这里用处不大，比赛评测是以0、1、2、3类来对比评测的
labels = ['建筑', '耕地', '林地',  '其他']

# 将labels写入标签文件
with open('data/labels.txt', 'w') as f:
    for v in labels:
        f.write(v + '\n')

# 随机打乱datas
np.random.seed(5)
np.random.shuffle(datas)

# 验证集与训练集的划分，0.05表示5%为训练集，95%为训练集
split_num = int(0.05*len(datas))

# 划分训练集和验证集
train_data = datas[:-split_num]
val_data = datas[-split_num:]

# 写入训练集list
with open('data/train_list.txt', 'w') as f:
    for img, lbl in train_data:
        f.write(img + ' ' + lbl + '\n')

# 写入验证集list
with open('data/val_list.txt', 'w') as f:
    for img, lbl in val_data:
        f.write(img + ' ' + lbl + '\n')

# 打印训练集和测试集大小
print('train:', len(train_data))
print('val:', len(val_data))


# #### **构建训练集和验证集**
dataset_root = './'
train_path = 'data/train_list.txt'
val_path = 'data/val_list.txt'
num_classes = 4

# 定义训练和验证时的transforms
train_transforms = [
    T.RandomHorizontalFlip(0.5),
    T.RandomVerticalFlip(0.5),
    T.RandomDistort(
        brightness_range=0.2, brightness_prob=0.5,
        contrast_range=0.2, contrast_prob=0.5,
        saturation_range=0.2, saturation_prob=0.5,
        hue_range=15, hue_prob=0.5),
    T.RandomPaddingCrop(crop_size=(256, 256)),
    T.Resize(target_size=(256, 256)),
    T.Normalize()
]
eval_transforms = [
    T.Resize((256, 256)),
    T.Normalize()
]

# 构建训练集
train_dataset = Dataset(transforms = train_transforms,
                  dataset_root = dataset_root,
                  num_classes = num_classes,
                  train_path = train_path,
                  mode = 'train')

# 构建验证集
eval_dataset = Dataset(transforms = eval_transforms,
                  dataset_root = dataset_root,
                  num_classes = num_classes,
                  val_path = val_path,
                  mode = 'val')


# ## 五、模型训练
# ### 1. 模型准备
# #### **构建模型**

model = HarDNet(num_classes=4)


# #### **构建优化器**

# 设置学习率
base_lr = 0.01
lr = paddle.optimizer.lr.PolynomialDecay(base_lr, power=0.9, decay_steps=1000, end_lr=0)

optimizer = paddle.optimizer.Momentum(lr, parameters=model.parameters(), momentum=0.9, weight_decay=4.0e-5)


# #### **构建损失函数**
losses = {}
losses['types'] = [CrossEntropyLoss()]
losses['coef'] = [1]


# ### 2. 模型训练
# 模型训练
train(
    model=model,
    train_dataset=train_dataset,
    val_dataset=eval_dataset,
    optimizer=optimizer,
    save_dir='output',
    iters=1000,
    batch_size=64,
    save_interval=200,
    log_iters=10,
    num_workers=0,
    losses=losses,
    use_vdl=True)


# ## 六、模型评估
# ### 1. 评估
evaluate(
        model,
        eval_dataset)

# ### 2. 多尺度+翻转评估
evaluate(
        model,
        eval_dataset,
        aug_eval=True,
        scales=[0.75, 1.0, 1.25],
        flip_horizontal=True)


# ### 3. 效果可视化
# #### **构建模型**
model = HarDNet(num_classes=4)

# #### **创建transform**
import paddleseg.transforms as T
transforms = T.Compose([
    T.Resize(target_size=(256, 256)),
    T.RandomHorizontalFlip(),
    T.Normalize()
])

# #### **构建待预测的图像列表**
def get_image_list(image_path):
    """Get image list"""
    valid_suffix = [
        '.JPEG', '.jpeg', '.JPG', '.jpg', '.BMP', '.bmp', '.PNG', '.png'
    ]
    image_list = []
    image_dir = None
    if os.path.isfile(image_path):
        if os.path.splitext(image_path)[-1] in valid_suffix:
            image_list.append(image_path)
    elif os.path.isdir(image_path):
        image_dir = image_path
        for root, dirs, files in os.walk(image_path):
            for f in files:
                if os.path.splitext(f)[-1] in valid_suffix:
                    image_list.append(os.path.join(root, f))
    else:
        raise FileNotFoundError(
            '`--image_path` is not found. it should be an image file or a directory including images'
        )

    if len(image_list) == 0:
        raise RuntimeError('There are not image file in `--image_path`')

    return image_list, image_dir
image_path = 'data/data77571/img_testA' # 也可以输入一个包含图像的目录
image_list, image_dir = get_image_list('data/data77571/img_testA')


# #### **预测**
predict(
        model,
        model_path='output/best_model/model.pdparams',
        transforms=transforms,
        image_list=image_list,
        image_dir=image_dir,
        save_dir='output/results'
    )
