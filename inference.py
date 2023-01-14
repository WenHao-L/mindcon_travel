import os
import argparse
import numpy as np
import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from mindspore.dataset.vision.utils import Inter
from mindvision.classification.models import resnet101


# parse_arguments
parser = argparse.ArgumentParser(description="MindSpore Inference")
parser.add_argument('--best_ckpt_path', default="./resnet101_ckpt/best.ckpt", type=str, help='ckpt path')
parser.add_argument('--test_dataset_dir', default='./dataset/test', type=str, help='test dataset dir') 
parser.add_argument('--inference_result_path', default='./result/result.txt', type=str, help='the result path to save') 
parser.add_argument('--sorted_inference_result_path', default='./result/sorted_result.txt', type=str, help='the sorted result path to save') 
args = parser.parse_args()

# 定义网络
num_class = 54
net = resnet101(num_class)

# 加载模型参数
param_dict = ms.load_checkpoint(args.best_ckpt_path)
ms.load_param_into_net(net, param_dict)
model = ms.Model(net)

# 加载数据集
mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
image_size = 224

transform_img = [
    vision.Decode(),
    vision.Resize(int(256 / 224 * image_size), interpolation=Inter.BICUBIC),
    vision.CenterCrop(image_size),
    vision.Normalize(mean=mean, std=std, is_hwc=True),
    vision.HWC2CHW()
]

test_dataset = ds.ImageFolderDataset(args.test_dataset_dir, shuffle=False)
test_dataset = test_dataset.map(input_columns="image", operations=transform_img)
test_dataset = test_dataset.batch(1)

images_path = './dataset/test/images'
file_name_list = os.listdir(images_path)
file_name_list = sorted(file_name_list)
# print(file_name_list)

for i, data in enumerate(test_dataset.create_dict_iterator()):
    image = data["image"].asnumpy()
    output = model.predict(ms.Tensor(image))
    pred = np.argmax(output.asnumpy(), axis=1)
    # print(pred)
    with open(args.inference_result_path, 'a+') as file:
        log = file_name_list[i] + ',' + str(pred[0])
        file.write(log + '\n')
        file.close()

# 对推理结果进行排序
result_list = []
with open(args.inference_result_path, 'r') as f:
    for line in f:
        result_list.append(line.strip())
# print('result:', result_list)

index_list = []
for result in result_list:
    end_num = result.find('.')
    index = int(result[0:end_num])
    index_list.append(index)
# print('index:', index_list)

order_list = []
for i in range(len(index_list)):
    order_list.append((index_list[i], result_list[i].split(',')[-1]))
# print('order:', order_list)

sorted_list = sorted(order_list, key=lambda tup: tup[0])
# print('sorted:', sorted_list)

with open(args.sorted_inference_result_path, "w") as f:
    for item in sorted_list:
        f.writelines(item[1])
        f.writelines('\n')
    f.close()