import os
try:
    """安装缺少的包"""
    os.system(f"pip install mindvision -i https://pypi.tuna.tsinghua.edu.cn/simple")
    os.system(f"pip install opencv-python --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple")
except:
    print("修复失败")

import sys
import time
import mindspore as ms
import mindspore.nn as nn

from mindspore import context
from mindspore.common import set_seed

from mindvision.classification.models import resnet101
from mindvision.engine.callback import LossMonitor
from mindvision.engine.loss import CrossEntropySmooth

from src.data import TravelImageDataset
from src.args import get_config
from src.utils import set_device
from src.callback import EvaluateCallBack


def sync_data(args, environment="train"):
    if environment == "train":
        workroot = "/home/work/user-job-dir"
    elif environment == "debug":
        workroot = "/home/ma-user/work/"

    data_dir = os.path.join(workroot, "dataset")
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    train_dir = os.path.join(workroot, "model")
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)

    if environment == 'train':
        obs_data_url = args.data_url
        args.data_url = data_dir

        try:
            import moxing as mox
            mox.file.copy_parallel(obs_data_url, data_dir) 
            print("Successfully Download {} to {}".format(obs_data_url, args.data_url))
        except Exception as e:
            print('moxing download {} to {} failed: '.format(obs_data_url, args.data_url) + str(e))
    
    return train_dir


def sync_model(args, environment="train"):
    if environment == "train":
        workroot = "/home/work/user-job-dir"
    elif environment == "debug":
        workroot = "/home/ma-user/work/"

    train_dir = os.path.join(workroot, "model")
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)

    if environment == 'train':
        obs_train_url = args.train_url
        args.train_url = train_dir

        try:
            import moxing as mox
            mox.file.copy_parallel(args.train_url, obs_train_url)
            print("Successfully Upload {} to {}".format(args.train_url, obs_train_url))
        except Exception as e:
            print('moxing upload {} to {} failed: '.format(args.train_url, obs_train_url) + str(e))


def train(args):

    mode = {
        0: context.GRAPH_MODE,
        1: context.PYNATIVE_MODE
    }
    context.set_context(mode=mode[args.graph_mode], device_target=args.device_target)
    context.set_context(enable_graph_kernel=False)
    if args.device_target == "Ascend":
        context.set_context(enable_auto_mixed_precision=True)
    rank = set_device(args)
    set_seed(args.seed + rank)

    # train_dir = sync_data(args)
    train_dir = args.train_url
    
    # 加载数据集
    travel_dataset = TravelImageDataset(args=args)
    train_dataset = travel_dataset.train_dataset
    val_dataset = travel_dataset.val_dataset
    step_size = train_dataset.get_dataset_size()

    print("step_size:", step_size)
    for data in train_dataset.create_dict_iterator():
        print("Image shape: {}".format(data['image'].shape), ", Label shape: {}".format(data['label'].shape))
        print("Image type: {}".format(data['image'].dtype), ", Label type: {}".format(data['label'].dtype))
        break

    # 构建模型
    network = resnet101(pretrained=True)

    class DenseHead(nn.Cell):
        def __init__(self, input_channel, num_classes):
            super(DenseHead, self).__init__()
            self.dense = nn.Dense(input_channel, num_classes)

        def construct(self, x):
            return self.dense(x)

    # 全连接层输入层的大小
    in_channels = network.head.dense.in_channels
    # 输出通道数大小
    head = DenseHead(in_channels, args.num_classes)
    # 重置全连接层
    network.head = head

    # 定义递减的学习率
    lr = nn.cosine_decay_lr(min_lr=0.00001, max_lr=0.001, total_step=args.epochs * step_size,
                            step_per_epoch=step_size, decay_epoch=args.epochs)

    # 定义优化器
    network_opt = nn.Adam(network.trainable_params(), lr, args.momentum)

    # 定义损失函数
    network_loss = CrossEntropySmooth(sparse=False,
                                      reduction="mean",
                                      smooth_factor=0.1,
                                      classes_num=args.num_classes)

    # 设定checkpoint
    os.makedirs(train_dir, exist_ok=True)
    ckpt_config = ms.CheckpointConfig(save_checkpoint_steps=step_size*20, keep_checkpoint_max=args.keep_checkpoint_max)
    ckpt_callback = ms.ModelCheckpoint(prefix='resnet101', directory=train_dir, config=ckpt_config)

    # 初始化模型
    model = ms.Model(network, loss_fn=network_loss, optimizer=network_opt, metrics={"acc"})

    # valid callback
    eval_callback = EvaluateCallBack(model, eval_dataset=val_dataset, ckpt_url=train_dir)
    loss_monitor = LossMonitor(lr, per_print_times=step_size)

    # 训练
    model.train(args.epochs,
                train_dataset,
                callbacks=[ckpt_callback, loss_monitor, eval_callback],
                dataset_sink_mode=False)
    
    # sync_model(args)
    

"""控制台输出记录到文件"""
class Logger(object):
    def __init__(self, file_name="Default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

    
if __name__ == '__main__':
    
    args = get_config()
    
    # 自定义目录存放日志文件
    log_path = args.log_url 
    os.makedirs(log_path, exist_ok=True)
    
    # 日志文件名按照程序运行时间设置
    log_file_name = os.path.join(log_path, 'log-' + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.log')
    
    # 记录正常的 print 信息
    sys.stdout = Logger(log_file_name)
    # 记录 traceback 异常信息
    sys.stderr = Logger(log_file_name)
    
    # 开始训练
    train(args)
