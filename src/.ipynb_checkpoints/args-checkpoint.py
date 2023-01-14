"""global args"""
import argparse
import ast


def parse_arguments():
    """parse_arguments"""
    parser = argparse.ArgumentParser(description="MindSpore Training")

    # path
    parser.add_argument('--data_url', default="./dataset", help='Location of data.')
    parser.add_argument('--train_url', default='./resnet101_ckpt', help='model to save/load')
    parser.add_argument('--fine_tune_ckpt_url', default='./resnet101_fine_tune', help='fine tune model to save/load')
    parser.add_argument('--log_url', default='./log', help='log path')
    
    # dataset 
    parser.add_argument("--image_size", default=224, type=int, help="Image Size.")
    parser.add_argument("--val_percent", default=0.1, type=float, help="The percentage of validation sets")

    # device
    parser.add_argument("--device_id", default=0, type=int, help="Device Id")
    parser.add_argument("--device_num", default=1, type=int, help="device num")
    parser.add_argument("--device_target", default="Ascend", choices=["GPU", "Ascend", "CPU"], type=str)
    parser.add_argument("--graph_mode", default=0, type=int, help="graph mode with 0, python with 1")

    # training
    parser.add_argument("-a", "--arch", metavar="ARCH", default="resnet101", help="model architecture")
    parser.add_argument("--num_classes", default=54, type=int)
    parser.add_argument("--epochs", default=300, type=int, help="number of total epochs to run")
    parser.add_argument("--batch-size", default=64, type=int, help="batch size")
    parser.add_argument("--pretrained", default=True, type=bool, help="use pre-trained model")
    parser.add_argument("--seed", default=42, type=int, help="seed for initializing training. ")
    
    # argument
    parser.add_argument("--beta", default=[0.9, 0.999], type=lambda x: [float(a) for a in x.split(",")],
                        help="beta for optimizer")
    parser.add_argument("--crop", default=True, type=ast.literal_eval, help="Crop when testing")
    parser.add_argument("--interpolation", default="bicubic", type=str, help="bicubic")
    parser.add_argument("--auto_augment", default="rand-m9-mstd0.5-inc1", type=str, help="auto_augment")
    parser.add_argument("--re_prob", default=0.25, type=float, help="re prob")
    parser.add_argument("--re_count", default=1, type=float, help="re count")
    parser.add_argument("--re_mode", default="pixel", type=str, help="re mode")
    parser.add_argument("--keep_checkpoint_max", default=20, type=int, help="keep checkpoint max num")
    parser.add_argument("--mix_up", default=0.8, type=float, help="mix up")
    parser.add_argument("--cutmix", default=1.0, type=float, help="cutmix")
    parser.add_argument("--mixup_prob", default=1., type=float, help="muxup prob")
    parser.add_argument("--switch_prob", default=0.5, type=float, help="switch prob")
    parser.add_argument("--mixup_mode", default="batch", type=str, help="mixup mode")
    parser.add_argument("-j", "--num_parallel_workers", default=1, type=int, help="number of data loading workers (default: 20)")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Label smoothing to use, default 0.0")

    # optimiter
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")

    args = parser.parse_args()

    return args


def get_config():
    """get_config"""

    args = parse_arguments()
    return args
