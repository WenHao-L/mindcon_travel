# Data operations, will be used in train.py and eval.py

import os
import glob
import shutil
import random

import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision
from mindspore.dataset.vision.utils import Inter

from .augment.auto_augment import _pil_interp, rand_augment_transform
from .augment.mixup import Mixup
from .augment.random_erasing import RandomErasing


class TravelImageDataset:
    """FoodImageDataset Define"""

    def __init__(self, args, training=True):

        img_label_dir = os.path.join(args.data_url, "img_label")
        images_dir = os.path.join(args.data_url, "images")
        os.makedirs(images_dir, exist_ok=True)

        # 提取数据为imagedataset格式
        label_nums_list = list(range(args.num_classes))
        for label_num in label_nums_list:
            images_subdir = os.path.join(images_dir, str(label_num))
            os.makedirs(images_subdir, exist_ok=True)

        img_label_list = os.listdir(img_label_dir)
        img_label_list = sorted(img_label_list)
        # print(img_label_list[0:6])
        for i in range(len(img_label_list)):
            if i % 2 == 1:
                txt_path = os.path.join(img_label_dir, img_label_list[i])
                # print(txt_path)
                with open(txt_path, "r") as f:
                    for line in f:
                        comma_p = line.find(",")
                        label = str(line[comma_p+2:])
                        img_path = os.path.join(img_label_dir, line[:comma_p])

                        dst_path = os.path.join(images_dir, label)
                        # print(dst_path)
                        shutil.copy(img_path, dst_path)
                    f.close()

        train_dir = os.path.join(args.data_url, "train")
        val_dir = os.path.join(args.data_url, "val")

        # 划分训练集和验证集
        print("===================== 划分训练集和验证集 =====================")
        print("== 验证集百分比: ", args.val_percent)

        for class_dir_name in os.listdir(images_dir):
            images_path_list = glob.glob(os.path.join(images_dir, class_dir_name, "*"))
            num_list = [i for i in range(len(images_path_list))]
            random.shuffle(num_list)
            train_nums = int(len(images_path_list) * (1 - args.val_percent))

            dst_train_dir = os.path.join(train_dir, class_dir_name)
            dst_val_dir = os.path.join(val_dir, class_dir_name)
            os.makedirs(dst_train_dir, exist_ok=True)
            os.makedirs(dst_val_dir, exist_ok=True)

            for i in num_list[0:train_nums]:
                (file_path, file_name) = os.path.split(images_path_list[i])
                dst_train_path = os.path.join(dst_train_dir, file_name)
                shutil.copy(images_path_list[i], dst_train_path)

            for i in num_list[train_nums:len(images_path_list)]:
                (file_path, file_name) = os.path.split(images_path_list[i])
                dst_val_path = os.path.join(dst_val_dir, file_name)
                shutil.copy(images_path_list[i], dst_val_path)
            
        print("========================= 划分完成！ =========================")

        if training:
            self.train_dataset = self.create_dataset_imagenet(train_dir, training=True, args=args)
            self.images_dataset = self.create_dataset_imagenet(images_dir, training=True, args=args)
        self.val_dataset = self.create_dataset_imagenet(val_dir, training=False, args=args)


    def create_dataset_imagenet(self, dataset_dir, args, repeat_num=1, training=True):
        """
        create a train or eval food dataset

        Args:
            dataset_dir(string): the path of dataset.
            do_train(bool): whether dataset is used for train or eval.
            repeat_num(int): the repeat times of dataset. Default: 1

        Returns:
            dataset
        """

        class_indexing = {str(i):i for i in range(args.num_classes)}
        device_num, rank_id = self._get_rank_info()
        shuffle = bool(training)
        if device_num == 1 or not training:
            data_set = ds.ImageFolderDataset(dataset_dir, num_parallel_workers=args.num_parallel_workers, 
                                             class_indexing=class_indexing, shuffle=shuffle)
        else:
            data_set = ds.ImageFolderDataset(dataset_dir, num_parallel_workers=args.num_parallel_workers, 
                                             class_indexing=class_indexing, shuffle=shuffle, num_shards=device_num, shard_id=rank_id)

        image_size = args.image_size

        # define map operations
        # BICUBIC: 3

        if training:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            aa_params = dict(
                translate_const=int(image_size * 0.45),
                img_mean=tuple([min(255, round(255 * x)) for x in mean]),
            )
            interpolation = args.interpolation
            auto_augment = args.auto_augment
            assert auto_augment.startswith('rand')
            aa_params['interpolation'] = _pil_interp(interpolation)

            transform_img = [
                vision.RandomCropDecodeResize(image_size, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3),
                                            interpolation=Inter.BICUBIC),
                vision.RandomHorizontalFlip(prob=0.5),
                vision.ToPIL()
            ]
            transform_img += [rand_augment_transform(auto_augment, aa_params)]
            transform_img += [
                vision.ToTensor(),
                vision.Normalize(mean=mean, std=std, is_hwc=False),
                RandomErasing(args.re_prob, mode=args.re_mode, max_count=args.re_count)
            ]
        else:
            mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
            std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
            # test transform complete
            if args.crop:
                transform_img = [
                    vision.Decode(),
                    vision.Resize(int(256 / 224 * image_size), interpolation=Inter.BICUBIC),
                    vision.CenterCrop(image_size),
                    vision.Normalize(mean=mean, std=std, is_hwc=True),
                    vision.HWC2CHW()
                ]
            else:
                transform_img = [
                    vision.Decode(),
                    vision.Resize(int(image_size), interpolation=Inter.BICUBIC),
                    vision.Normalize(mean=mean, std=std, is_hwc=True),
                    vision.HWC2CHW()
                ]

        transform_label = transforms.TypeCast(mstype.int32)

        data_set = data_set.map(input_columns="image", num_parallel_workers=args.num_parallel_workers,
                                operations=transform_img)
        data_set = data_set.map(input_columns="label", num_parallel_workers=args.num_parallel_workers,
                                operations=transform_label)
        
        if (args.mix_up > 0. or args.cutmix > 0.)  and not training:
            # if use mixup and not training(False), one hot val data label
            one_hot = transforms.OneHot(num_classes=args.num_classes)
            data_set = data_set.map(input_columns="label", num_parallel_workers=args.num_parallel_workers,
                                    operations=one_hot)
        # apply batch operations
        data_set = data_set.batch(args.batch_size, drop_remainder=True,
                                num_parallel_workers=args.num_parallel_workers)

        if (args.mix_up > 0. or args.cutmix > 0.) and training:
            mixup_fn = Mixup(
                mixup_alpha=args.mix_up, cutmix_alpha=args.cutmix, cutmix_minmax=None,
                prob=args.mixup_prob, switch_prob=args.switch_prob, mode=args.mixup_mode,
                label_smoothing=args.label_smoothing, num_classes=args.num_classes)

            data_set = data_set.map(operations=mixup_fn, input_columns=["image", "label"],
                                    num_parallel_workers=args.num_parallel_workers)
        if not training:
            transform_label_val = transforms.TypeCast(mstype.float32)
            data_set = data_set.map(input_columns="label", num_parallel_workers=args.num_parallel_workers,
                                    operations=transform_label_val)
            
        # apply dataset repeat operation
        data_set = data_set.repeat(repeat_num)

        return data_set


    def _get_rank_info(self):
        """
        get rank size and rank id
        """
        rank_size = int(os.environ.get("RANK_SIZE", 1))

        if rank_size > 1:
            from mindspore.communication.management import get_rank, get_group_size
            rank_size = get_group_size()
            rank_id = get_rank()
        else:
            rank_size = rank_id = None  # =None?

        return rank_size, rank_id
