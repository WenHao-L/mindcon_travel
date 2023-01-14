import os
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank


def set_device(args):
    """Set device and ParallelMode(if device_num > 1)"""
    rank = 0
    # set context and device
    device_target = args.device_target
    device_num = int(os.environ.get("DEVICE_NUM", 1))

    if device_target == "Ascend":
        if device_num > 1:
            context.set_context(device_id=int(os.getenv('DEVICE_ID', 0)))
            init(backend_name='hccl')
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            # context.set_auto_parallel_context(pipeline_stages=2, full_batch=True)

            rank = get_rank()
        else:
            context.set_context(device_id=args.device_id)
    elif device_target == "GPU":
        if device_num > 1:
            init(backend_name='nccl')
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            rank = get_rank()
        else:
            context.set_context(device_id=args.device_id)
    elif device_target == "CPU":
        pass
    else:
        raise ValueError("Unsupported platform.")

    return rank