# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""callback function"""

import os
from mindspore import save_checkpoint
from mindspore.communication.management import get_rank
from mindspore.train.callback import Callback


class EvaluateCallBack(Callback):
    """EvaluateCallBack"""

    def __init__(self, model, eval_dataset, ckpt_url):
        super(EvaluateCallBack, self).__init__()
        self.model = model
        self.eval_dataset = eval_dataset
        self.ckpt_url = ckpt_url
        self.best_acc = 0.
        # self.rank = get_rank()
        self.rank = 0

    def epoch_end(self, run_context):
        """
            Test when epoch end, save best model with best.ckpt.
        """
        cb_params = run_context.original_args()
        result = self.model.eval(self.eval_dataset)
        
        if result["acc"] > self.best_acc:
            self.best_acc = result["acc"]
            self.best_acc_path = os.path.join(self.ckpt_url, "best.ckpt")
            save_checkpoint(cb_params.train_network, self.best_acc_path)

        print("ckpt_%s | epoch: %s acc: %s, best acc is %s" %
              (self.rank, cb_params.cur_epoch_num, result["acc"], self.best_acc), flush=True)