# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torch.nn.parallel import DataParallel, DistributedDataParallel
import torch.nn as nn



class EnsembleGLModel(nn.Module):
    def __init__(self, modelGlobal, modelLocal, modelLocalPrev):
        super(EnsembleGLModel, self).__init__()

        if isinstance(modelGlobal, (DistributedDataParallel, DataParallel)):
            modelGlobal = modelGlobal.module
        if isinstance(modelLocal, (DistributedDataParallel, DataParallel)):
            modelLocal = modelLocal.module
        if isinstance(modelLocalPrev, (DistributedDataParallel, DataParallel)):
            modelLocalPrev = modelLocalPrev.module

        self.modelGlobal = modelGlobal
        self.modelLocal = modelLocal
        self.modelLocalPrev = modelLocalPrev


