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


class MultiTSModel(nn.Module):
    def __init__(self, modelTeacherList, modelStudent):
        super(MultiTSModel, self).__init__()
        
        temp_list = []
        for modelTeacher in modelTeacherList:
            if isinstance(modelTeacher, (DistributedDataParallel, DataParallel)):
                modelTeacher = modelTeacher.module
            temp_list.append(modelTeacher)
        
        self.modelTeacher = nn.ModuleList(temp_list)
        
        if isinstance(modelStudent, (DistributedDataParallel, DataParallel)):
            modelStudent = modelStudent.module
        self.modelStudent = modelStudent