import torch.nn as nn
import torch.nn.functional as F

from pdma.base import ModelBase
from pdma.utils import setup_logger


log = setup_logger(__name__)


class BERTClassifier(ModelBase):
    """ Input: number of classes
        - BERT hidden size is 768 
        - Pooled output: last_hidden_state of token `[CLS]` for classification task
        Output: raw output from after dropout, fc layer
    """
    
    def __init__(self, n_classes):
        super().__init__()
        PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.2)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.drop(pooled_output)
        return self.out(output)
        


# class MnistModel(ModelBase):
#     def __init__(self, num_classes=10):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, num_classes)
# 
#         log.info(f'<init>: \n{self}')
# 
#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)

        