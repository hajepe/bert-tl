from torch.utils.data import Dataset, DataLoader


class ProcessDataset(Dataset):
  def __init__(self, chunks, labels, tokenizer, max_len):
    self.chunks = chunks
    self.labels = labels
    self.tokenizer = tokenizer
    self.max_len = max_len
  def __len__(self):
    return len(self.chunks)
  def __getitem__(self, item):
    review = str(self.chunks[item])
    target = self.labels[item]
    encoding = self.tokenizer.encode_plus(
      review,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
    )
    return {
      'review_text': review,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'labels': torch.tensor(target, dtype=torch.long)
    }


def create_data_loader(df, tokenizer, max_len, batch_size):
  ds = ProcessDataset(
    reviews=df.content.to_numpy(),
    targets=df.sentiment.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
  )
  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=4
  )
  
# class AugmentationFactoryBase(abc.ABC):
#     def build_transforms(self, train):
#         return self.build_train() if train else self.build_test()
# 
#     @abc.abstractmethod
#     def build_train(self):
#         pass
# 
#     @abc.abstractmethod
#     def build_test(self):
#         pass
# 
# 
# class MNISTTransforms(AugmentationFactoryBase):
# 
#     MEANS = [0]
#     STDS = [1]
# 
#     def build_train(self):
#         return T.Compose([T.ToTensor(), T.Normalize(self.MEANS, self.STDS)])
# 
#     def build_test(self):
#         return T.Compose([T.ToTensor(), T.Normalize(self.MEANS, self.STDS)])
