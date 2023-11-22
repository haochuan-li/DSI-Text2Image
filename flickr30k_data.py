import os
import re

from dataclasses import dataclass

import datasets
import torch
import clip
import torch.nn.functional as F
from torchvision import transforms
from transformers import TrainingArguments, TrainerCallback, ViTModel,PreTrainedTokenizer, DataCollatorWithPadding
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset

from PIL import Image

def pre_caption(caption):
    caption = re.sub(
        r"([.!\"()*#:;~])",       
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    return caption


class flickr30k_train(Dataset):
    def __init__(self, filename, ann_root , prec_file=None):        
        self.annotation = datasets.load_dataset(
            'json',
            data_files=os.path.join(ann_root,filename),
            ignore_verifications=False,
        )['train']
        
        self.data = torch.load(prec_file, map_location='cpu')
        # print("Loading Precomputed Clip Embed:", self.data.shape)
        
        assert len(self.data) == len(self.annotation)
        
        self.img_ids = {}  
        n = 0
        for ann in self.annotation:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        ann = self.annotation[index]
        input_ids = self.data[index]
        
        # print("Input Ids:", input_ids.shape)
        return input_ids, str(self.img_ids[ann['image_id']]) 


class flickr30k_i2t_train(Dataset):
    """Use a different multi-modal interaction method, and it change from DSI's co-training to a two-step training
       Indexing: image-to-text(id)
       Retrieval: text-to-text(id) same as DSI
       This is done by leverage the encoder-decoder architecture of T5, and use the vit encoder to encode the image, 
       and use the vit encoder embedding to serve as Key-Value pairs for the T5 decoder to generate the text(id)
       Similar ideas is used in BLIP https://arxiv.org/abs/2201.12086
       
       Image-To-Text(Indexing) Training:
       Training:
       - Input: Image Embedding from ViT
       - Output: Image Id from T5
       Inference:
       - Input: Image Embedding from ViT
       - Output: Image Id from T5

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, filename, ann_root , image_processor, max_words=32, prompt='question: '):        
        self.annotation = datasets.load_dataset(
            'json',
            data_files=os.path.join(ann_root,filename),
            ignore_verifications=False,
        )['train']
        
        self.transform = transforms.Compose([                        
                    transforms.RandomResizedCrop(224,scale=(0.5, 1.0),interpolation=InterpolationMode.BICUBIC),
                ])
        
        self.image_processor = image_processor
        
        self.image_root = ann_root
        self.max_words = max_words      
        self.prompt = prompt
        
        self.img_ids = {}  
        n = 0
        for ann in self.annotation:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        ann = self.annotation[index]
        image_path = os.path.join(self.image_root,ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.image_processor(self.transform(image), return_tensors='pt').pixel_values
        # print("image input_ids:", image.shape)

        return image, str(self.img_ids[ann['image_id']]) 

@dataclass
class ImageIndexingCollator(DataCollatorWithPadding):
    def __call__(self, features):
        inputs = {}
        images = torch.concat([x[0] for x in features], dim=0)
        inputs['images'] = images
        imageids = [x[1] for x in features]
        
        # treat docids as sequence of tokens --- naive structured string identifier
        labels = self.tokenizer(
            imageids, padding="longest", return_tensors="pt"
        ).input_ids

        # replace padding token id's of the labels by -100 according to https://huggingface.co/docs/transformers/model_doc/t5#training
        labels[labels == self.tokenizer.pad_token_id] = -100
        inputs['labels'] = labels
        return inputs
    
@dataclass
class IndexingCollator(DataCollatorWithPadding):
    def __call__(self, features):
        # doc + query = features
        # features = [(input_ids, docid), (input_ids, docid), ...]
        # input_ids = [{'input_ids': x[0]} for x in features]
        inputs = {}
        docids = [x[1] for x in features]
        # inputs = super().__call__(input_ids)
        input_ids = torch.stack([x[0] for x in features], dim=0)
        input_ids /= input_ids.norm(dim=-1, keepdim=True)
        
        # print("Same as norm:", input_ids == input_ids_norm)
        # print("collate input ids:", input_ids.shape)
        # treat docids as sequence of tokens --- naive structured string identifier
        labels = self.tokenizer(
            docids, padding="longest", return_tensors="pt"
        ).input_ids

        # replace padding token id's of the labels by -100 according to https://huggingface.co/docs/transformers/model_doc/t5#training
        labels[labels == self.tokenizer.pad_token_id] = -100
        inputs['labels'] = labels
        inputs['input_ids'] = input_ids.unsqueeze(1)
        return inputs

@dataclass
class ImageQueryEvalCollator(DataCollatorWithPadding):
    def __call__(self, features):
        inputs = {}
        images = torch.concat([x[0] for x in features], dim=0)
        inputs['images'] = images
        labels = [x[1] for x in features]

        return inputs, labels
    
@dataclass
class QueryEvalCollator(DataCollatorWithPadding):
    def __call__(self, features):
        # input_ids = [{'input_ids': x[0]} for x in features]
        inputs = {}
        input_ids = torch.stack([x[0] for x in features], dim=0)
        input_ids /= input_ids.norm(dim=-1, keepdim=True)
        # print("Same as norm:", input_ids == input_ids_norm)
        labels = [x[1] for x in features]
        inputs['input_ids'] = input_ids.unsqueeze(1)

        return inputs, labels

    
if __name__ == "__main__":
    from transformers import T5Tokenizer, T5ForConditionalGeneration

    model_name = "t5-large"

    t5_tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir='cache')
    t5_model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir='cache')
    train_dataset = flickr30k_train('flickr30k_multi_task_train.json', '../flickr30k_images', './train_prec.pt')
    test_dataset = flickr30k_train('flickr30k_valid.json', '../flickr30k_images', './test_prec.pt') 

    print(train_dataset[0],test_dataset[0])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, collate_fn=IndexingCollator(t5_tokenizer, padding='longest'))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, collate_fn=QueryEvalCollator(t5_tokenizer, padding='longest'))
    i = next(iter(train_loader))
    j = next(iter(test_loader))
    print("Train Batch:",i['input_ids'].shape, i['labels'].shape)
    print("Test Batch:",i['input_ids'].shape, i['labels'].shape)
    