import os
import re

from dataclasses import dataclass

import datasets
import torch
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
    """ Did a simple workaround to use almost the same training structure as DSI
    - Treat Image as Visual Tokens, and use codebook directly from T5 pretrained word embeddings to find the closest visual token
    - This is done in a no-grad manner, so the codebook is not updated during training
    - This could be(probably should be) done better by pretraining this image captioning/tokenization task, 
      and both image embeddings and codebook should be linear projected and then caculate the distance
      Similar ideas are used in https://link.zhihu.com/?target=https%3A//openreview.net/forum%3Fid%3DLt8bMlhiwx2
      and https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2209.15162
      
      Training and Inference are the same as DSI

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, filename, ann_root , image_processor, visual_encoder, tokenizer, codebook, max_words=32, prompt='question: '):        
        self.annotation = datasets.load_dataset(
            'json',
            data_files=os.path.join(ann_root,filename),
            ignore_verifications=False,
        )['train']
        
        self.transform = transforms.Compose([                        
                    transforms.RandomResizedCrop(224,scale=(0.5, 1.0),interpolation=InterpolationMode.BICUBIC),
                ])
        self.image_processor = image_processor
        self.codebook = codebook
        
        self.image_root = ann_root
        self.max_words = max_words      
        self.prompt = prompt
        self.tokenizer = tokenizer
        self.visual_encoder = visual_encoder
        
        self.img_ids = {}  
        n = 0
        for ann in self.annotation:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    
        
    def __len__(self):
        return len(self.annotation)
    
    def get_visual_token(self, img_emb, max_length=32):
        z_flattened = F.normalize(img_emb).view(-1, 768) # vit-base hidden dim is 768
        # ecuclidean distance between z_flattened and codebook
        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(self.codebook**2, dim=1) - \
            2*(torch.matmul(z_flattened, self.codebook.t()))

        min_encoding_indices = torch.argmin(d, dim=1)
        # for memory efficiency, randomly sample from the min_encoding_indices
        # can be done better by bottom-up attention
        rand_inds = torch.randint(len(min_encoding_indices), (max_length,))
        out = min_encoding_indices[rand_inds]
        out[-1] = self.tokenizer.eos_token_id 
        return out
    
    def __getitem__(self, index):    
        ann = self.annotation[index]
        if ann.get('image',None):
            image_path = os.path.join(self.image_root,ann['image'])        
            image = Image.open(image_path).convert('RGB')   
            image = self.image_processor(self.transform(image), return_tensors='pt')
            with torch.no_grad():
                image_emb = self.visual_encoder(**image)
            
            input_ids = self.get_visual_token(image_emb.last_hidden_state)
            # print("image input_ids:", input_ids.shape)
        else:
            caption = self.prompt+pre_caption(ann['caption']) 
            input_ids = self.tokenizer(caption,
                                   return_tensors="pt",
                                   truncation='only_first',
                                   max_length=self.max_words).input_ids[0]
            # print("caption input_ids:", input_ids.shape)
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
        input_ids = [{'input_ids': x[0]} for x in features]
        docids = [x[1] for x in features]
        inputs = super().__call__(input_ids)
        
        # treat docids as sequence of tokens --- naive structured string identifier
        labels = self.tokenizer(
            docids, padding="longest", return_tensors="pt"
        ).input_ids

        # replace padding token id's of the labels by -100 according to https://huggingface.co/docs/transformers/model_doc/t5#training
        labels[labels == self.tokenizer.pad_token_id] = -100
        inputs['labels'] = labels
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
        input_ids = [{'input_ids': x[0]} for x in features]
        labels = [x[1] for x in features]
        inputs = super().__call__(input_ids)

        return inputs, labels

    
if __name__ == "__main__":
    from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, TrainerCallback,AutoImageProcessor, ViTModel
    from torch.utils.data import DataLoader

    image_processor = AutoImageProcessor.from_pretrained("google/vit-large-patch16-224-in21k")
    # vit = ViTModel.from_pretrained("google/vit-large-patch16-224-in21k")
    
    model_name = "t5-large"
    L = 32  # only use the first 32 tokens of documents (including title)
    t5_tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir='cache')
    # t5_model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir='cache')
    
    # codebook = t5_model.get_input_embeddings().weight
    # codebook.requires_grad_(False)
    # codebook_norm = F.normalize(codebook)
    
    train_dataset = flickr30k_i2t_train('flickr30k_i2t_train.json', '../flickr30k_images',image_processor)
    print(train_dataset[0])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, collate_fn=ImageIndexingCollator(t5_tokenizer, padding='longest'))
    i = next(iter(train_loader))
    print("Train Batch:",i, i['images'].shape, i['labels'].shape)
    # train_dataset = flickr30k_train('flickr30k_multi_task_train.json', '../flickr30k_images',image_processor,vit,t5_tokenizer,codebook_norm)
    # val_dataset = flickr30k_train('flickr30k_valid.json', '../flickr30k_images',image_processor,vit, t5_tokenizer,codebook_norm) 
    # # test_dataset = flickr30k_retrieval_eval(transform_train, '../flickr30k_images/', '../flickr30k_images', 'test')          
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, collate_fn=IndexingCollator(t5_tokenizer, padding='longest'))
    # i = next(iter(train_loader))
    # print("Train Batch:",i)
    # val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, collate_fn=QueryEvalCollator(t5_tokenizer, padding='longest'))
    # y = next(iter(val_loader))
    # # print(len(i), i['input_ids'].dim(), i['images'].shape, i['labels'].shape, i['task_type'].shape, i['task_type'])
    # print("Eval Batch:",y)
    
    # image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    # model = ViTModel.from_pretrained("google/vit-large-patch16-224-in21k")
    # print(model(train_dataset[0][0].unsqueeze(0)).last_hidden_state.shape)
    
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='cache')
    # id_token = tokenizer(str(train_dataset[0][2]), padding="longest", return_tensors="pt").input_ids
    # print(id_token)