import torch
import datasets
import open_clip
from open_clip import tokenizer
import os, re
from PIL import Image
from tqdm import tqdm

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

train_filename = "flickr30k_multi_task_train.json"
test_filename = 'flickr30k_valid.json'
ann_root = "../flickr30k_images"
image_root = ann_root

model, _, image_processor = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s34b_b88k')
model.cuda().eval()

train_annotation = datasets.load_dataset(
            'json',
            data_files=os.path.join(ann_root,train_filename),
            ignore_verifications=False,
        )['train']

test_annotation = datasets.load_dataset(
            'json',
            data_files=os.path.join(ann_root,test_filename),
            ignore_verifications=False,
        )['train']

print(train_annotation, test_annotation)

prompt = "question: "

train_img_ids = {}  
test_img_ids = {}  

n = 0
for ann in train_annotation:
    img_id = ann['image_id']
    if img_id not in train_img_ids.keys():
        train_img_ids[img_id] = n
        n += 1 
        
n = 0
for ann in test_annotation:
    img_id = ann['image_id']
    if img_id not in test_img_ids.keys():
        test_img_ids[img_id] = n
        n += 1 

print("train img ids:", len(train_img_ids))
print("test img ids:", len(test_img_ids))
"""
train_input_ids = []  
for i, ann in enumerate(tqdm(train_annotation, desc="Train_Input_Ids")):
    if ann.get('image',None):
        print("Processing Image {}".format(i+1))
        image_path = os.path.join(image_root,ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = image_processor(image).unsqueeze(0)
        with torch.no_grad():
            input_ids = model.encode_image(image.cuda()).float()
    else:
        print("Processing Text {}".format(i+1))
        caption = prompt+pre_caption(ann['caption']) 
        text_tok = tokenizer.tokenize(caption)
        with torch.no_grad():
            input_ids = model.encode_text(text_tok.cuda()).float()

    train_input_ids.append(input_ids)

train_input_ids_cat = torch.concat(train_input_ids, dim=0)
print("Train Input Ids:", train_input_ids_cat.shape)

torch.save(train_input_ids_cat, "train_prec_large.pt")
"""
test_input_ids = []  
for i, ann in enumerate(tqdm(test_annotation, desc="Test_Input_Ids")):
    if ann.get('image',None):
        print("Processing Image {}".format(i+1))
        image_path = os.path.join(image_root,ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = image_processor(image).unsqueeze(0)
        with torch.no_grad():
            input_ids = model.encode_image(image.cuda()).float()
    else:
        print("Processing Text {}".format(i+1))
        caption = prompt+pre_caption(ann['caption']) 
        text_tok = tokenizer.tokenize(caption)
        with torch.no_grad():
            input_ids = model.encode_text(text_tok.cuda()).float()

    test_input_ids.append(input_ids)

test_input_ids_cat = torch.concat(test_input_ids, dim=0)
print("Test Input Ids:", test_input_ids_cat.shape)
torch.save(test_input_ids_cat, "test_prec_large.pt")
# model, image_processor = clip.load("ViT-L/14")
# model.cuda().eval()