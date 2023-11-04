import datasets
import random
import numpy as np
import json, os

random.seed(313)

ann_root = '../flickr30k_images'
filename = 'flickr30k_train.json'
filename_val = 'flickr30k_val.json'
        
annotation = json.load(open(os.path.join(ann_root,filename),'r'))
img_set = set()

NUM_TRAIN = 8000
NUM_EVAL = 2000

rand_inds = list(range(len(annotation)))
random.shuffle(rand_inds)

current_docid = 0

with open('flickr30k_i2t_train.json', 'w') as tf, \
     open('flickr30k_t2i_train.json','w') as ttf, \
        open('flickr30k_multi_task_train.json','w') as mtf,\
         open('flickr30k_t2i_valid.json', 'w') as vf:
            for i in rand_inds:
                img_name = annotation[i]['image']
                #title = data[ind]['document']['title']  # we use title as the doc identifier to prevent two docs have the same text
                # title = data[ind]['title']  # we use title as the doc identifier to prevent two docs have the same text
                if img_name not in img_set:
                    img_set.add(img_name)

                    jitem = json.dumps({'image_id': str(current_docid), 'image': annotation[i]['image']})
                    
                    tf.write(jitem + '\n')
                    mtf.write(jitem + '\n')
                        
                    jitem = json.dumps({'image_id': str(current_docid), 'caption': annotation[i]['caption']})
                    # ttf.write(jitem + '\n')

                    ttf.write(jitem + '\n')
                    if len(img_set) > NUM_TRAIN:
                        vf.write(jitem + '\n')
                    else:
                        mtf.write(jitem + '\n')
                    current_docid += 1
                    if len(img_set) == NUM_TRAIN + NUM_EVAL:
                        break
                print(f"Creating training and validation dataset: {'{:.1%}'.format(len(img_set)/(NUM_TRAIN + NUM_EVAL))}", end='\r')

