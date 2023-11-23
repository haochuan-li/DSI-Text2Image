from flickr30k_data import flickr30k_train, IndexingCollator, QueryEvalCollator, flickr30k_i2t_train, ImageIndexingCollator, ImageQueryEvalCollator
from transformers import T5Tokenizer, T5ForConditionalGeneration, PrinterCallback,TrainingArguments, TrainerCallback, AutoImageProcessor, ViTModel
from trainer import IndexingTrainer
import numpy as np
import torch
import wandb
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Evaluation callback for the dev set.
class QueryEvalCallback(TrainerCallback):
    def __init__(self, test_dataset, logger, restrict_decode_vocab, args: TrainingArguments, tokenizer: T5Tokenizer):
        self.tokenizer = tokenizer
        self.logger = logger
        self.args = args
        self.test_dataset = test_dataset
        self.restrict_decode_vocab = restrict_decode_vocab
        self.dataloader = DataLoader(
            test_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=QueryEvalCollator(
                self.tokenizer,
                padding='longest'
            ),
            shuffle=False,
            drop_last=False,
            num_workers=self.args.dataloader_num_workers,
        )

    def on_epoch_end(self, args, state, control, **kwargs):
        hit_at_1 = 0
        hit_at_10 = 0
        model = kwargs['model'].eval()
        
        for batch in tqdm(self.dataloader, desc='Evaluating dev queries'):
            inputs, labels = batch
            with torch.no_grad():
                batch_beams = model.generate(
                    inputs_embeds = inputs['input_ids'].to(model.device),
                    # encoder_outputs = (inputs['input_ids'].to(model.device),),
                    max_length=20,
                    num_beams=10,
                    prefix_allowed_tokens_fn=self.restrict_decode_vocab,
                    num_return_sequences=10,
                    early_stopping=True, 
                    ).reshape(inputs['input_ids'].shape[0], 10, -1)
                for beams, label in zip(batch_beams, labels):
                    rank_list = self.tokenizer.batch_decode(beams,
                                                            skip_special_tokens=True)  # beam search should not return repeated docids but somehow due to T5 tokenizer there some repeats.
                    print("Predict List:",rank_list, "Label:",label)
                    hits = np.where(np.array(rank_list)[:10] == label)[0]
                    if len(hits) != 0:
                        hit_at_10 += 1
                        if hits[0] == 0:
                            hit_at_1 += 1
        self.logger.log({"Hits@1": hit_at_1 / len(self.test_dataset), "Hits@10": hit_at_10 / len(self.test_dataset)})
        
class ImageQueryEvalCallback(TrainerCallback):
    def __init__(self, test_dataset, logger, restrict_decode_vocab, args: TrainingArguments, tokenizer: T5Tokenizer, visual_encoder: ViTModel):
        self.tokenizer = tokenizer
        self.visual_encoder = visual_encoder
        self.logger = logger
        self.args = args
        self.test_dataset = test_dataset
        self.restrict_decode_vocab = restrict_decode_vocab
        self.dataloader = DataLoader(
            test_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=ImageQueryEvalCollator(
                self.tokenizer,
                padding='longest'
            ),
            shuffle=False,
            drop_last=False,
            num_workers=self.args.dataloader_num_workers,
        )

    def on_step_end(self, args, state, control, **kwargs):
        hit_at_1 = 0
        hit_at_10 = 0
        model = kwargs['model'].eval()
        
        for batch in tqdm(self.dataloader, desc='Evaluating dev queries'):
            inputs, labels = batch
            with torch.no_grad():
                image_embeds = self.visual_encoder(inputs['images']).last_hidden_state
            with torch.no_grad():
                batch_beams = model.generate(
                    inputs_embeds=image_embeds.to(model.device),
                    max_length=20,
                    num_beams=10,
                    prefix_allowed_tokens_fn=self.restrict_decode_vocab,
                    num_return_sequences=10,
                    early_stopping=True, ).reshape(inputs['images'].shape[0], 10, -1)
                for beams, label in zip(batch_beams, labels):
                    rank_list = self.tokenizer.batch_decode(beams,
                                                            skip_special_tokens=True)  # beam search should not return repeated docids but somehow due to T5 tokenizer there some repeats.
                    print("Predict List:",rank_list, "Label:",label)
                    hits = np.where(np.array(rank_list)[:10] == label)[0]
                    if len(hits) != 0:
                        hit_at_10 += 1
                        if hits[0] == 0:
                            hit_at_1 += 1
        self.logger.info({"Hits@1": hit_at_1 / len(self.test_dataset), "Hits@10": hit_at_10 / len(self.test_dataset)})


def compute_metrics(eval_preds):
    num_predict = 0
    num_correct = 0
    for predict, label in zip(eval_preds.predictions, eval_preds.label_ids):
        num_predict += 1
        if len(np.where(predict == 1)[0]) == 0:
            continue
        if np.array_equal(label[:np.where(label == 1)[0].item()],
                          predict[np.where(predict == 0)[0][0].item() + 1:np.where(predict == 1)[0].item()]):
            num_correct += 1

    return {'accuracy': num_correct / num_predict}


def main_t2i():
    print("Running text2image Retrieval...")
    
    key = 'cab22b56672abca555605b07536a36a2c5c4ef39'
    wandb.login(key=key)
    wandb.login()
    wandb.init(project="DSI", name='NQ-10k-t5-large_i2t')
    
    model_name = "t5-large"
    
    t5_tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir='cache')
    t5_model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir='cache')

    
    train_dataset = flickr30k_train('flickr30k_multi_task_train.json', '../flickr30k_images', './train_prec_large.pt')
    eval_dataset = flickr30k_train('flickr30k_multi_task_train.json', '../flickr30k_images', './train_prec_large.pt') 
    
    # This is the actual eval set.
    test_dataset = flickr30k_train('flickr30k_valid.json', '../flickr30k_images', './test_prec_large.pt') 

    ################################################################
    # docid generation constrain, we only generate integer docids.
    SPIECE_UNDERLINE = "▁"
    INT_TOKEN_IDS = []
    # print("Vob:", tokenizer.get_vocab().items())
    for token, id in t5_tokenizer.get_vocab().items():
        if token[0] == SPIECE_UNDERLINE:
            if token[1:].isdigit():
                INT_TOKEN_IDS.append(id)
        if token == SPIECE_UNDERLINE:
            INT_TOKEN_IDS.append(id)
        elif token.isdigit():
            INT_TOKEN_IDS.append(id)
    INT_TOKEN_IDS.append(t5_tokenizer.eos_token_id)

    # print(len(INT_TOKEN_IDS), max(INT_TOKEN_IDS), min(INT_TOKEN_IDS))
    
    def restrict_decode_vocab(batch_idx, prefix_beam):
        return INT_TOKEN_IDS
    ################################################################

    # key is in Collator, concurrenctly training with doc -> docid and query -> docid
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=0.0005,
        warmup_steps=10000,
        # weight_decay=0.01,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        evaluation_strategy='steps',
        eval_steps=1000,
        # eval_steps=10,
        max_steps=1000000,
        dataloader_drop_last=False,  # necessary
        report_to='wandb',
        logging_steps=50,
        save_strategy='no',
        # fp16=True,  # gives 0/nan loss at some point during training, seems this is a transformers bug.
        dataloader_num_workers=10,
        # gradient_accumulation_steps=2
    )

    trainer = IndexingTrainer(
        model=t5_model,
        tokenizer=t5_tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=IndexingCollator(
            t5_tokenizer,
            padding='longest',
        ),
        compute_metrics=compute_metrics,
        callbacks=[QueryEvalCallback(test_dataset, wandb, restrict_decode_vocab, training_args, t5_tokenizer)],
        restrict_decode_vocab=restrict_decode_vocab,
        mode='t2i',
    )
    
    trainer.train()

def main_i2t():
    print("Running image2text Indexing...")
    model_name = "t5-base"

    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    
    t5_tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir='cache')
    t5_model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir='cache')
    
    train_dataset = flickr30k_i2t_train('flickr30k_i2t_train.json', '../flickr30k_images',image_processor)
    eval_dataset = flickr30k_i2t_train('flickr30k_i2t_train.json', '../flickr30k_images',image_processor) 
    
    # This is the actual eval set.
    test_dataset = flickr30k_i2t_train('flickr30k_i2t_train.json', '../flickr30k_images',image_processor) 

    ################################################################
    # docid generation constrain, we only generate integer docids.
    SPIECE_UNDERLINE = "▁"
    INT_TOKEN_IDS = []
    # print("Vob:", tokenizer.get_vocab().items())
    for token, id in t5_tokenizer.get_vocab().items():
        if token[0] == SPIECE_UNDERLINE:
            if token[1:].isdigit():
                INT_TOKEN_IDS.append(id)
        if token == SPIECE_UNDERLINE:
            INT_TOKEN_IDS.append(id)
        elif token.isdigit():
            INT_TOKEN_IDS.append(id)
    INT_TOKEN_IDS.append(t5_tokenizer.eos_token_id)

    # print(len(INT_TOKEN_IDS), max(INT_TOKEN_IDS), min(INT_TOKEN_IDS))
    
    def restrict_decode_vocab(batch_idx, prefix_beam):
        return INT_TOKEN_IDS
    ################################################################

    # key is in Collator, concurrenctly training with doc -> docid and query -> docid
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=0.0005,
        warmup_steps=10000,
        # weight_decay=0.01,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy='steps',
        eval_steps=1,
        max_steps=1000000,
        dataloader_drop_last=False,  # necessary
        report_to='wandb',
        logging_steps=1,
        save_strategy='no',
        # fp16=True,  # gives 0/nan loss at some point during training, seems this is a transformers bug.
        dataloader_num_workers=0,
        # gradient_accumulation_steps=2
    )

    trainer = IndexingTrainer(
        model=t5_model,
        tokenizer=t5_tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=ImageIndexingCollator(
            t5_tokenizer,
            padding='longest',
        ),
        compute_metrics=compute_metrics,
        callbacks=[ImageQueryEvalCallback(test_dataset, wandb, restrict_decode_vocab, training_args, t5_tokenizer,vit)],
        restrict_decode_vocab=restrict_decode_vocab,
        mode='i2t',
    )
    
    trainer.train()

def test_inference():
    print("Running text2image Retrieval...")
    model_name = "t5-base"

    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    
    t5_tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir='cache')
    t5_model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir='cache')
    
    codebook = t5_model.get_input_embeddings().weight
    codebook.requires_grad_(False)
    codebook_norm = F.normalize(codebook)
    
    train_dataset = flickr30k_train('flickr30k_multi_task_train.json', '../flickr30k_images',image_processor, vit, t5_tokenizer,codebook_norm)
    eval_dataset = flickr30k_train('flickr30k_multi_task_train.json', '../flickr30k_images',image_processor, vit, t5_tokenizer,codebook_norm) 
    # This is the actual eval set.
    test_dataset = flickr30k_train('flickr30k_valid.json', '../flickr30k_images',image_processor,vit, t5_tokenizer,codebook_norm) 

    ################################################################
    # docid generation constrain, we only generate integer docids.
    SPIECE_UNDERLINE = "▁"
    INT_TOKEN_IDS = []
    # print("Vob:", tokenizer.get_vocab().items())
    for token, id in t5_tokenizer.get_vocab().items():
        if token[0] == SPIECE_UNDERLINE:
            if token[1:].isdigit():
                INT_TOKEN_IDS.append(id)
        if token == SPIECE_UNDERLINE:
            INT_TOKEN_IDS.append(id)
        elif token.isdigit():
            INT_TOKEN_IDS.append(id)
    INT_TOKEN_IDS.append(t5_tokenizer.eos_token_id)

    # print(len(INT_TOKEN_IDS), max(INT_TOKEN_IDS), min(INT_TOKEN_IDS))
    
    def restrict_decode_vocab(batch_idx, prefix_beam):
        return INT_TOKEN_IDS
    ################################################################

    # key is in Collator, concurrenctly training with doc -> docid and query -> docid
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=0.0005,
        warmup_steps=10000,
        # weight_decay=0.01,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        evaluation_strategy='steps',
        eval_steps=1000,
        max_steps=1000000,
        dataloader_drop_last=False,  # necessary
        report_to='wandb',
        logging_steps=5,
        save_strategy='no',
        # fp16=True,  # gives 0/nan loss at some point during training, seems this is a transformers bug.
        dataloader_num_workers=10,
        # gradient_accumulation_steps=2
    )

    trainer = IndexingTrainer(
        model=t5_model,
        tokenizer=t5_tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=IndexingCollator(
            t5_tokenizer,
            padding='longest',
        ),
        compute_metrics=compute_metrics,
        callbacks=[QueryEvalCallback(test_dataset, wandb, restrict_decode_vocab, training_args, t5_tokenizer)],
        restrict_decode_vocab=restrict_decode_vocab,
        mode = 't2i'
    )
    tdl = DataLoader(eval_dataset, batch_size=10, shuffle=True,collate_fn=IndexingCollator(t5_tokenizer, padding='longest'))
    print(next(iter(tdl)))
    i = next(iter(tdl))
    _,pred_docid, label_docid = trainer.prediction_step(
                                                    model=t5_model,
                                                    inputs=i
                                                )
    label_docid[label_docid==-100] = t5_tokenizer.pad_token_id
    # print(pred_docid, label_docid)
    print(t5_tokenizer.batch_decode(pred_docid, skip_special_tokens=True),t5_tokenizer.batch_decode(label_docid, skip_special_tokens=True))
    

if __name__ == "__main__":
    main_t2i()
    # main_i2t()
    # test_inference()
