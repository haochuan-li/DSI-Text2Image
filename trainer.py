from typing import Dict, List, Tuple, Optional, Any, Union
from transformers.trainer import Trainer
from torch import nn
import torch

class IndexingTrainer(Trainer):
    def __init__(self, restrict_decode_vocab, mode, visual_encoder=None, **kwds):
        super().__init__(**kwds)
        self.restrict_decode_vocab = restrict_decode_vocab
        self.mode = mode
    

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.mode == 'i2t':
            im = inputs['images']
            with torch.no_grad():
                image_embeds = self.visual_encoder(im).last_hidden_state.unsqueeze(0) # (1, batch_size, num_patches, embed_size)
            print("image_embeds:", image_embeds.shape)
            
            loss = model(encoder_outputs=image_embeds, labels=inputs['labels']).loss
        elif self.mode == 't2i':
            # print("Clip Encoder Outputs:", inputs['input_ids'].device)
            # loss = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['labels']).loss
            loss = model(inputs_embeds=inputs['input_ids'], labels=inputs['labels']).loss
            # loss = model(encoder_outputs=(inputs['input_ids'],), labels=inputs['labels']).loss
        if return_outputs:
            return loss, [None, None]  # fake outputs
        return loss

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool = False,
            ignore_keys: Optional[List[str]] = None,
        ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        model.eval()
        # eval_loss = super().prediction_step(model, inputs, True, ignore_keys)[0]
        with torch.no_grad():
            # greedy search
            doc_ids = model.generate(
                inputs_embeds = inputs['input_ids'].to(model.device),
                # encoder_outputs = (inputs['input_ids'].to(model.device),),
                # inputs['input_ids'].to(self.args.device),
                max_length=20,
                # num_beams=3,
                # num_return_sequences=3,
                prefix_allowed_tokens_fn=self.restrict_decode_vocab,
                early_stopping=True,)
            
        return (None, doc_ids, inputs['labels'])
