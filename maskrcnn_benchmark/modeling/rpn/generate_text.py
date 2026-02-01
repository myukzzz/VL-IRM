import torch
import torch.nn.functional as F
from torch import nn

from transformers import T5TokenizerFast

from .modeling_t5 import T5Config, T5ForConditionalGeneration



class Generate_with_T5(nn.Module):
    def __init__(
            self,
            vit_model="eva_clip_g",
            img_size=224,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision="fp16",
            freeze_vit=True,
            num_query_token=32,
            prompt="",
            max_txt_len=32,
            apply_lemmatizer=False,
            cfg=None,
    ):
        super().__init__()

        t5_model = 'google/flan-t5-base'
        generate_loss_weight = 1.0
        use_focal_loss = False
        self.use_all_negative = True

        self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model)
        t5_config = T5Config.from_pretrained(t5_model)
        t5_config.dense_act_fn = "gelu"
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            t5_model, config=t5_config
        )


        self.max_txt_len = max_txt_len

        self.t5_proj = nn.Linear(
            256, self.t5_model.config.hidden_size
        )
        self.generate_loss_weight = generate_loss_weight
        self.use_focal_loss = use_focal_loss

    def forward(self, object_features, object_descriptions, object_features_att_mask,rel_weight=None):
        inputs_t5 = self.t5_proj(object_features)  ##40 1 256-768
        atts_t5 = object_features_att_mask

        output_tokens = self.t5_tokenizer(
            object_descriptions,
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,  ##32
            return_tensors="pt",
        ).to(object_features.device)  ##description
        output_tokens.data['input_ids']
        encoder_atts = atts_t5

        targets = output_tokens.input_ids.masked_fill(
            output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
        )  ###39 4
        inputs_embeds = inputs_t5
        loss = {}

        #########################
        rel_weight_new=[]
        # if rel_weight is not None:
        #     for idx,tokens in enumerate(output_tokens.data['input_ids']):
        #         rel_weight_new.append(rel_weight[idx].repeat(len(tokens)))
        #     rel_weight_all=torch.cat(rel_weight_new)
        # else:
        #     rel_weight_all=None
        ###################################

        # outputs = self.t5_model(
        #     inputs_embeds=inputs_embeds,
        #     attention_mask=encoder_atts,
        #     decoder_attention_mask=output_tokens.attention_mask,
        #     return_dict=True,
        #     labels=targets,
        #     use_focal_loss=self.use_focal_loss,  # false
        #     rel_weight=rel_weight_all,
        # )
        #
        outputs = self.t5_model(
            inputs_embeds=inputs_embeds,
            attention_mask=encoder_atts,
            decoder_attention_mask=output_tokens.attention_mask,
            return_dict=True,
            labels=targets,
            use_focal_loss=self.use_focal_loss,  # false
        )

        t5_loss = {'t5_loss': outputs.loss * self.generate_loss_weight}
        loss.update(t5_loss)

        return loss

    @torch.no_grad()
    def text_decoder(
            self,
            text_decoder_inputs,
            use_nucleus_sampling=False,
            num_beams=5,###beam搜索top
            max_length=10,####10就可以
            min_length=1,
            top_p=0.9,#########sample方式 无sample无用
            repetition_penalty=1.0,######惩罚重复——用处不大
            length_penalty=1.0,#######惩罚长句
            num_captions=1,
            temperature=1,##########logit缩放，没啥用
    ):
        object_features = text_decoder_inputs['object_features']

        inputs_t5 = self.t5_proj(object_features)
        atts_t5 = text_decoder_inputs['atts_t5']

        encoder_atts = atts_t5
        inputs_embeds = inputs_t5

        outputs = self.t5_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=encoder_atts,###全1
            do_sample=use_nucleus_sampling,####false
            top_p=top_p,
            temperature=temperature,
            num_beams=num_beams,
            max_new_tokens=max_length,#######最大长度
            min_length=min_length,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_beams,  # num_captions, 3*300
            output_scores=True,
            return_dict_in_generate=True,
        )
        output_text = self.t5_tokenizer.batch_decode(
            outputs.sequences, skip_special_tokens=True
        )  ####900,4___________900
        ##outputs.sequences 解码结果
        ###num_beams 为输入序列的topk解码

        if num_beams>1:##########
            output_sequences_scores = outputs.sequences_scores.sigmoid()
        else:
            scores = torch.stack(list(outputs.scores),dim=0) #[30, 900, 32128]
            log_probs = F.log_softmax(scores, dim=-1)
            top_logprobs, predicted_classes = log_probs.topk(1)
            top_logprobs = top_logprobs.transpose(1,0)
            indexes = outputs.sequences > 0
            sum_top_logprobs = []
            for top_logprob, index in zip(top_logprobs, indexes):
                sum_top_logprobs.append(torch.sum(top_logprob[index[1:]], dim=0))
            output_sequences_scores = torch.tensor(sum_top_logprobs).to(log_probs.device) #[900]

        output_dict = {
            'pred_object_descriptions': output_text,
            'logprobs': output_sequences_scores,
        }

        return output_dict