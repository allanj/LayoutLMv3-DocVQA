import torch.nn as nn
import torch
from typing import Optional, Union, List, Tuple
from transformers.models.bart.modeling_bart import shift_tokens_right
from transformers.models.layoutlmv3.modeling_layoutlmv3 import (
    LayoutLMv3PreTrainedModel,
    LayoutLMv3Encoder,
    LayoutLMv3Layer,
    LayoutLMv3Model,
    LayoutLMv3PatchEmbeddings,
    LayoutLMv3Config)
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaLayer, RobertaConfig
from transformers.modeling_outputs import Seq2SeqLMOutput, Seq2SeqModelOutput, BaseModelOutputWithPastAndCrossAttentions
from transformers.utils import logging
from torch.nn import CrossEntropyLoss

logger = logging.get_logger(__name__)

def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)

def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), float("-inf"))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

class RobertaDecoder(RobertaPreTrainedModel):
    def __init__(self, config, embeddings=None):
        super().__init__(config)
        self.config = config
        config.is_decoder = True
        config.add_cross_attention = True
        self.layer = nn.ModuleList([RobertaLayer(config) for _ in range(config.num_hidden_layers)])
        self.embeddings = embeddings

    def forward_embedding(self,
                input_ids=None,
                token_type_ids=None,
                position_ids=None,
                inputs_embeds=None,
        ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = self.embeddings.create_position_ids_from_input_ids(input_ids, self.embeddings.padding_idx).to(
                    input_ids.device
                )
            else:
                position_ids = self.embeddings.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.embeddings.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.embeddings.word_embeddings(input_ids)
        token_type_embeddings = self.embeddings.token_type_embeddings(token_type_ids)

        embeddings_rep = inputs_embeds + token_type_embeddings
        position_embeddings = self.embeddings.position_embeddings(position_ids)
        embeddings_rep += position_embeddings


        embeddings_rep = self.embeddings.LayerNorm(embeddings_rep)
        embeddings_rep = self.embeddings.dropout(embeddings_rep)
        return embeddings_rep

    def forward(
            self,
            input_ids=None,
            position_ids=None,
            token_type_ids=None,
            attention_mask=None,
            cross_attn_head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            head_mask=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif input_ids is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            # Auto generate decoder input ids when it and inputs_embed are not provided
            # if input_ids is None and inputs_embeds is None:
            #     input_ids = shift_tokens_right(
            #         input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            #     )
            # input_shape = input_ids.size()
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # if past_key_values is not None:
        #     print(len(past_key_values), type(past_key_values[0]), len(past_key_values[0]))
        #     print('--', past_key_values[0][0].size())

        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        embedding_output = self.forward_embedding(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        # tgt_length = input_shape[1]
        ### Using embedding_output instead of input_embeds (which can be None), to fix None issue since the code only use its dtype
        attention_mask_expanded = self._prepare_decoder_attention_mask(attention_mask, input_shape, embedding_output,
                                                                       past_key_values_length)

        cross_attention_mask_expanded = _expand_mask(encoder_attention_mask, embedding_output.dtype,
                                                     tgt_len=input_shape[1])
        decoder_head_mask_expanded = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        decoder_outputs = self.internal_forward(
            hidden_states=embedding_output,
            attention_mask=attention_mask_expanded,
            head_mask=decoder_head_mask_expanded,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=cross_attention_mask_expanded,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return decoder_outputs

    def internal_forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(self.device)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask



class LayoutLMv3TransformerModel(LayoutLMv3PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.encoder = LayoutLMv3Model(config)
        self.embeddings = self.encoder.embeddings
        roberta_config = RobertaConfig.from_pretrained('roberta-base')
        self.decoder = RobertaDecoder(roberta_config, self.encoder.embeddings)
        self.init_weights()

    def get_input_embeddings(self):
        return self.encoder.embeddings.word_embeddings

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def set_input_embeddings(self, value):
        self.encoder.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids=None,
        bbox=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        head_mask=None,
        inputs_embeds=None,
        pixel_values=None,
        output_attentions=None,
        output_hidden_states=None,
        use_cache: Optional[bool] = None,
        return_dict=None,
        **kwargs
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            bbox=bbox,
            pixel_values=pixel_values,
        )
        encoder_hidden_states = encoder_outputs[0]
        batch_size, seq_len, _ = encoder_hidden_states.size()
        visual_attention_mask = torch.ones(
            (batch_size, seq_len - attention_mask.size(1)), dtype=torch.long, device=encoder_hidden_states.device
        )
        updated_attention_mask = torch.cat([attention_mask, visual_attention_mask], dim=1)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=None,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=updated_attention_mask, ## use the extended attention mask
            head_mask=None,
            cross_attn_head_mask=None,
            past_key_values=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs
        # if not return_dict:
        #     return (sequence_output,) + encoder_outputs[1:]

        # return BaseModelOutput(
        #     last_hidden_state=sequence_output,
        #     hidden_states=encoder_outputs.hidden_states,
        #     attentions=encoder_outputs.attentions,
        # )
        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class LayoutLMv3ForConditionalGeneration(LayoutLMv3PreTrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head.weight"]

    def __init__(self, config: LayoutLMv3Config):
        super().__init__(config)
        self.layoutlmv3 = LayoutLMv3TransformerModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.layoutlmv3.embeddings.word_embeddings.num_embeddings)))
        self.lm_head = nn.Linear(config.hidden_size, self.layoutlmv3.embeddings.word_embeddings.num_embeddings, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.layoutlmv3.get_encoder()

    def get_decoder(self):
        return self.layoutlmv3.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_input_embeddings(self):
        return self.layoutlmv3.get_input_embeddings()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        bbox: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        is_train: bool = True,
        **kwargs,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """
        if not is_train:
            return self.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
                max_length=100,
                num_beams=1,
                use_cache=True,
                return_dict=return_dict,
                **kwargs,
            )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.layoutlmv3(
            input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past


if __name__ == '__main__':
    from transformers import RobertaModel, RobertaConfig
    from transformers import LayoutLMv3TokenizerFast, LayoutLMv3Tokenizer, LayoutLMv3FeatureExtractor, \
        LayoutLMv3Processor
    # old = RobertaModel.from_pretrained('roberta-base')
    #
    # new_model = RobertaDecoder(RobertaConfig.from_pretrained('roberta-base'))
    #
    # new_model.layer.load_state_dict(old.encoder.layer.state_dict(), strict=False)

    model = LayoutLMv3ForConditionalGeneration(LayoutLMv3Config.from_pretrained('microsoft/layoutlmv3-base'))
    model.config.decoder_start_token_id = model.config.eos_token_id
    model.config.is_encoder_decoder = True
    # tokenizer = LayoutLMv3TokenizerFast.from_pretrained('microsoft/layoutlmv3-base')
    # res = tokenizer.encoder_plus(['Hello', 'world'], boxes = [[0,0,0,0] for _ in range(2)], return_tensors='pt')
    # print(res)
    # input_ids = torch.tensor(([]))

    input_ids =  torch.tensor([[0, 20920, 232, 2, 1], [0, 20920, 232, 100, 2]])
    attention_mask = torch.tensor([[1,1,1,1,0], [1,1,1,1,1]])
    bbox = torch.tensor([[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],  [0, 0, 0, 0],  [0, 0, 0, 0]],
                         [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],  [0, 0, 0, 0],  [0, 0, 0, 0]]
                         ])
    result = model(input_ids = input_ids, attention_mask = attention_mask, bbox = bbox, is_train=False)

    # from transformers import BartForConditionalGeneration, BartTokenizerFast
    # bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    # bar_tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-base')
    # bart_res = bar_tokenizer.batch_encode_plus(["how are you doing"], return_tensors='pt')
    # bart_result = bart_model.generate(input_ids=bart_res['input_ids'], attention_mask=bart_res['attention_mask'], num_beams=1)