import torch.nn as nn
import torch
from typing import Optional, Union, List, Tuple
from transformers.models.bart.modeling_bart import shift_tokens_right
from transformers.models.layoutlmv3.modeling_layoutlmv3 import (
    LayoutLMv3PreTrainedModel,
    LayoutLMv3Encoder,
    LayoutLMv3PatchEmbeddings,
    LayoutLMv3Config)
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaLayer
from transformers.modeling_outputs import Seq2SeqLMOutput, Seq2SeqModelOutput, BaseModelOutputWithPastAndCrossAttentions
from transformers.utils import logging
from torch.nn import CrossEntropyLoss

logger = logging.get_logger(__name__)


class LayoutLMv3TextEmbeddingsWithOptionalLayout(nn.Module):
    """
    LayoutLMv3 text embeddings. Same as `RobertaEmbeddings` but with added spatial (layout) embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

        self.x_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size)
        self.y_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size)
        self.h_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.shape_size)
        self.w_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.shape_size)

    def calculate_spatial_position_embeddings(self, bbox):
        try:
            left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
            upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
            right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
            lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        except IndexError as e:
            raise IndexError("The `bbox` coordinate values should be within 0-1000 range.") from e

        h_position_embeddings = self.h_position_embeddings(torch.clip(bbox[:, :, 3] - bbox[:, :, 1], 0, 1023))
        w_position_embeddings = self.w_position_embeddings(torch.clip(bbox[:, :, 2] - bbox[:, :, 0], 0, 1023))

        # below is the difference between LayoutLMEmbeddingsV2 (torch.cat) and LayoutLMEmbeddingsV1 (add)
        spatial_position_embeddings = torch.cat(
            [
                left_position_embeddings,
                upper_position_embeddings,
                right_position_embeddings,
                lower_position_embeddings,
                h_position_embeddings,
                w_position_embeddings,
            ],
            dim=-1,
        )
        return spatial_position_embeddings

    def create_position_ids_from_input_ids(self, input_ids, padding_idx):
        """
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding
        symbols are ignored. This is modified from fairseq's `utils.make_positions`.
        """
        # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
        mask = input_ids.ne(padding_idx).int()
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask)) * mask
        return incremental_indices.long() + padding_idx

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)

    def forward(
        self,
        input_ids=None,
        bbox=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        use_spatial_embeddings=True,
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = self.create_position_ids_from_input_ids(input_ids, self.padding_idx).to(
                    input_ids.device
                )
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings
        if use_spatial_embeddings:
            spatial_position_embeddings = self.calculate_spatial_position_embeddings(bbox)
            embeddings = embeddings + spatial_position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

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
        self.layer = nn.ModuleList([RobertaLayer(config) for _ in range(config.decoder_layers)])
        self.embeddings = embeddings

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
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            use_spatial_embeddings=False,
        )
        # tgt_length = input_shape[1]
        ### Using embedding_output instead of input_embeds (which can be None), to fix None issue since the code only use its dtype
        attention_mask_expanded = self._prepare_decoder_attention_mask(attention_mask, input_shape, embedding_output,
                                                                       past_key_values_length)

        cross_attention_mask_expanded = _expand_mask(encoder_attention_mask, embedding_output.dtype,
                                                     tgt_len=input_shape[1])
        decoder_head_mask_expanded = self.get_head_mask(head_mask, self.config.decoder_layers)

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

        if config.text_embed:
            self.embeddings = LayoutLMv3TextEmbeddingsWithOptionalLayout(config)

        if config.visual_embed:
            # use the default pre-training parameters for fine-tuning (e.g., input_size)
            # when the input_size is larger in fine-tuning, we will interpolate the position embeddings in forward
            self.patch_embed = LayoutLMv3PatchEmbeddings(config)

            size = int(config.input_size / config.patch_size)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
            self.pos_embed = nn.Parameter(torch.zeros(1, size * size + 1, config.hidden_size))
            self.pos_drop = nn.Dropout(p=0.0)

            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)

            if self.config.has_relative_attention_bias or self.config.has_spatial_attention_bias:
                self.init_visual_bbox(image_size=(size, size))

            self.norm = nn.LayerNorm(config.hidden_size, eps=1e-6)

        self.encoder = LayoutLMv3Encoder(config)
        self.decoder = RobertaDecoder(config, self.embeddings)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def init_visual_bbox(self, image_size=(14, 14), max_len=1000):
        """
        Create the bounding boxes for the visual (patch) tokens.
        """
        visual_bbox_x = torch.div(
            torch.arange(0, max_len * (image_size[1] + 1), max_len), image_size[1], rounding_mode="trunc"
        )
        visual_bbox_y = torch.div(
            torch.arange(0, max_len * (image_size[0] + 1), max_len), image_size[0], rounding_mode="trunc"
        )
        visual_bbox = torch.stack(
            [
                visual_bbox_x[:-1].repeat(image_size[0], 1),
                visual_bbox_y[:-1].repeat(image_size[1], 1).transpose(0, 1),
                visual_bbox_x[1:].repeat(image_size[0], 1),
                visual_bbox_y[1:].repeat(image_size[1], 1).transpose(0, 1),
            ],
            dim=-1,
        ).view(-1, 4)

        cls_token_box = torch.tensor([[0 + 1, 0 + 1, max_len - 1, max_len - 1]])
        self.visual_bbox = torch.cat([cls_token_box, visual_bbox], dim=0)

    def calculate_visual_bbox(self, device, dtype, batch_size):
        visual_bbox = self.visual_bbox.repeat(batch_size, 1, 1)
        visual_bbox = visual_bbox.to(device).type(dtype)
        return visual_bbox

    def forward_image(self, pixel_values):
        embeddings = self.patch_embed(pixel_values)

        # add [CLS] token
        batch_size, seq_len, _ = embeddings.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add position embeddings
        if self.pos_embed is not None:
            embeddings = embeddings + self.pos_embed

        embeddings = self.pos_drop(embeddings)
        embeddings = self.norm(embeddings)

        return embeddings

    def forward(
        self,
        input_ids=None,
        bbox=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        decoder_input_ids=None,
        head_mask=None,
        inputs_embeds=None,
        pixel_values=None,
        output_attentions=None,
        output_hidden_states=None,
        use_cache: Optional[bool] = None,
        return_dict=None,
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
            device = inputs_embeds.device
        elif pixel_values is not None:
            batch_size = len(pixel_values)
            device = pixel_values.device
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds or pixel_values")

        if input_ids is not None or inputs_embeds is not None:
            if attention_mask is None:
                attention_mask = torch.ones(((batch_size, seq_length)), device=device)
            if token_type_ids is None:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
            if bbox is None:
                bbox = torch.zeros(tuple(list(input_shape) + [4]), dtype=torch.long, device=device)

            embedding_output = self.embeddings(
                input_ids=input_ids,
                bbox=bbox,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
            )

        final_bbox = final_position_ids = None
        patch_height = patch_width = None
        if pixel_values is not None:
            patch_height, patch_width = int(pixel_values.shape[2] / self.config.patch_size), int(
                pixel_values.shape[3] / self.config.patch_size
            )
            visual_embeddings = self.forward_image(pixel_values)
            visual_attention_mask = torch.ones(
                (batch_size, visual_embeddings.shape[1]), dtype=torch.long, device=device
            )
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, visual_attention_mask], dim=1)
            else:
                attention_mask = visual_attention_mask

            if self.config.has_relative_attention_bias or self.config.has_spatial_attention_bias:
                if self.config.has_spatial_attention_bias:
                    visual_bbox = self.calculate_visual_bbox(device, dtype=torch.long, batch_size=batch_size)
                    if bbox is not None:
                        final_bbox = torch.cat([bbox, visual_bbox], dim=1)
                    else:
                        final_bbox = visual_bbox

                visual_position_ids = torch.arange(
                    0, visual_embeddings.shape[1], dtype=torch.long, device=device
                ).repeat(batch_size, 1)
                if input_ids is not None or inputs_embeds is not None:
                    position_ids = torch.arange(0, input_shape[1], device=device).unsqueeze(0)
                    position_ids = position_ids.expand(input_shape)
                    final_position_ids = torch.cat([position_ids, visual_position_ids], dim=1)
                else:
                    final_position_ids = visual_position_ids

            if input_ids is not None or inputs_embeds is not None:
                embedding_output = torch.cat([embedding_output, visual_embeddings], dim=1)
            else:
                embedding_output = visual_embeddings

            embedding_output = self.LayerNorm(embedding_output)
            embedding_output = self.dropout(embedding_output)
        elif self.config.has_relative_attention_bias or self.config.has_spatial_attention_bias:
            if self.config.has_spatial_attention_bias:
                final_bbox = bbox
            if self.config.has_relative_attention_bias:
                position_ids = self.embeddings.position_ids[:, : input_shape[1]]
                position_ids = position_ids.expand_as(input_ids)
                final_position_ids = position_ids

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, None, device, dtype=embedding_output.dtype
        )

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        encoder_outputs = self.encoder(
            embedding_output,
            bbox=final_bbox,
            position_ids=final_position_ids,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            patch_height=patch_height,
            patch_width=patch_width,
        )

        sequence_output = encoder_outputs[0]

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=None,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
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
        self.model = LayoutLMv3TransformerModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.embeddings.word_embeddings.num_embeddings, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

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
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
                max_length=100,
                num_beams=1,
                use_cache=True,
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

        outputs = self.model(
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
