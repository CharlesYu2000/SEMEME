import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
import logging

logger = logging.getLogger("sememe")

class OnOff_MistralDecoderLayer(nn.Module):
    def __init__(self, original_decoder_layer):
        super().__init__()
        self.hidden_size = original_decoder_layer.hidden_size

        self.self_attn = original_decoder_layer.self_attn
        self.mlp = original_decoder_layer.mlp
        self.input_layernorm = original_decoder_layer.input_layernorm
        self.post_attention_layernorm = original_decoder_layer.post_attention_layernorm
        # transformers>=5.0 reads layer.attention_type when building the attention mask
        self.attention_type = getattr(original_decoder_layer, "attention_type", None)

        self.pass_layer = False

    def turn_off(self):
        self.pass_layer = True

    def turn_on(self):
        self.pass_layer = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` or `(batch_size, 1, query_sequence_length, key_sequence_length)`
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding.
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        # skip this decoder layer
        if self.pass_layer:
            return hidden_states

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        self_attn_output = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )

        if isinstance(self_attn_output, tuple):
            if len(self_attn_output) == 2:
                hidden_states, self_attn_weights = self_attn_output
                present_key_value = None
            elif len(self_attn_output) == 3:
                hidden_states, self_attn_weights, present_key_value = self_attn_output
            else:
                raise ValueError(f"Unexpected number of outputs from self.self_attn: {len(self_attn_output)}")
        else:
            hidden_states = self_attn_output
            self_attn_weights = None
            present_key_value = None

        if residual.device != hidden_states.device:
            residual = residual.to(hidden_states.device)

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        if residual.device != hidden_states.device:
            residual = residual.to(hidden_states.device)

        hidden_states = residual + hidden_states
        return hidden_states

def block_replace(model):
    num_layers = len(model.model.layers)
    for i in range(num_layers):
        model.model.layers[i] = OnOff_MistralDecoderLayer(model.model.layers[i])
    print("Mistral block replacement complete.")
    return model

def turn_off(model, block_idx):
    model.model.layers[block_idx].turn_off()

def turn_on(model, block_idx):
    model.model.layers[block_idx].turn_on()

def scan(model, num_blocks):
    alive_list = []
    skip_list = []
    for i in range(num_blocks):
        if model.model.layers[i].pass_layer == True:
            skip_list.append(i)
        elif model.model.layers[i].pass_layer == False:
            alive_list.append(i)
    print(
        f"pass layer: {skip_list}\n"
        f"do layer: {alive_list}"
    )
