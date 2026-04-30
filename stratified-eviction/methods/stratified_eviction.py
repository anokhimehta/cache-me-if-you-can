#methods/stratified-eviction.py

from kvpress.presses.expected_attention_press import ExpectedAttentionPress
from dataclasses import dataclass
import torch

#we subclass to inherit the score() method which computes the expected attention scores based on the model's actual attention patterns
@dataclass
class ModalityStratifiedPress(ExpectedAttentionPress):
    vision_weight: float = 0.2
    
    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        if self.compression_ratio == 0:
            return keys, values

        #batch size, number of heads, sequence length, head dimension
        bsz, num_kv_heads, k_len, head_dim = keys.shape
        n_kept = int(k_len * (1 - self.compression_ratio))

        #figure out how many vision tokens there are
        image_grid_thw = kwargs.get("image_grid_thw", None)
        if image_grid_thw is not None:
            n_visual = sum(t * h * w // 4 for t, h, w in image_grid_thw)
        else:
            n_visual = 0
        n_visual = min(n_visual, k_len)

        # split the kept budget between visual and text tokens
        vision_budget = min(round(n_kept * self.vision_weight), n_visual)
        text_budget = n_kept - vision_budget

        # score all tokens using inherited ExpectedAttention scoring
        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)

        # split scores by modality
        vis_scores = scores[:, :, :n_visual]
        txt_scores = scores[:, :, n_visual:]

        # top-k independently within each modality
        _, vis_idx = vis_scores.topk(vision_budget, dim=-1)
        _, txt_idx = txt_scores.topk(text_budget, dim=-1)
        txt_idx = txt_idx + n_visual

        # merge and sort to preserve original order
        kept_idx, _ = torch.cat([vis_idx, txt_idx], dim=-1).sort(dim=-1)

        # gather the kept keys and values
        expand_idx = kept_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        keys = keys.gather(2, expand_idx).contiguous()
        values = values.gather(2, expand_idx).contiguous()

        return keys, values
    

def get_press(compression_ratio=0.5, vision_weight=0.2):
    return ModalityStratifiedPress(
        compression_ratio=compression_ratio,
        vision_weight=vision_weight
    )