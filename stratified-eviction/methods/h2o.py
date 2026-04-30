from kvpress import ExpectedAttentionPress

def get_press(compression_ratio=0.5, **kwargs):
    return ExpectedAttentionPress(compression_ratio=compression_ratio)