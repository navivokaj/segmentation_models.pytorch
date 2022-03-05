from ._base import EncoderMixin
from timm.models.convnext import ConvNeXt
import torch.nn as nn

class ConvNeXtEncoder(ConvNeXt, EncoderMixin):
    def __init__(self, out_channels, depth=5, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 3

        del self.head
        del self.norm_pre

    def get_stages(self):
        return [
            nn.Identity(),
            self.stem,
            self.stages[0],
            self.stages[1],
            self.stages[2],
            self.stages[3],
        ]
    
    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)
        
        return features
    
    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("head.fc.bias", None)
        state_dict.pop("head.fc.weight", None)
        state_dict.pop("head.norm.bias", None)
        state_dict.pop("head.norm.weight", None)
        super().load_state_dict(state_dict, **kwargs)

convnext_weights = {
    "convnext_tiny": {
        "imagenet": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth"
    },
    "convnext_small": {
        "imagenet": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth"
    },
}

pretrained_settings = {}
for model_name, sources in convnext_weights.items():
    pretrained_settings[model_name] = {}
    for source_name, source_url in sources.items():
        pretrained_settings[model_name][source_name] = {
            "url": source_url,
            "input_range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "input_space": "RGB",
        }

timm_convnext_encoders = {
    "timm-convnext_tiny": {
        "encoder": ConvNeXtEncoder,
        "pretrained_settings": pretrained_settings["convnext_tiny"],
        "params": {
            "out_channels": (3, 96, 96, 192, 384, 768), 
            "depths": (3, 3, 9, 3),
            "dims": (96, 192, 384, 768),  
        },
    },
    "timm-convnext_small": {
        "encoder": ConvNeXtEncoder,
        "pretrained_settings": pretrained_settings["convnext_small"],
        "params": {
            "out_channels": (3, 96, 96, 192, 384, 768), 
            "depths": (3, 3, 27, 3),
            "dims": (96, 192, 384, 768),  
        },
    },
}
