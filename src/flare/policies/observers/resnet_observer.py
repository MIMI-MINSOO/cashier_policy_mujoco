import einops
import torch
import torchvision
import torch.nn as nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from flare.policies.observers.base_observer import BaseObserver
from flare.models.spatial_softmax import SpatialSoftmax


class ResNetObserver(BaseObserver):
    """
    Extracts 1D or 2D embeddings from a ResNet backbone and concat with states.

    When tokenize=False: Returns concatenated 1D features
    When tokenize=True: Returns 2D token features where ResNet features are spatial tokens
                       and state is added as an additional token.
    """

    def __init__(
        self,
        state_key: str = 'observation.state',
        image_keys: list[str] = ['observation.image'],
        resize_shape: tuple[int, int] = (240, 320),
        crop_shape: tuple[int, int] = (224, 308),
        state_dim: int = 21,
        tokenize: bool = False,
    ):
        super().__init__(
            state_key=state_key,
            image_keys=image_keys,
            resize_shape=resize_shape,
            crop_shape=crop_shape,
        )

        self.tokenize = tokenize
        self.state_dim = state_dim

        # ResNet backbone setup
        backbone = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
            norm_layer=FrozenBatchNorm2d,
        )
        self.backbone = IntermediateLayerGetter(backbone, return_layers={"layer4": "feature_map"})

        if self.tokenize:
            self.spatial_pool = nn.AdaptiveAvgPool2d((3, 3))
            self.state_projector = nn.Linear(state_dim, 512)
        else:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def observe(self, batch):
        if self.tokenize:
            return self._observe_tokenized(batch)
        else:
            return self._observe_flattened(batch)

    def _observe_tokenized(self, batch):
        b = batch[self.state_key].shape[0]
        s = batch[self.image_keys[0]].shape[1]
        n = len(self.image_keys)

        images = self.get_images(batch)
        img_features = self.backbone(images)["feature_map"]
        img_features = self.spatial_pool(img_features)

        img_tokens = einops.rearrange(img_features, 'bsn c h w -> bsn (h w) c')

        img_tokens = einops.rearrange(
            img_tokens, '(b s n) hw c -> b (s n hw) c', b=b, s=s, n=n
        )

        states = self.get_states(batch).flatten(start_dim=1)
        state_tokens = self.state_projector(states).unsqueeze(1)

        tokens = torch.cat([state_tokens, img_tokens], dim=1)

        return tokens

    def _observe_flattened(self, batch):
        features = []
        features.append(self.get_states(batch).flatten(start_dim=1))

        b = batch[self.state_key].shape[0]
        s = batch[self.state_key].shape[1]
        n = len(self.image_keys)

        images = self.get_images(batch)
        img_features = self.pool(self.backbone(images)["feature_map"])
        features.append(
            einops.rearrange(
                img_features, '(b s n) c h w -> b (s n c h w)', b=b, s=s, n=n
            )
        )

        features = torch.cat(features, dim=1)
        return features


# ============ Diffusion Policy RGB Encoder ============

def _replace_bn_with_gn(module):
    """Replace all BatchNorm2d with GroupNorm (num_groups = num_features // 16)."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(
                num_groups=child.num_features // 16,
                num_channels=child.num_features,
            ))
        else:
            _replace_bn_with_gn(child)
    return module


def _get_output_shape(module, input_shape):
    """Dry run to get output shape."""
    with torch.no_grad():
        dummy = torch.zeros(input_shape)
        output = module(dummy)
    return output.shape


class DiffusionRgbEncoder(nn.Module):
    """ResNet18 + GroupNorm + SpatialSoftmax encoder for Diffusion Policy.

    Output: (B, spatial_softmax_num_keypoints * 2) per image.
    """

    def __init__(
        self,
        resize_shape: tuple[int, int] = (240, 320),
        crop_shape: tuple[int, int] = (216, 288),
        crop_is_random: bool = True,
        spatial_softmax_num_keypoints: int = 32,
    ):
        super().__init__()

        # Preprocessing
        self.resize = torchvision.transforms.Resize(resize_shape) if resize_shape else None
        if crop_shape is not None:
            self.center_crop = torchvision.transforms.CenterCrop(crop_shape)
            self.random_crop = torchvision.transforms.RandomCrop(crop_shape) if crop_is_random else self.center_crop
        else:
            self.center_crop = None
            self.random_crop = None

        # ResNet18 backbone without final FC and avgpool, BatchNorm → GroupNorm
        backbone = torchvision.models.resnet18(weights=None)
        self.backbone = nn.Sequential(*(list(backbone.children())[:-2]))
        _replace_bn_with_gn(self.backbone)

        # Get feature map shape via dry run
        dummy_h, dummy_w = crop_shape if crop_shape else resize_shape
        feature_map_shape = _get_output_shape(self.backbone, (1, 3, dummy_h, dummy_w))[1:]

        # SpatialSoftmax pooling
        self.pool = SpatialSoftmax(feature_map_shape, num_kp=spatial_softmax_num_keypoints)
        self.feature_dim = spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(self.feature_dim, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor with pixel values in [0, 1].
        Returns:
            (B, feature_dim) image feature vector.
        """
        if self.resize is not None:
            x = self.resize(x)
        if self.center_crop is not None:
            if self.training:
                x = self.random_crop(x)
            else:
                x = self.center_crop(x)

        x = self.backbone(x)
        x = torch.flatten(self.pool(x), start_dim=1)
        x = self.relu(self.out(x))
        return x
