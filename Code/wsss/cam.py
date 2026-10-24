from __future__ import annotations

import torch
import torch.nn.functional as F

from wsss.models import cam_layer


class CAMGenerator:
    """Grad-CAM, Grad-CAM++ and LayerCAM for the supported CNN backbones."""

    def __init__(self, model, model_name: str, method: str = "layercam"):
        if method not in {"gradcam", "gradcam++", "layercam"}:
            raise ValueError("method must be gradcam, gradcam++ or layercam")
        self.model, self.method = model, method
        self.activations = None
        self.handle = cam_layer(model, model_name).register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output):
        self.activations = output

    def close(self):
        self.handle.remove()

    def __call__(self, x: torch.Tensor, class_index: int = 0):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        activation = self.activations
        gradient, = torch.autograd.grad(logits[:, class_index].sum(), activation, retain_graph=False)
        positive_gradient = gradient.clamp(min=0)

        if self.method == "gradcam":
            weights = gradient.mean(dim=(2, 3), keepdim=True)
            cam = (weights * activation).sum(dim=1, keepdim=True).relu()
        elif self.method == "gradcam++":
            grad2, grad3 = positive_gradient.square(), positive_gradient.pow(3)
            activation_sum = activation.sum(dim=(2, 3), keepdim=True)
            alpha = grad2 / (2.0 * grad2 + activation_sum * grad3 + 1e-8)
            weights = (alpha * positive_gradient).sum(dim=(2, 3), keepdim=True)
            cam = (weights * activation).sum(dim=1, keepdim=True).relu()
        else:
            cam = (positive_gradient * activation).sum(dim=1, keepdim=True).relu()

        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        low = cam.amin(dim=(2, 3), keepdim=True)
        high = cam.amax(dim=(2, 3), keepdim=True)
        cam = (cam - low) / (high - low).clamp_min(1e-8)
        return cam[:, 0].detach(), logits.detach()
