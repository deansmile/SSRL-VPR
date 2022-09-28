from models_mae import MaskedAutoencoderViT
import torch
import models_mae

class maeTL(MaskedAutoencoderViT):
    def __init__(self):
        super().__init__()
        arch='mae_vit_large_patch16'
        chkpt_dir = '/scratch/ds5725/mae/mae_visualize_vit_large_ganloss.pth'
        backbone = getattr(models_mae, arch)()
        checkpoint = torch.load(chkpt_dir, map_location='cpu')
        msg = backbone.load_state_dict(checkpoint['model'], strict=False)
        print(msg)
#         layers = list(backbone.children())[:3]
#         self.feature_extractor = nn.Sequential(*layers)

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = super().forward_encoder(imgs, mask_ratio)
        return latent