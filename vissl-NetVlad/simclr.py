from omegaconf import OmegaConf
from vissl.utils.hydra_config import AttrDict

from vissl.utils.hydra_config import compose_hydra_configuration, convert_to_attrdict
from vissl.models import build_model
from vissl.utils.checkpoint import init_model_from_consolidated_weights
from classy_vision.generic.util import load_checkpoint
from torchinfo import summary
import torch.nn as nn

def setModel():
  cfg = [
    'config=pretrain/simclr/simclr_8node_resnet.yaml',
    'config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/home/ds5725/vissl/pretrain/resnet_simclr.torch', # Specify path for the model weights.
    'config.MODEL.FEATURE_EVAL_SETTINGS.EVAL_MODE_ON=True', # Turn on model evaluation mode.
    'config.MODEL.FEATURE_EVAL_SETTINGS.FREEZE_TRUNK_ONLY=True', # Freeze trunk. 
    'config.MODEL.FEATURE_EVAL_SETTINGS.EXTRACT_TRUNK_FEATURES_ONLY=True', # Extract the trunk features, as opposed to the HEAD.
    'config.MODEL.FEATURE_EVAL_SETTINGS.SHOULD_FLATTEN_FEATS=False', # Do not flatten features.
    'config.MODEL.FEATURE_EVAL_SETTINGS.LINEAR_EVAL_FEAT_POOL_OPS_MAP=[["res5", ["Identity", []]]]' # Extract only the res5avg features.
  ]

  # NOTE: After this everything is the same as the above example of extracting 
  # the TRUNK features.

  # Compose the hydra configuration.
  cfg = compose_hydra_configuration(cfg)

  # Convert to AttrDict. This method will also infer certain config options
  # and validate the config is valid.
  _, cfg = convert_to_attrdict(cfg)

  model = build_model(cfg.MODEL, cfg.OPTIMIZER)

  # Load the checkpoint weights.
  weights = load_checkpoint(checkpoint_path=cfg.MODEL.WEIGHTS_INIT.PARAMS_FILE)

  # # Initializei the model with the simclr model weights.
  init_model_from_consolidated_weights(
      config=cfg,
      model=model,
      state_dict=weights,
      state_dict_key_name="classy_state_dict",
      skip_layers=[],  # Use this if you do not want to load all layers
  )
  return model

  # input_size = (1, 3, 224, 224)
  # print(summary(model,input_size))
# f=open("scratch.txt","w")
# for name, param in model.state_dict().items():
#   f.write(name+"\n")

# layers = list(model.children())
# encoder = nn.Sequential(*layers)

model=setModel()

input_size = (1, 3, 480, 640)
print(summary(model, input_size))