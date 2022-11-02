from omegaconf import OmegaConf
from vissl.utils.hydra_config import AttrDict

from vissl.utils.hydra_config import compose_hydra_configuration, convert_to_attrdict
from vissl.models import build_model
from classy_vision.generic.util import load_checkpoint
from vissl.utils.checkpoint import init_model_from_consolidated_weights
from torchinfo import summary

def setModel():
    cfg = [
    "config=benchmark/linear_image_classification/imagenet1k/eval_resnet_8gpu_transfer_in1k_linear.yaml",
    "config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/scratch/ds5725/ssl_vpr/pretrain/converted_vissl_rn50_jigsaw_in22k_goyal19.torch",
    'config.MODEL.FEATURE_EVAL_SETTINGS.EVAL_MODE_ON=True', # Turn on model evaluation mode.
    'config.MODEL.FEATURE_EVAL_SETTINGS.FREEZE_TRUNK_ONLY=True', # Freeze trunk. 
    'config.MODEL.FEATURE_EVAL_SETTINGS.EXTRACT_TRUNK_FEATURES_ONLY=True', # Extract the trunk features, as opposed to the HEAD.
    'config.MODEL.FEATURE_EVAL_SETTINGS.SHOULD_FLATTEN_FEATS=False', # Do not flatten features.
    'config.MODEL.FEATURE_EVAL_SETTINGS.LINEAR_EVAL_FEAT_POOL_OPS_MAP=[["res5", ["Identity", []]]]',
    ]

    # Compose the hydra configuration.
    cfg = compose_hydra_configuration(cfg)
    # Convert to AttrDict. This method will also infer certain config options
    # and validate the config is valid.
    _, cfg = convert_to_attrdict(cfg)

    model = build_model(cfg.MODEL, cfg.OPTIMIZER)
    weights = load_checkpoint(checkpoint_path=cfg.MODEL.WEIGHTS_INIT.PARAMS_FILE)


    # Initializei the model with the simclr model weights.
    init_model_from_consolidated_weights(
        config=cfg,
        model=model,
        state_dict=weights,
        state_dict_key_name="model_state_dict",
        skip_layers=[],  # Use this if you do not want to load all layers
    )
    return model

# model=setModel()
# input_size=(1,3,480,640)
# print(summary(model,input_size))
