from hydra.core.global_hydra import GlobalHydra
from hydra import initialize

# Clear any previous Hydra instance
GlobalHydra.instance().clear()

# Initialize Hydra safely
initialize(config_path="configs", version_base="1.2")

from GroundingDINO.groundingdino.util.inference import Model
from path_config import GROUNDING_DINO_CHECKPOINT_PATH, GROUNDING_DINO_CONFIG_PATH, SAM_CHECKPOINT_PATH, SAM2_CHECKPOINT_PATH_CPU, SAM2_CONFIG_PATH_CPU
from segment_anything import sam_model_registry, SamPredictor
from samv2.sam2.build_sam import build_sam2_video_predictor
from samv2.sam2.utils.misc import variant_to_config_mapping

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE="cpu"

#Load Gounding DINO model
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

#Load SAM and SAM2 models
SAM_ENCODER_VERSION = "vit_b"

sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
sam_predictor = SamPredictor(sam)
print('Success of sam_predictor')

sam2_model = build_sam2_video_predictor(variant_to_config_mapping["tiny"], SAM2_CHECKPOINT_PATH_CPU)
print('Success of sam2_model')

