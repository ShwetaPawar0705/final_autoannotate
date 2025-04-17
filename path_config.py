import os

HOME = os.getcwd()
# print("HOME:", HOME)

GROUNDING_DINO_CONFIG_PATH = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
# print(GROUNDING_DINO_CONFIG_PATH, "; exist:", os.path.isfile(GROUNDING_DINO_CONFIG_PATH))

GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(HOME, "weights", "groundingdino_swint_ogc.pth")
# print(GROUNDING_DINO_CHECKPOINT_PATH, "; exist:", os.path.isfile(GROUNDING_DINO_CHECKPOINT_PATH))

#huge model
# SAM_CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
# print(SAM_CHECKPOINT_PATH, "; exist:", os.path.isfile(SAM_CHECKPOINT_PATH))

#base model
SAM_CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_b_01ec64.pth")
# print(SAM_CHECKPOINT_PATH, "; exist:", os.path.isfile(SAM_CHECKPOINT_PATH))

SAM2_CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam2_hiera_large.pt")
# print(SAM2_CHECKPOINT_PATH, "; exist:", os.path.isfile(SAM2_CHECKPOINT_PATH))

SAM2_CHECKPOINT_PATH_CPU = os.path.join(HOME, "weights", "sam2_hiera_tiny.pt")
# print(SAM2_CHECKPOINT_PATH_CPU, "; exist:", os.path.isfile(SAM2_CHECKPOINT_PATH_CPU))

SAM2_CONFIG_PATH = os.path.join(HOME, "segment_anything_2", "sam2/sam2_hiera_l.yaml")
# print(SAM2_CONFIG_PATH, "; exist:", os.path.isfile(SAM2_CONFIG_PATH))

SAM2_CONFIG_PATH_CPU = os.path.join(HOME, "segment_anything_2", "sam2/sam2_hiera_t.yaml")
# print(SAM2_CONFIG_PATH, "; exist:", os.path.isfile(SAM2_CONFIG_PATH))
