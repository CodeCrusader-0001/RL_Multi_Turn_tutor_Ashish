import os, random
import numpy as np

BASE        = "/kaggle/working/rl_llm_tutor"
DATA_DIR    = f"{BASE}/data"
MODELS_DIR  = f"{BASE}/models"
OUTPUT_DIR  = f"{BASE}/output"
SRC_DIR     = f"{BASE}/src"
for _d in [DATA_DIR, MODELS_DIR, OUTPUT_DIR, SRC_DIR]:
    os.makedirs(_d, exist_ok=True)

MISTRAL_MODEL      = "mistral-small-latest"
MAX_TURNS          = 8
N_DIALOGUES        = 2000
N_AUG_TARGET       = 3000
N_AUG_SAMPLES      = N_AUG_TARGET - N_DIALOGUES
N_AUG2_SAMPLES     = 500
N_EVAL             = 15
DISCOUNT           = 0.9
STATE_DIM          = 25
N_ACTIONS          = 4
ACTION_NAMES       = ["Instruct", "Encourage", "Refocus", "Question"]
SEED               = 42
N_WORKERS          = 4
CKPT_EVERY         = 200

STEPS_PER_EPOCH    = 10_000
N_TRAIN_STEPS      = 15 * STEPS_PER_EPOCH
SUPER_N_STEPS      = 15 * STEPS_PER_EPOCH

BATCH_SIZE         = 512
CQL_ALPHA          = 4.0
CQL_LR             = 5e-5
BC_LR              = 1e-3
DQN_LR             = 5e-5
SUPER_CQL_ALPHA    = 8.0
SUPER_CQL_LR       = 3e-5
SUPER_SEEDS        = [42, 7, 13]
REWARD_SHAPE_BONUS = 0.5

random.seed(SEED)
np.random.seed(SEED)
