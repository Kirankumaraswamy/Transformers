import os
from datetime import datetime

now = datetime.now()
EXPERIMENT_NAME = now.strftime("%Y%m%d%H%M%S.%f")

EXPERIMENTS_DIR = "experiments_save"
LOG_FILE = os.path.join(os.path.dirname(__file__), os.pardir, EXPERIMENTS_DIR, EXPERIMENT_NAME, EXPERIMENT_NAME)
JSON_FILE_DIR = os.path.join(os.path.dirname(__file__), os.pardir, EXPERIMENTS_DIR, EXPERIMENT_NAME)
CHECKPOINTS_PATH = os.path.join(os.path.dirname(__file__), os.pardir, EXPERIMENTS_DIR, EXPERIMENT_NAME, 'models')
DATA_DIR_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'data')
TENSORBOARD_DIR = os.path.join(os.path.dirname(__file__), os.pardir, EXPERIMENTS_DIR, "runs", EXPERIMENT_NAME)

LOCAL = True
os.makedirs(CHECKPOINTS_PATH, exist_ok=True)

os.makedirs(DATA_DIR_PATH, exist_ok=True)

BOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'
PAD_TOKEN = "<pad>"

RESTART_EXPERIMENT = False

# DEBUG OPTIONS
DEBUG_CODE = False
DEBUG_CODE_TRAIN_EPOCH_LENGTH = 200
