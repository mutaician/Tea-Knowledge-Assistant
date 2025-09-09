import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(ROOT_DIR, "data")

OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")

VECTOR_DB_DIR = os.path.join(OUTPUTS_DIR, "vector_db")

CODE_DIR = os.path.join(ROOT_DIR, "code")

PROMPT_CONFIG_FPATH = os.path.join(CODE_DIR, "config", "prompt_config.yaml")
