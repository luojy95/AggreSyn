import os

ROOT_DIR = os.getcwd() # get absolute path of root directory

ENV_LIGHTING_PATH = f'{ROOT_DIR}/data/envlight/*'
BALLAST_MATERIAL_PATH = f'{ROOT_DIR}/data/texture/ballast/*'
BALLAST_MESH_PATH = f'{ROOT_DIR}/data/mesh/*'
FINE_MATERIAL_PATH = f'{ROOT_DIR}/data/texture/fine/*'

OUTPUT_DIR = f'{ROOT_DIR}/outputs'
TMP_DIR = f'{ROOT_DIR}/tmp'

def hash_string(s):
    return abs(hash(s)) % (10 ** 8)