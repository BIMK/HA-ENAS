import argparse
from yacs.config import CfgNode
import sys

sys.path.append('../')

cfg = CfgNode(new_allowed=True)


def load_configs():
    parser = argparse.ArgumentParser(description="Config file options.")
    parser.add_argument("--config_path", type=str, default='config/config.yaml')

    args = parser.parse_args()
    path = args.config_path
    print("config_path:", path)

    cfg.merge_from_file(path)



