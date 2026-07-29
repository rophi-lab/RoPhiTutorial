import time

import argparse

from omegaconf import OmegaConf, DictConfig

from utils.terminal.argparse_utils import parse_unknown_args, parse_nested_args
from visualizer import get_vis_manager


def run_visualizer(cfg_visualizer: DictConfig):
    """Run the visualizer with the given configuration.
    @param[in] cfg_visualizer: Configuration for the visualizer.
    """

    vis_manager = get_vis_manager(cfg_visualizer)

    vis_manager.initialize()
    vis_manager.start()
    vis_manager.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args, unknown = parser.parse_known_args()
    d_cmd_cfg = parse_unknown_args(unknown)
    d_cmd_cfg = parse_nested_args(d_cmd_cfg)

    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(cfg, d_cmd_cfg)
    print(OmegaConf.to_yaml(cfg))

    run_visualizer(cfg)
