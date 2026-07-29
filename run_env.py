"""Launch a configured environment with LCM pub/sub managers."""

import argparse

from omegaconf import OmegaConf, DictConfig

from communication.lcm.publisher.pub_manager.env import get_env_pub_manager
from communication.lcm.subscriber.sub_manager.env import get_env_sub_manager
from env import get_env
from utils.terminal.argparse_utils import parse_nested_args, parse_unknown_args


def run_env(cfg_env: DictConfig) -> None:
    """Create env + LCM managers, then run until quit."""
    env = get_env(cfg_env)

    sub = get_env_sub_manager(cfg_env["sub_manager"])
    env.set_sub_que_dict(sub.get_sub_que_dict())

    pub = get_env_pub_manager(cfg_env["pub_manager"])
    env.set_pub_que_dict(pub.get_pub_que_dict())

    sub.start()
    pub.start()

    env.initialize()
    env.start()

    pub.stop()
    sub.stop()
    env.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args, unknown = parser.parse_known_args()

    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(cfg, parse_nested_args(parse_unknown_args(unknown)))
    print(OmegaConf.to_yaml(cfg))

    run_env(cfg)
