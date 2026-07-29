import threading
import argparse

from omegaconf import OmegaConf, DictConfig

import time

from controller import get_controller
from communication.lcm.subscriber.sub_manager.controller import (
    get_controller_sub_manager,
)
from communication.lcm.publisher.pub_manager.controller import (
    get_controller_pub_manager,
)

from utils.terminal.argparse_utils import parse_unknown_args, parse_nested_args


def run_controller(cfg_controller: DictConfig):
    """Run the controller with the given configuration.
    @param[in] cfg_controller: Configuration for the environment.
    """
    controller = get_controller(cfg_controller)

    sub = get_controller_sub_manager(cfg_controller["sub_manager"])
    sub_que_dict = (
        sub.get_sub_que_dict()
    )  # sub_que_dict should be a pointer to the dict of data buffer
    controller.set_sub_que_dict(sub_que_dict)

    pub = get_controller_pub_manager(cfg_controller["pub_manager"])
    # pub_que_dict should be a pointer to the dict of data buffer
    ctrl_pub_que = pub.get_ctrl_pub_que()
    pub_que_dict = pub.get_pub_que_dict()
    controller.set_ctrl_pub_que(ctrl_pub_que)
    controller.set_pub_que_dict(pub_que_dict)

    # this will spin off a thread for the subscriber
    sub.start()
    # this will spin off a thread for the publisher
    pub.start()

    controller.initialize()
    controller.start()

    # this will safely terminate the thread
    sub.stop()
    pub.stop()
    controller.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)

    args, unknown = parser.parse_known_args()
    d_cmd_cfg = parse_unknown_args(unknown)
    d_cmd_cfg = parse_nested_args(d_cmd_cfg)

    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(cfg, d_cmd_cfg)
    print(OmegaConf.to_yaml(cfg))

    run_controller(cfg)
