import threading
import argparse

from omegaconf import OmegaConf, DictConfig

from robot_platform.hardware import get_hardware_platform

from communication.lcm.subscriber.sub_manager.hardware import get_hardware_sub_manager
from communication.lcm.publisher.pub_manager.hardware import get_hardware_pub_manager

import time

from utils.mode import VisMode


def run_hardware(cfg_platform: DictConfig):
    """Run the environment with the given configuration.
    @param[in] cfg_env: Configuration for the environment.
    """

    hardware_platform = get_hardware_platform(cfg_platform)

    sub = get_hardware_sub_manager(cfg_platform["sub_manager"])
    sub_que_dict = (
        sub.get_sub_que_dict()
    )  # sub_que_dict should be a pointer to the dict of data buffer
    hardware_platform.set_sub_que_dict(sub_que_dict)

    pub = get_hardware_pub_manager(cfg_platform["pub_manager"])

    # pub_que_dict should be a pointer to the dict of data buffer
    pub_que_dict = pub.get_pub_que_dict()
    hardware_platform.set_pub_que_dict(pub_que_dict)

    # this will spin off a thread for the subscriber
    sub.start()
    # this will spin off a thread for the publisher
    pub.start()

    # We run the environment in the original thread
    hardware_platform.initialize()
    hardware_platform.start()

    # this will safely terminate the subscriber thread
    pub.stop()
    # this will safely terminate the publisher thread
    sub.stop()
    # hardware_platform.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args, unknown = parser.parse_known_args()

    cfg = OmegaConf.load(args.config)
    print(OmegaConf.to_yaml(cfg))

    run_hardware(cfg)
