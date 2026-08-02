import threading
import argparse

from omegaconf import OmegaConf, DictConfig

import time

from perception import get_perception
from communication.lcm.subscriber.sub_manager.perception import get_percep_sub_manager
from communication.lcm.publisher.pub_manager.perception import get_percep_pub_manager


def run_perception(cfg_perception: DictConfig):
    """Run the perception with the given configuration.
    @param[in] cfg_perception: Configuration for the environment.
    """
    perception = get_perception(cfg_perception)

    sub = get_percep_sub_manager(cfg_perception["sub_manager"])
    sub_que_dict = (
        sub.get_sub_que_dict()
    )  # sub_que_dict should be a pointer to the dict of data buffer
    perception.set_sub_que_dict(sub_que_dict)

    pub = get_percep_pub_manager(cfg_perception["pub_manager"])
    # pub_que_dict should be a pointer to the dict of data buffer
    percep_pub_que_dict = pub.get_pub_que_dict()
    perception.set_percep_pub_que_dict(percep_pub_que_dict)

    # this will spin off a thread for the subscriber
    sub.start()
    # this will spin off a thread for the publisher
    pub.start()

    # we initialize the perception module after the subscriber to
    # enable initailze variables that depends on the subscribed data
    perception.initialize()
    perception.start()

    # this will safely terminate the subscriber thread
    sub.stop()
    # this will safely terminate the publisher thread
    pub.stop()
    perception.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args, unknown = parser.parse_known_args()

    cfg = OmegaConf.load(args.config)
    print(OmegaConf.to_yaml(cfg))

    run_perception(cfg)
