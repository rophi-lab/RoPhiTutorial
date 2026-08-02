import threading
import argparse
from queue import Queue

from omegaconf import OmegaConf, DictConfig

import time

from sensor import get_sensor_manager
from communication.lcm.publisher.pub_manager.sensor import get_sensor_pub_manager
from communication.lcm.subscriber.sub_manager.sensor import get_sensor_sub_manager


def run_sensors(cfg_sensors: DictConfig):
    """Run the sensors with the given configuration.
    This is only needed for real-world experiments.
    @param[in] cfg_sensors: Configuration for the environment.
    """
    sensor_manager = get_sensor_manager(cfg_sensors)
    sensor_manager.initialize_sensors_and_sensor_info()
    # TODO: implement FK subscriber
    sub = get_sensor_sub_manager(cfg_sensors["sub_manager"])
    sub_que_dict = (
        sub.get_sub_que_dict()
    )  # sub_que_dict should be a pointer to the dict of data buffer

    print("sub_que_dict_keys: ", sub_que_dict.keys())
    sensor_manager.set_sub_que_dict(sub_que_dict)
    # sensor_manager.set_fk_sub_queue(Queue())

    pub = get_sensor_pub_manager(cfg_sensors["pub_manager"])
    # pub_que_dict should be a pointer to the dict of data buffer
    sensor_pub_que_dict = pub.get_pub_que_dict()
    print("sensor_pub_que_dict_keys: ", sensor_pub_que_dict.keys())
    sensor_manager.set_pub_que_dict(sensor_pub_que_dict)

    # this will spin off a thread for the subscriber
    sub.start()
    # this will spin off a thread for the publisher
    pub.start()

    sensor_manager.start()

    # this will safely terminate the subscriber thread
    sub.stop()
    # this will safely terminate the publisher thread
    pub.stop()
    sensor_manager.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show a live cv2 window (color + depth) for each realsense camera",
    )
    args, unknown = parser.parse_known_args()
    print("args config: ", args.config)
    cfg = OmegaConf.load(args.config)

    # CLI override: enable visualization on every realsense camera without
    # having to edit the config file.
    if args.visualize and "realsense" in cfg:
        for cam_cfg in cfg["realsense"].values():
            cam_cfg["visualize"] = True

    print(OmegaConf.to_yaml(cfg))

    run_sensors(cfg)
