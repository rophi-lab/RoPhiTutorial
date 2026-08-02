from queue import Queue

from communication.lcm.publisher.BasePubManager import BasePubManager
from communication.lcm.publisher.data_publisher.Point3DPublisher import Point3DPublisher


class Point3DPubManager(BasePubManager):
    """
    Class for managing the publisher for point data.
    """

    def __init__(self, point_channel: str = "point_data", *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pub_que_dict[point_channel] = Queue()

        self.publisher_dict[point_channel] = Point3DPublisher(
            self._lcm_instance, point_channel, self.pub_que_dict[point_channel]
        )
