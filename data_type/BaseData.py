class BaseData:
    """
    Base class for data objects.
    """

    def __init__(self, name: str = "unknown data"):
        self.timestamp = -1.0
        self.name = name

    def __str__(self):
        """
        String representation of the data object.
        """
        return f"{self.__class__.__name__}" + f", name: {self.name}"

    def set_time(self, timestamp: float):
        """
        Set the timestamp for the data object.
        """
        self.timestamp = timestamp

    def set_data(self, data):
        """
        Set the data for the data object.
        """
        raise NotImplementedError("set_data() must be implemented in the subclass")

    def get_time(self) -> float:
        """
        Get the timestamp for the data object.
        """
        return self.timestamp

    def get_data(self):
        """
        Get the data from the data object.
        """
        raise NotImplementedError("get_data() must be implemented in the subclass")
