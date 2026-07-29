from collections import deque


class MovingAverage:
    def __init__(self, window_size):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
        self.sum = 0.0

    def update(self, value):
        if len(self.values) == self.window_size:
            self.sum -= self.values[0]  # oldest value will be popped
        self.values.append(value)
        self.sum += value
        return self.average()

    def average(self):
        return self.sum / len(self.values) if self.values else 0.0
