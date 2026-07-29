import threading
import queue


class CommandLineListener:
    def __init__(self):
        self.input_queue = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._listen, daemon=True)

    def _listen(self):
        while not self._stop_event.is_set():
            try:
                user_input = input()  # blocks, but only inside this thread
                self.input_queue.put(user_input)
            except EOFError:
                break

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join()

    def get_next_input(self):
        try:
            return self.input_queue.get_nowait()
        except queue.Empty:
            return None


# Example usage
if __name__ == "__main__":
    import time

    listener = CommandLineListener()
    listener.start()

    try:
        while True:
            cmd = listener.get_next_input()
            if cmd is not None:
                print(f"Received command: {cmd}")
            else:
                print("No command yet. Doing other work...")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping listener...")
        listener.stop()
