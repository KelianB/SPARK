import threading
import queue

# MultiFLARE imports
from utils.visualization import save_img


class AsyncImageSaver:
    def __init__(self):
        self.image_queue = queue.Queue(maxsize=1e6)
        self.running = False
        self.worker_thread = None

    def start(self):
        """Start a background thread."""
        self.running = True
        self.worker_thread = threading.Thread(target=self._process_images, daemon=True)
        self.worker_thread.start()

    def stop(self):
        """Stop the background thread after processing all images in the queue."""
        if not self.running:
            raise RuntimeError("Image saver is not running.")
        self.running = False
        self.worker_thread.join() # wait for the worker thread to finish

    def queue(self, image, filename: str):
        """Add an image to the queue to be saved to disk."""
        if not self.running:
            raise RuntimeError("Image saver is not running.")
        self.image_queue.put((image, filename))

    def _process_images(self):
        """Process images in the queue and write them to disk."""
        while self.running or not self.image_queue.empty():
            try:
                # Get an image and filename from the queue, save it and mark the task as done
                image, filename = self.image_queue.get(timeout=1)
                save_img(filename, image)
                self.image_queue.task_done()
            except queue.Empty:
                continue

    def __enter__(self):
        """Start the thread when entering the context."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the thread when exiting the context."""
        self.stop()
