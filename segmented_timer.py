import time

class SegmentedTimer:
    def __init__(self, label="Total"):
        self.label = label
        self.segments = []
        self.start = time.time()
        self.last = self.start

    def __enter__(self):
        self.start = time.time()
        self.last = self.start
        return self

    def mark(self, segment_label):
        """Record a time mark with a custom label."""
        now = time.time()
        elapsed = now - self.last
        self.segments.append((segment_label, elapsed))
        self.last = now

    def skip(self):
        """Timer updates its internal state without recording a segment."""
        now = time.time()
        self.last = now

    def __exit__(self, exc_type, exc_value, traceback):
        total_time = time.time() - self.start
        self.segments.append((self.label, total_time))

        max_time_len = max(len(f"{seg_time:.4f}") for _, seg_time in self.segments)
        max_label_len = max(len(label) for label, _ in self.segments)
        lines = [
            f"{seg_label:{max_label_len + 1}s}: {seg_time:>{max_time_len}.4f} seconds"
            for seg_label, seg_time in self.segments
        ]
        total_width = max(len(line) for line in lines)

        header_text = " Timing Summary "
        header_line = header_text.center(total_width, '-')

        print(header_line)
        for line in lines:
            print(line)
        print("-" * total_width)
