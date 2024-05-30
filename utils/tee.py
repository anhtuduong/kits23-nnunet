class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # Ensure each write is flushed immediately

    def flush(self):
        for f in self.files:
            f.flush()