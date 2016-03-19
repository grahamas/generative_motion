

class CircularArray(object):
    def __init__(self, init_val):
        self.array = init_val
        self.start = 0
    def __len__(self):
        return len(self.array)
    def append(self, arr):
        arr_len = len(arr)
        assert arr_len <= len(self)
        next_start = (self.start + arr_len) % len(self)
        if next_start > self.start:
            self.array[self.start:next_start] = arr
        else:
            arr_cut = arr_len - (len(self) - self.start)
            self.array[self.start:] = arr[:arr_cut]
            if next_start > 0:
                self.array[:next_start] = arr[arr_cut:]
        self.start = next_start
    def get(self):
        return self.array[self.start:] + self.array[:self.start]

