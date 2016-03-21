

class CircularArray(object):
    def __init__(self, init_val):
        self.array = init_val
        self.start = 0
    def __len__(self):
        return len(self.array)
    def append(self, arr):
        arr_len = len(arr)
        if arr_len > len(self):
            raise NotImplementedError
        next_start = (self.start + arr_len) % len(self)
        if next_start > self.start:
            self.array[self.start:next_start] = arr
        else:
            arr_cut = len(self) - self.start
            self.array[self.start:] = arr[:arr_cut]
            self.array[:next_start] = arr[arr_cut:]
        self.start = next_start
        return self
    def get(self):
        return self.array[self.start:] + self.array[:self.start]

