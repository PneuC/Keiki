class BufferedLazyList:
    def __init__(self, capacity=None):
        self.__main = []
        self.__buffer = []
        self.__updated = True
        self.__capacity = capacity

    def add(self, *items):
        """
        Add the items into buffer. Note that the update() should be

        :param items: The items should be add to the list.
        :return: None
        """
        self.__buffer += items
        self.__updated = False

    def update(self, condition=lambda x:False):
        """
        Merge the buffer into main list where the objects satisfy the condition will be overwrite.

        :param condition: The condition to remove a objects
        :param invoke: if true, .update() will be invoked for each element in the list.
        :return: None
        """
        buffer_index = 0
        for i in range(len(self.__main)):
            if condition(self.__main[i]):
                if buffer_index < len(self.__buffer):
                    self.__main[i] = self.__buffer[buffer_index]
                    buffer_index += 1
                continue
            self.__main[i].update()
        if self.__capacity:
            end = min(len(self.__buffer), buffer_index + self.__capacity - len(self.__main))
            self.__main += self.__buffer[buffer_index: end]
            for remain in self.__buffer[end:]:
                if hasattr(remain, 'dead'):
                    remain.dead = True
        else:
            self.__main += self.__buffer[buffer_index:]
        self.__buffer.clear()
        self.__updated = True

    def clear(self):
        self.__main.clear()
        self.__buffer.clear()
        self.__updated = True

    def __iter__(self):
        if not self.__updated:
            raise RuntimeWarning('The buffered lazy list has been accessed before update!')
        for item in self.__main:
            yield item

    def __len__(self):
        return len(self.__main)
