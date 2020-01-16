class F:
    pass


class Leaf(F):
    def __init__(self, val):
        self.val = val

    def __str__(self):
        return self.val

    def to_str(self, namer):
        return namer(self.val)

    def __len__(self):
        return 1


class Node(F):
    comp = None

    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __str__(self):
        return f'{self.left} {self.comp} {self.right}'

    def to_str(self, namer):
        left_name = self.left.to_str(namer)
        right_name = self.right.to_str(namer)
        return f'{left_name} {self.comp} {right_name}'

    def __len__(self):
        return len(self.left) + len(self.right)


class Or(Node):
    comp = 'OR'


class And(Node):
    comp = 'AND'
