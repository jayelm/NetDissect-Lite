class F:
    pass


class Leaf(F):
    def __init__(self, val):
        self.val = val

    def __str__(self):
        return str(self.val)

    def to_str(self, namer):
        return namer(self.val)

    def __len__(self):
        return 1

    def __hash__(self):
        return hash(str(self))


class Node(F):
    pass


class UnaryNode(Node):
    op = None

    def __init__(self, val):
        self.val = val

    def __str__(self):
        return f'{self.op} {self.val}'

    def to_str(self, namer):
        not_name = self.val.to_str(namer)
        return f'{self.op} {not_name}'

    def __len__(self):
        return 1 + len(self.val)

    def __hash__(self):
        return hash(str(self))


class BinaryNode(Node):
    op = None

    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __str__(self):
        return f'{self.left} {self.op} {self.right}'

    def to_str(self, namer):
        left_name = self.left.to_str(namer)
        right_name = self.right.to_str(namer)
        return f'{left_name} {self.op} {right_name}'

    def __len__(self):
        return len(self.left) + len(self.right)

    def __hash__(self):
        return hash(str(self))


class Not(UnaryNode):
    op = 'NOT'


class Or(BinaryNode):
    op = 'OR'


class And(BinaryNode):
    op = 'AND'