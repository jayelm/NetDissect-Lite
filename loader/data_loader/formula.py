import random


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

    def __repr__(self):
        return f"Leaf({str(self)})"

    def get_vals(self):
        return [self.val]


class Node(F):
    pass


class UnaryNode(Node):
    op = None

    def __init__(self, val):
        self.val = val

    def __str__(self):
        return f'({self.op} {self.val})'

    def to_str(self, namer):
        not_name = self.val.to_str(namer)
        return f'({self.op} {not_name})'

    def __len__(self):
        return 1 + len(self.val)

    def __hash__(self):
        return hash(str(self))

    def __repr__(self):
        return f"{self.op}({self.val})"

    def get_vals(self):
        return self.val.get_vals()


class BinaryNode(Node):
    op = None

    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __str__(self):
        return f'({self.left} {self.op} {self.right})'

    def to_str(self, namer):
        left_name = self.left.to_str(namer)
        right_name = self.right.to_str(namer)
        return f'({left_name} {self.op} {right_name})'

    def __len__(self):
        return len(self.left) + len(self.right)

    def __hash__(self):
        return hash(str(self))

    def __repr__(self):
        return f"{self.op}({self.left}, {self.right})"

    def get_vals(self):
        vals = []
        vals.extend(self.left.get_vals())
        vals.extend(self.right.get_vals())
        return vals


class Not(UnaryNode):
    op = 'NOT'


class Or(BinaryNode):
    op = 'OR'


class And(BinaryNode):
    op = 'AND'


UNARY_OPS = [Not]
BINARY_OPS = [Or, And]


def parse(fstr, reverse_namer=lambda x: x):
    """
    Parse a string representation back into formula.
    Reverse_namer converts back from names to actual integer indices
    """
    if not fstr:
        raise ValueError('empty string')
    if fstr[0] == '(' and fstr[-1] == ')':
        # Composition
        fstr = fstr[1:-1]
        # Check unary ops
        # XXX: if ops overlap, parsing will not work!
        for unop in UNARY_OPS:
            unop_ = f"{unop.op} "
            if fstr.startswith(unop_):
                val_f = parse(fstr[len(unop_):], reverse_namer=reverse_namer)
                return unop(val_f)
        for binop in BINARY_OPS:
            _binop_ = f" {binop.op} "
            if _binop_ in fstr:
                fst, snd = fstr.split(_binop_)
                # Janky parsing - fst and snd must have equal number of ()s
                if fst.count('(') != fst.count(')') or snd.count('(') != snd.count(')'):
                    continue
                val_fst = parse(fst, reverse_namer=reverse_namer)
                val_snd = parse(snd, reverse_namer=reverse_namer)
                return binop(val_fst, val_snd)
        raise ValueError(f"Couldn't parse {fstr}")
    else:
        return Leaf(reverse_namer(fstr))

def minor_negate(f):
    """
    Randomly negate a leaf
    """
    if isinstance(f, Leaf):
        return Not(f)
    elif isinstance(f, Not):
        # Special case: if the val is a leaf, just return the val itself
        if isinstance(f.val, Leaf):
            return f.val
        else:
            return Not(minor_negate(f.val))
    elif isinstance(f, And):
        # Binary
        if random.random() < 0.5:
            return And(minor_negate(f.left), f.right)
        else:
            return And(f.left, minor_negate(f.right))
    elif isinstance(f, Or):
        # Binary
        if random.random() < 0.5:
            return Or(minor_negate(f.left), f.right)
        else:
            return Or(f.left, minor_negate(f.right))
    else:
        raise RuntimeError
