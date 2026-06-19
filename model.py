"""
Build Your Own teenygrad: A Tiny Tensor Autograd Engine

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - prod
import math
def prod(shape):
    # TODO: Multiply together the elements of a shape tuple to get the total number of elements.
    result = math.prod(shape)
    return result

shape = (2,3,4)
print(prod)

# Step 2 - argsort
import torch
def argsort(values):
    # TODO: Return the indices that would sort values in ascending order.
    value = torch.tensor(values)
    result = torch.argsort(value)
    return result.tolist()

values = (3,1,2)
print(argsort)

# Step 3 - make_op_enums
from enum import Enum, auto
def make_op_enums():
    # TODO: create four enum classes naming every supported operation kind
    class UnaryOps(Enum):
        NEG = auto()
        RELU = auto()
        LOG = auto()
        EXP = auto()
        SQRT = auto()
        SIGMOID = auto()

    class BinaryOps(Enum):
        ADD = auto()
        SUB = auto()
        MUL = auto()
        DIV = auto()
        CMPLT = auto()
        MAX = auto()

    class ReduceOps(Enum):
        SUM = auto()
        MAX = auto()

    class MovementOps(Enum):
        RESHAPE = auto()
        EXPAND = auto()
        PERMUTE = auto()

    return (UnaryOps, BinaryOps, ReduceOps, MovementOps)

# Step 4 - LazyBuffer
class LazyBuffer:
    def __init__(self, np_array):
        # TODO: wrap np_array as an ndarray and expose shape and dtype
        conv_numpy = np.array(np_array)
        self._np = conv_numpy
        self.shape = conv_numpy.shape
        self.dtype = conv_numpy.dtype

# Step 5 - lazybuffer_const
def const(value, shape):
    # TODO: Create a new LazyBuffer of the given shape filled with a constant value.
    raw_array = np.full(shape, value, dtype = np.float32)
    return LazyBuffer(raw_array)

LazyBuffer.const = staticmethod(const)

# Step 6 - rand
def rand(shape, seed=None):
    # TODO: return a LazyBuffer of uniform random floats in [0, 1) with given shape
    rng = np.random.default_rng(seed)
    flt = rng.random(shape).astype(np.float32)
    return LazyBuffer(flt)

# Step 7 - lazybuffer_unary_e
def e(self, op):
    # TODO: apply a unary elementwise op (NEG, RELU, LOG, EXP, SQRT, SIGMOID)
    x =  self._np

    op_name = op.name if hasattr(op, 'name') else str(op)

    if op_name == 'NEG':
        out = -x
    elif op_name == 'RELU':
        out = np.maximum(0, x)
    elif op_name == 'LOG':
        out = np.log(x)
    elif op_name == 'EXP':
        out = np.exp(x)
    elif op_name == 'SQRT':
        out = np.sqrt(x)
    elif op_name == 'SIGMOID':
        out = 1.0 / (1.0 + np.exp(-x))
    else:
        raise ValueError(f"Unsupported unary operator: {op}")
    
    return LazyBuffer(out)

LazyBuffer.e = e

# Step 8 - lazybuffer_binary_e
def lazybuffer_binary_e(self, op, other):
    # TODO: apply a binary elementwise op between two LazyBuffers, return a new LazyBuffer
    x = self._np
    y = other._np

    op_name = op.name if hasattr(op, 'name') else str(op)

    if op_name == 'ADD':
        out = x + y
    elif op_name == 'SUB':
        out = x - y
    elif op_name == 'MUL':
        out = x * y
    elif op_name == 'DIV':
        out = x / y
    elif op_name == 'CMPLT':
        out = (x < y).astype(np.float32)
    elif op_name == 'MAX':
        out = np.maximum(x, y)
    else:
        raise ValueError(f"Unsupported binary operator: {op}")

    return LazyBuffer(out)

# Step 9 - lazybuffer_r
def r(self, op, axis):
    # TODO: reduce the underlying array along axis (SUM or MAX), keeping reduced dims as size 1
    x = self._np

    op_name = op.name if hasattr(op, 'name') else str(op)

    if op_name == 'SUM':
        out = np.sum(x, axis=axis, keepdims=True)
    elif op_name == 'MAX':
        out = np.max(x, axis=axis, keepdims=True)
    else:
        raise ValueError(f"Unsupported reduction operator: {op}")

    return LazyBuffer(out)

# Step 10 - lazybuffer_reshape
def reshape(self, new_shape):
    # TODO: return a new LazyBuffer with the array reshaped to new_shape
    reshaped_data = self._np.reshape(new_shape)
    return LazyBuffer(reshaped_data)

# Step 11 - lazybuffer_expand
def expand(self, new_shape):
    # TODO: broadcast this buffer's size-1 dims out to new_shape
    expanded_data = np.broadcast_to(self._np, new_shape)
    return LazyBuffer(expanded_data)

# Step 12 - lazybuffer_permute
def permute(self, order):
    # TODO: return a new LazyBuffer with axes reordered according to order
    permuted_data = self._np.transpose(order)
    return LazyBuffer(permuted_data)

    # ── Patch: attach reduce/movement methods to LazyBuffer (insert after Step 012) ──
UnaryOps, BinaryOps, ReduceOps, MovementOps = make_op_enums()

LazyBuffer.lazybuffer_binary_e = lazybuffer_binary_e   # from Step 008
LazyBuffer.lazybuffer_reduce_e = r                      # from Step 009
LazyBuffer.lazybuffer_movement_e = lambda self, op, arg: (
    reshape(self, arg) if op == MovementOps.RESHAPE else
    expand(self, arg)  if op == MovementOps.EXPAND  else
    permute(self, arg) if op == MovementOps.PERMUTE else
    (_ for _ in ()).throw(ValueError(f"bad movement op {op}"))
)

# Step 13 - Function
import torch

class Function:
    def __init__(self, *tensors):
        # TODO: record needs_input_grad, requires_grad, and parents for backprop
        self.needs_input_grad = [t.requires_grad for t in tensors]
        # 2. Determine requires_grad based on strict priority rules
        if not tensors:
            self.requires_grad = False
        elif any(t.requires_grad is True for t in tensors):
            self.requires_grad = True
        elif any(t.requires_grad is None for t in tensors):
            self.requires_grad = None
        else:
            self.requires_grad = False

        if self.requires_grad is True:
            self.parents = tensors

# Step 14 - function_forward_backward_stubs
def function_forward_backward_stubs():
    # TODO: attach forward and backward stubs to Function that raise NotImplementedError
    def forward(self, *args):
        raise NotImplementedError("Forward not implemented")

    def backward(self, *args):
        raise NotImplementedError("backward not implemented")

    Function.forward = forward
    Function.backward = backward

# Step 15 - apply
@classmethod
def apply(cls, *tensors, **kwargs):
    ctx = cls(*tensors)
    new = ctx.forward(*[t.lazydata for t in tensors], **kwargs)
    out = Tensor(new, requires_grad=ctx.requires_grad)
    if ctx.requires_grad:
        out._ctx = ctx
    return out

    pass


# Provided: attaches apply onto the Function base class. Leave this as-is.
for _obj in list(globals().values()):
    if isinstance(_obj, type):
        for _k in _obj.__mro__:
            if _k.__name__ == 'Function':
                _k.apply = apply

# Step 16 - Neg
import numpy as np

class Neg(Function):
    def forward(self, x):
        return x.__class__(-x._np)

    def backward(self, grad_output):
        return grad_output.__class__(-grad_output._np)

# Step 17 - Relu
import numpy as np

# 1. Inject the missing binary method back into the platform's LazyBuffer
if not hasattr(LazyBuffer, 'lazybuffer_binary_e'):
    def _binary_e(self, op, other):
        x = self._np
        y = other._np if hasattr(other, '_np') else other
        if op == 'MAX': res = np.maximum(x, y)
        elif op == 'CMPGT': res = (x > y).astype(np.float32)
        elif op == 'MUL': res = x * y
        elif op == 'DIV': res = x / y
        else: res = x
        return LazyBuffer(res)
    LazyBuffer.lazybuffer_binary_e = _binary_e

# 2. Implement the Relu Function using the restored method
class Relu(Function):
    def forward(self, x):
        self.x = x
        zero = x.const(0.0, x.shape)
        return x.lazybuffer_binary_e('MAX', zero)

    def backward(self, grad_output):
        zero = self.x.const(0.0, self.x.shape)
        mask = self.x.lazybuffer_binary_e('CMPGT', zero)
        return grad_output.lazybuffer_binary_e('MUL', mask)

# Step 18 - Log
import numpy as np

# 1. Inject the missing binary method back into the platform's LazyBuffer
if not hasattr(LazyBuffer, 'lazybuffer_binary_e'):
    def _binary_e(self, op, other):
        x = self._np
        y = other._np if hasattr(other, '_np') else other
        if op == 'MAX': res = np.maximum(x, y)
        elif op == 'CMPGT': res = (x > y).astype(np.float32)
        elif op == 'MUL': res = x * y
        elif op == 'DIV': res = x / y
        else: res = x
        return LazyBuffer(res)
    LazyBuffer.lazybuffer_binary_e = _binary_e

# 2. Implement the Log Function
class Log(Function):
    def forward(self, x):
        self.x = x
        return x.e('LOG') # Unary works perfectly with .e()

    def backward(self, grad_output):
        # Uses the patched binary method to cleanly execute division
        return grad_output.lazybuffer_binary_e('DIV', self.x)

# Step 19 - Exp
import numpy as np
if not hasattr(LazyBuffer, 'lazybuffer_binary_e'):
    def _binary_e(self, op, other):
        x = self._np
        y = other._np if hasattr(other, '_np') else other
        if op == 'MAX': res = np.maximum(x, y)
        elif op == 'CMPGT': res = (x > y).astype(np.float32)
        elif op == 'MUL': res = x * y
        elif op == 'DIV': res = x / y
        else: res = x
        return LazyBuffer(res)
    LazyBuffer.lazybuffer_binary_e = _binary_e


class Exp(Function):
    def forward(self, x):
        # TODO: compute the elementwise exponential and keep what backward needs
        self.ret = x.e('EXP')
        return self.ret

    def backward(self, grad_output):
        # TODO: turn the upstream gradient into the gradient w.r.t. the input
        return grad_output.lazybuffer_binary_e('MUL', self.ret)

# Step 20 - Sqrt
class Sqrt(Function):
    def forward(self, x):
        # TODO: compute the elementwise square root and cache it for backward
        self.ret = x.e('SQRT')
        return self.ret

    def backward(self, grad_output):
        half = grad_output.const(0.5, grad_output.shape)

        grad_scaled = grad_output.lazybuffer_binary_e('MUL', half)
        return grad_scaled.lazybuffer_binary_e('DIV', self.ret)

# Step 21 - Sigmoid
# Step 21 - Sigmoid
class Sigmoid(Function):
    def forward(self, x):
        self.ret = x.e(UnaryOps.SIGMOID)
        return self.ret

    def backward(self, grad_output):
        one = grad_output.const(1.0, grad_output.shape)
        one_minus_y = one.lazybuffer_binary_e(BinaryOps.SUB, self.ret)
        y_one_minus_y = self.ret.lazybuffer_binary_e(BinaryOps.MUL, one_minus_y)
        return grad_output.lazybuffer_binary_e(BinaryOps.MUL, y_one_minus_y)

# Step 22 - Add
class Add(Function):
    def forward(self, x, y):
        return x.lazybuffer_binary_e(BinaryOps.ADD, y)

    def backward(self, grad_output):
        grad_x = grad_output if self.needs_input_grad[0] else None
        grad_y = grad_output if self.needs_input_grad[1] else None
        return grad_x, grad_y

# Step 23 - Sub
class Sub(Function):
    def forward(self, x, y):
        return x.lazybuffer_binary_e(BinaryOps.SUB, y)

    def backward(self, grad_output):
        return (
            grad_output if self.needs_input_grad[0] else None,
            grad_output.e(UnaryOps.NEG) if self.needs_input_grad[1] else None
        )

# Step 24 - Mul
class Mul(Function):
    def forward(self, x, y):
        # Save inputs for the chain rule calculation
        self.x = x 
        self.y = y
        return x.lazybuffer_binary_e(BinaryOps.MUL, y)

    def backward(self, grad_output):
        # Explicitly checking indexes and routes None if an input doesn't need grad
        grad_x = grad_output.lazybuffer_binary_e(BinaryOps.MUL, self.y) if self.needs_input_grad[0] else None
        grad_y = grad_output.lazybuffer_binary_e(BinaryOps.MUL, self.x) if self.needs_input_grad[1] else None
        return grad_x, grad_y

# Step 25 - Div
class Div(Function):
    def forward(self, x, y):
        self.x = x
        self.y = y
        self.ret = x.lazybuffer_binary_e(BinaryOps.DIV, y)
        return self.ret

    def backward(self, grad_output):
        grad_x = grad_y = None
        if self.needs_input_grad[0]:
            grad_x = grad_output.lazybuffer_binary_e(BinaryOps.DIV, self.y)
        if self.needs_input_grad[1]:
            neg_grad = grad_output.e(UnaryOps.NEG)
            num = neg_grad.lazybuffer_binary_e(BinaryOps.MUL, self.x)
            den = self.y.lazybuffer_binary_e(BinaryOps.MUL, self.y)
            grad_y = num.lazybuffer_binary_e(BinaryOps.DIV, den)
        return grad_x, grad_y

# Step 26 - sum_function_forward
class Sum(Function):
    def forward(self, x, axis):
    # TODO: Reduce x with ReduceOps.SUM over axis (keepdims) and cache shape/axis.
        self.input_shape = x.shape
        return x.lazybuffer_reduce_e(ReduceOps.SUM, axis)

# Step 27 - sum_function_backward
def backward(self, grad_output):
    # TODO: broadcast the summed gradient back to the original input shape
    return grad_output.lazybuffer_movement_e(MovementOps.EXPAND, self.input_shape)

# Step 28 - max_function_forward
class Max(Function):
    def forward(self, x, axis):
        # TODO: reduce x with the MAX reduce op along axis and cache for backward
        self.x = x
        self.axis = axis
        self.ret = x.lazybuffer_reduce_e(ReduceOps.MAX, axis)
        return self.ret

# Step 29 - max_function_backward
def backward(self, grad_output):
    # TODO: route grad_output back to the input elements that were the maximum
       ret_expanded = self.ret.lazybuffer_movement_e(MovementOps.EXPAND, self.x.shape)
       lt1 = self.x.lazybuffer_binary_e(BinaryOps.CMPLT, ret_expanded)
       lt2 = ret_expanded.lazybuffer_binary_e(BinaryOps.CMPLT, self.x)
       not_lt1 = lt1.e(UnaryOps.NEG).lazybuffer_binary_e(BinaryOps.ADD, lt1.const(1.0, lt1.shape))
       not_lt2 = lt2.e(UnaryOps.NEG).lazybuffer_binary_e(BinaryOps.ADD, lt2.const(1.0, lt2.shape))
       mask = not_lt1.lazybuffer_binary_e(BinaryOps.MUL, not_lt2)

       counts = mask.lazybuffer_reduce_e(ReduceOps.SUM, self.axis).lazybuffer_movement_e(MovementOps.EXPAND, self.x.shape)
       normalized_mask = mask.lazybuffer_binary_e(BinaryOps.DIV, counts)
       return grad_output.lazybuffer_movement_e(MovementOps.EXPAND, self.x.shape).lazybuffer_binary_e(BinaryOps.MUL, normalized_mask)


Max.backward = backward

# Step 30 - Reshape
class Reshape(Function):
    def forward(self, x, shape):
        self.input_shape = x.shape
        return x.lazybuffer_movement_e(MovementOps.RESHAPE, shape)

    def backward(self, grad_output):
        return grad_output.lazybuffer_movement_e(MovementOps.RESHAPE, self.input_shape)

# Step 31 - expand_function_forward
def expand_function_forward(ctx, x, shape):
    ctx.input_shape = x.shape
    return x.lazybuffer_movement_e(MovementOps.EXPAND, shape)

# Step 32 - expand_function_backward
def expand_function_backward(ctx, grad_output):
    # TODO: Sum grad_output over the broadcast axes back to ctx.input_shape...
    axes = tuple(i for i, (in_dim, out_dim) in enumerate(
            zip(ctx.input_shape, grad_output.shape)) if in_dim == 1 and out_dim != 1)
    reduced = grad_output.lazybuffer_reduce_e(ReduceOps.SUM, axes) if axes else grad_output
    return reduced.lazybuffer_movement_e(MovementOps.RESHAPE, ctx.input_shape)

return forward, backward

_expand_forward, _expand_backward = expand_function_forward_backward()
Expand = type('Expand', (Function,), {'forward': _expand_forward, 'backward': _expand_backward})

# Step 33 - permute_function_forward_backward
def permute_function_forward_backward():
    # TODO: return (forward, backward); forward reorders axes, backward inverts the order
    def forward(self, x, order):
        self.order = order
        return x.lazybuffer_movement_e(MovementOps.PERMUTE, order)

    def backward(self, grad_output):
        inv_order = tuple(np.argsort(self.order))
        return grad_output.lazybuffer_movement_e(MovementOps.PERMUTE, inv_order)

    return (forward, backward)

# Step 34 - Tensor
class Tensor:
    def __init__(self, data, requires_grad=False, _ctx=None):
        if isinstance(data, LazyBuffer):
            self.data = data
        else:
            self.data = LazyBuffer(np.array(data, dtype=np.float32))

        self.requires_grad = requires_grad
        self.grad = None
        self._ctx = _ctx

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def lazydata(self):
        return self.data

    def numpy(self):
        return self.data._np

# Step 35 - tensor_from_data
def tensor_from_data(data, requires_grad=False):
    # TODO: wrap a number, list, or numpy array in a LazyBuffer held by a Tensor
    if isinstance(data, LazyBuffer):
        buf = data
    
    else:
        arr = np.array(data, dtype = np.float32)
        buf = LazyBuffer(arr)
    
    return Tensor(buf, requires_grad=requires_grad)

# Step 36 - tensor_creation_helpers
def tensor_creation_helpers():
    # TODO: return (zeros_fn, ones_fn, full_fn) building constant-filled Tensors
    def zeros_fn(shape, requires_grad = True):
        return Tensor(LazyBuffer(np.zeros(shape, dtype = np.float32)),requires_grad = requires_grad)

    def ones_fn(shape, requires_grad=False):
        return Tensor(LazyBuffer(np.ones(shape, dtype=np.float32)), requires_grad=requires_grad)

    def full_fn(shape, value, requires_grad=False):
        return Tensor(LazyBuffer(np.full(shape, value, dtype=np.float32)), requires_grad=requires_grad)


    return zeros_fn, ones_fn, full_fn

# Step 37 - tensor_randn
def tensor_randn(shape, seed=None, requires_grad=False):
    # TODO: Create a Tensor of standard-normal samples for the given shape.
    lazybuffer_rand = rand
    shape = tuple(shape)
    n = math.prod(shape) if shape else 1
    u_buf = rand((2 * n,), seed=seed)
    u = u_buf._np

    u1 = u[:n]
    u2 = u[n:2 * n]
    
    eps = 1e-12
    z = np.sqrt(-2.0 * np.log(u1 + eps)) * np.cos(2.0 * np.pi * u2)

    z = z.reshape(shape).astype(np.float32)
    return Tensor(LazyBuffer(z), requires_grad=requires_grad)

# Step 38 - build_topological_order
def build_topological_order(tensor):
    visited = set()
    order = []

    def dfs(node):
        if id(node) in visited:
            return
        visited.add(id(node))
        if node._ctx is not None:
            for parent in node._ctx.parents:
                dfs(parent)
        order.append(node)

    dfs(tensor)
    return order

# Step 39 - tensor_backward
# Step 39 - tensor_backward
def tensor_backward(tensor):
    order = build_topological_order(tensor)

    tensor.grad = Tensor(LazyBuffer(np.ones(tensor.shape, dtype=np.float32)))

    for node in reversed(order):
        if node._ctx is None:
            continue

        grad_output = node.grad.data
        grads = node._ctx.backward(grad_output)

        if not isinstance(grads, tuple):
            grads = (grads,)

        needs_grad = getattr(node._ctx, 'needs_input_grad', None)

        for i, (parent, g) in enumerate(zip(node._ctx.parents, grads)):
            if g is None:
                continue
            if needs_grad is not None and not needs_grad[i]:
                continue
            if needs_grad is None and not parent.requires_grad:
                continue  # fall back to checking the parent Tensor directly

            g_tensor = Tensor(g)
            if parent.grad is None:
                parent.grad = g_tensor
            else:
                parent.grad = Tensor(parent.grad.data.lazybuffer_binary_e(BinaryOps.ADD, g_tensor.data))

    return None

# Step 40 - bind_unary_tensor_methods
def bind_unary_tensor_methods():
    # TODO: map neg/relu/log/exp/sqrt/sigmoid names to callables using function_apply
     return {
        'neg': lambda t: Neg.apply(t),
        'relu': lambda t: Relu.apply(t),
        'log': lambda t: Log.apply(t),
        'exp': lambda t: Exp.apply(t),
        'sqrt': lambda t: Sqrt.apply(t),
        'sigmoid': lambda t: Sigmoid.apply(t),
    }

# Step 41 - broadcasted
def broadcasted(x, y):
    shape_x = x.shape
    shape_y = y.shape

    
    ndim = max(len(shape_x), len(shape_y))
    padded_x = (1,) * (ndim - len(shape_x)) + shape_x
    padded_y = (1,) * (ndim - len(shape_y)) + shape_y

    out_shape = tuple(max(a, b) for a, b in zip(padded_x, padded_y))

    def align(t, padded_shape):
        if t.shape == out_shape:
            return t  #
       
        if t.shape != padded_shape:
            t = Reshape.apply(t, shape=padded_shape)
        if padded_shape != out_shape:
            t = Expand.apply(t, shape=out_shape)
        return t

    new_x = align(x, padded_x)
    new_y = align(y, padded_y)

    return new_x, new_y

# Step 42 - bind_binary_tensor_methods (not yet solved)
# TODO: implement

# Step 43 - bind_movement_tensor_methods (not yet solved)
# TODO: implement

# Step 44 - bind_reduce_tensor_methods (not yet solved)
# TODO: implement

# Step 45 - tensor_mean (not yet solved)
# TODO: implement

# Step 46 - tensor_transpose (not yet solved)
# TODO: implement

# Step 47 - tensor_matmul_2d (not yet solved)
# TODO: implement

# Step 48 - tensor_softmax (not yet solved)
# TODO: implement

# Step 49 - tensor_log_softmax (not yet solved)
# TODO: implement

# Step 50 - sparse_categorical_cross_entropy (not yet solved)
# TODO: implement

# Step 51 - Linear (not yet solved)
# TODO: implement

# Step 52 - MLP (not yet solved)
# TODO: implement

# Step 53 - sgd_step (not yet solved)
# TODO: implement

# Step 54 - zero_grad (not yet solved)
# TODO: implement

# Step 55 - make_toy_digit_dataset (not yet solved)
# TODO: implement

# Step 56 - accuracy (not yet solved)
# TODO: implement

# Step 57 - train_mlp (not yet solved)
# TODO: implement

# Step 58 - evaluate_mlp (not yet solved)
# TODO: implement

