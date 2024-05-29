import tensorflow as tf
import numpy as np
import iisignature

_zero = np.array(0.0, dtype="float32")

def _sigGradImp(g, x, m):
    o = iisignature.sigbackprop(g, x, m)
    return o, _zero

@tf.custom_gradient
def Sig(x, m):
    def grad(dy):
        grad_val, _ = tf.py_function(func=_sigGradImp, inp=[dy, x, m], Tout=[tf.float32, tf.float32])
        return grad_val, None

    y = tf.py_function(func=lambda x, m: iisignature.sig(x, m), inp=[x, m], Tout=tf.float32)
    return y, grad

class SigLayer(tf.keras.layers.Layer):
    def __init__(self, m, **kwargs):
        super(SigLayer, self).__init__(**kwargs)
        self.m = m

    def build(self, input_shape):
        self.output_shape = (*input_shape[:-2], iisignature.siglength(input_shape[-1], self.m))
        self.built = True
        super().build(input_shape)

    def call(self, inputs):
        res=Sig(inputs, self.m)
        res.set_shape(self.output_shape)
        return res

    def compute_output_shape(self, input_shape):
        if not self.built:
            self.build(input_shape)
        return self.output_shape


def _logSigGradImp(g, x, s, method):
    return iisignature.logsigbackprop(g, x, s, method)

def LogSig(b, m, method):
    s = iisignature.prepare(b, m)
    @tf.custom_gradient
    def _LogSig(x):
        def grad(dy):
            grad_fn = lambda dy, x: _logSigGradImp(dy, x, s, method)
            grad_val = tf.py_function(func=grad_fn, inp=[dy, x], Tout=tf.float32)
            return grad_val
            
        y = tf.py_function(func=lambda xi: iisignature.logsig(xi, s, ""), inp=[x], Tout=tf.float32)
        return y, grad
    return _LogSig

class LogSigLayer(tf.keras.layers.Layer):
    def __init__(self, m, method="", **kwargs):
        super(LogSigLayer, self).__init__(**kwargs)
        self.m = m
        self.method = method

    def build(self, input_shape):
        self.s = iisignature.prepare(input_shape[-1], self.m)
        self.b = input_shape[-1]
        self._logsig = LogSig(self.b, self.m, self.method)
        self.output_shape_ = (*input_shape[:-2], iisignature.logsiglength(input_shape[-1], self.m))
        super().build(input_shape)

    def call(self, inputs):
        res = self._logsig(inputs)
        res.set_shape(self.output_shape_)
        return res

    def compute_output_shape(self, input_shape):
        return self.output_shape_

def _sigScaleGradImp(g, x, y, m):
    o = iisignature.sigscalebackprop(g, x, y, m)
    return o[0], o[1], _zero

@tf.custom_gradient
def SigScale(x, y, m):
    def grad(dy):
        grad_val_x, grad_val_y, _ = tf.py_function(func=_sigScaleGradImp, inp=[dy, x, y, m], Tout=[tf.float32, tf.float32, tf.float32])
        return grad_val_x, grad_val_y, None
    z = tf.py_function(func=lambda x, y, m: iisignature.sigscale(x, y, m), inp=[x, y, m], Tout=tf.float32)
    return z, grad

class SigScaleLayer(tf.keras.layers.Layer):
    def __init__(self, m, **kwargs):
        super(SigScaleLayer, self).__init__(**kwargs)
        self.m = m

    def build(self, input_shape):
        self.output_shape_ = input_shape[0]
        super().build(input_shape)

    def call(self, inputs):
        sigs, scale = inputs
        res =  SigScale(sigs, scale, self.m)
        res.set_shape(self.output_shape_)
        return res

    def compute_output_shape(self, input_shapes):
        return input_shapes[0]


# Gradient implementation
def _sigJoinGradImp(g, x, y, m):
    o = iisignature.sigjoinbackprop(g, x, y, m)
    return o[0], o[1], _zero

def _sigJoinGradFixedImp(g, x, y, m, fixed_last):
    o = iisignature.sigjoinbackprop(g, x, y, m, fixed_last)
    return o[0], o[1], _zero, np.array(o[2], dtype="float32")


# Custom gradient function
def SigJoin(m, fixed_last=None):
    if fixed_last is None:
        @tf.custom_gradient
        def _SigJoin(x, y):
            def grad(dy):
                grad_val_x, grad_val_y, _ = tf.py_function(func=_sigJoinGradImp, inp=[dy, x, y, m], Tout=[tf.float32, tf.float32, tf.float32])
                return grad_val_x, grad_val_y
            z = tf.py_function(func=lambda x, y, m: iisignature.sigjoin(x, y, m), inp=[x, y, m], Tout=tf.float32)
            return z, grad
    else:
        raise NotImplementedError('Fixed last method not implemented yet')
    return _SigJoin

# Keras layer for SigJoin
class SigJoinLayer(tf.keras.layers.Layer):
    def __init__(self, m, fixed_last=None, **kwargs):
        super(SigJoinLayer, self).__init__(**kwargs)
        self.m = m
        self.fixed_last = fixed_last

    def build(self, input_shape):
        self.output_shape_ = input_shape[0]
        self._sigjoin = SigJoin(self.m, self.fixed_last)
        super().build(input_shape)

    def call(self, inputs):
        x, y = inputs
        res = self._sigjoin(x, y)
        res.set_shape(self.output_shape_)
        return res

    def compute_output_shape(self, input_shape):
        return self.output_shape_

