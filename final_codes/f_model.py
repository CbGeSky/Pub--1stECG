# 最终的三个模型结构
# f_model.py

from keras.models import *
from keras.layers import *

def dot_product(x, kernel):
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)

class AttentionWithContext(Layer):
    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')
        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
            self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)
        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)
        if self.bias:
            uit += self.b
        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)
        a = K.exp(ait)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

def build_model_01(num_classes=9,len_target=25000):
    num_filter=12
    main_input = Input(shape=(len_target,12), dtype='float32', name='main_input')
    # block 1
    x = Conv1D(num_filter, 3, padding='same')(main_input)
    x = LeakyReLU(alpha=0.3)(x)
    x = Conv1D(num_filter, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Conv1D(num_filter, 24, strides = 2, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)
    # block 2
    x = Conv1D(num_filter, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Conv1D(num_filter, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Conv1D(num_filter, 24, strides = 2, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)
    # block 3
    x = Conv1D(num_filter, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Conv1D(num_filter, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Conv1D(num_filter, 24, strides = 2, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)

    # block 4
    x = Conv1D(num_filter, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Conv1D(num_filter, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Conv1D(num_filter, 24, strides = 2, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)
    # block 5
    x = Conv1D(num_filter, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Conv1D(num_filter, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Conv1D(num_filter, 48, strides = 2, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    cnnout = Dropout(0.2)(x)
    # GPU only version
    #x = Bidirectional(CuDNNGRU(12,return_sequences=True,return_state=False))(cnnout)
    x = Bidirectional(GRU(12,return_sequences=True,return_state=False))(cnnout)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)
    x = AttentionWithContext()(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)
    # for multi-labeled problems, change from softmax to sigmoid
    # main_output = Dense(num_classes,activation='softmax')(x)
    main_output = Dense(num_classes,activation='sigmoid')(x)
    model = Model(main_input,main_output)
    return model

