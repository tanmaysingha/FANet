# -*- coding: utf-8 -*-
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Implementation of FANet using Tensorflow 2.1.0 and Keras 2.3.0

"""

from functools import reduce
import tensorflow as tf
from tensorflow import keras

#### Custom function for conv2d: conv_block
def conv_block(inputs, conv_type, kernel, kernel_size, strides, padding='same', relu=True):
  
  if(conv_type == 'ds'):
    x = tf.keras.layers.SeparableConv2D(kernel, kernel_size, padding=padding, strides = strides)(inputs)
  else:
    x = tf.keras.layers.Conv2D(kernel, kernel_size, padding=padding, strides = strides)(inputs)  
  
  x = tf.keras.layers.BatchNormalization()(x)
  
  if (relu):
    #x = tf.keras.activations.relu(x)
    x = tf.keras.layers.Activation(lambda x: tf.nn.swish(x)) (x)
  
  return x

#### residual custom method
def _res_bottleneck(inputs, filters, kernel, t, s, r=False):
    
    
    tchannel = tf.keras.backend.int_shape(inputs)[-1] * t

    x = conv_block(inputs, 'conv', tchannel, (1, 1), strides=(1, 1))

    x = tf.keras.layers.DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    #x = tf.keras.activations.relu(x)
    x = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(x)

    x = conv_block(x, 'conv', filters, (1, 1), strides=(1, 1), padding='same', relu=False)

    if r:
        x = tf.keras.layers.add([x, inputs])
    return x

"""#### Bottleneck custom method"""

def bottleneck_block(inputs, filters, kernel, t, strides, n):
  x = _res_bottleneck(inputs, filters, kernel, t, strides)
  
  for i in range(1, n):
    x = _res_bottleneck(x, filters, kernel, t, 1, True)

  return x    

MOMENTUM = 0.997
EPSILON = 1e-4

def SeparableConvBlock(num_channels, kernel_size, strides, freeze_bn=False):
    f1 = tf.keras.layers.SeparableConv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same',
                                use_bias=True, dilation_rate=2,)
    
    f2 = tf.keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name='')
    #f2 = BatchNormalization(freeze=freeze_bn, name=f'{name}/bn')
    return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), (f1, f2))


def model(num_classes=19, input_size=(1024, 2048, 3)):

  # Input Layer
  input_layer = tf.keras.layers.Input(shape=input_size, name = 'input_layer')

  ## Step 1: Learning to DownSample
  lds_layer = conv_block(input_layer, 'conv', 32, (3, 3), strides = (2, 2))
  lds_layer = conv_block(lds_layer, 'ds', 48, (3, 3), strides = (2, 2))
  lds_layer = conv_block(lds_layer, 'ds', 64, (3, 3), strides = (2, 2))
  C3 = lds_layer

  ## Step 2: Bottleneck Blocks
  C4 = bottleneck_block(C3, 64, (3, 3), t=6, strides=2, n=3)
  C5 = bottleneck_block(C4, 96, (3, 3), t=6, strides=2, n=3)
  C5 = bottleneck_block(C5, 128, (3, 3), t=6, strides=1, n=3)

  """## Step 3: BiFPN"""

  P3_in = C3
  P4_in = C4
  P5_in = C5
  P6_in = tf.keras.layers.Conv2D(64, 1, 1, padding='same', activation=None, name='resample_p6/conv2d')(C5)
  P6_in = tf.keras.layers.BatchNormalization()(P6_in)

  P6_in = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='resample_p6/maxpool')(P6_in)
  P7_in = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='resample_p7/maxpool')(P6_in)
  P7_U = tf.keras.layers.UpSampling2D()(P7_in)
  P6_td = tf.keras.layers.Add()([P6_in, P7_U])
  P6_td = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P6_td)
  P6_td = SeparableConvBlock(num_channels=64, kernel_size=3, strides=1,)(P6_td)
  P5_in_1 = tf.keras.layers.Conv2D(64, 1, 1, padding='same', activation=None, name=f'fpn_cells/cell/fnode1/conv2d')(P5_in)
  P5_in_1 = tf.keras.layers.BatchNormalization()(P5_in_1)
  P6_U = tf.keras.layers.UpSampling2D()(P6_td)
  P5_td = tf.keras.layers.Add()([P5_in_1, P6_U])
  P5_td = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P5_td)
  P5_td = SeparableConvBlock(num_channels=64, kernel_size=3, strides=1,)(P5_td)

  P5_U = tf.keras.layers.UpSampling2D()(P5_td)
  P4_td = tf.keras.layers.Add()([P4_in, P5_U])
  P4_td = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P4_td)
  P4_td = SeparableConvBlock(num_channels=64, kernel_size=3, strides=1,)(P4_td)

  P4_U = tf.keras.layers.UpSampling2D()(P4_td)
  P3_out = tf.keras.layers.Add()([P3_in, P4_U])
  P3_out = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P3_out)
  P3_out = SeparableConvBlock(num_channels=64, kernel_size=3, strides=1,)(P3_out)

  P3_D = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P3_out)
  P4_out = tf.keras.layers.Add()([P4_in, P4_td, P3_D])
  P4_out = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P4_out)
  P4_out = SeparableConvBlock(num_channels=64, kernel_size=3, strides=1,)(P4_out)
  P5_in_2 = tf.keras.layers.Conv2D(64, 1, 1, padding='same', activation=None, name = f'fpn_cells/cell/fnode5/conv2d')(P5_in)
  P5_in_2 = tf.keras.layers.BatchNormalization()(P5_in_2)
  P4_D = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P4_out)

  P5_out = tf.keras.layers.Add()([P5_in_2, P5_td, P4_D])
  P5_out = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P5_out)
  P5_out = SeparableConvBlock(num_channels=64, kernel_size=3, strides=1,)(P5_out)
  P5_D = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P5_out)
  P6_out = tf.keras.layers.Add()([P6_in, P6_td, P5_D])
  P6_out = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P6_out)
  P6_out = SeparableConvBlock(num_channels=64, kernel_size=3, strides=1,)(P6_out)
  P6_D = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_out)
  P7_out = tf.keras.layers.Add()([P7_in, P6_D])
  P7_out = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P7_out)
  P7_out = SeparableConvBlock(num_channels=64, kernel_size=3, strides=1)(P7_out)

  P7_final_U = tf.keras.layers.UpSampling2D()(P7_out)

  P6_U = tf.keras.layers.Add()([P7_final_U, P6_out, P6_in]) 
  P6_U = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P6_U)
  P6_U = SeparableConvBlock(num_channels=64, kernel_size=3, strides=1)(P6_U)
  P6_U = tf.keras.layers.UpSampling2D()(P6_U)
  P5_U = tf.keras.layers.Add()([P5_out, P6_U])
 
  P5_U = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P5_U)
  P5_U = SeparableConvBlock(num_channels=64, kernel_size=3, strides=1)(P5_U)
  P5_U = tf.keras.layers.UpSampling2D()(P5_U)
  P4_U = tf.keras.layers.Add()([P4_out, P5_U])
  P4_U = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P4_U)
  P4_U = SeparableConvBlock(num_channels=64, kernel_size=3, strides=1)(P4_U)
  P4_U = tf.keras.layers.UpSampling2D()(P4_U)
  P3_U = tf.keras.layers.Add()([P3_in, P3_out, P4_U]) 
  P3_U = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P3_U)
  P3_U = SeparableConvBlock(num_channels=64, kernel_size=3, strides=1)(P3_U)

  """## Step 4: Classifier"""

  classifier = tf.keras.layers.SeparableConv2D(64, (3, 3), padding='same', strides = (1, 1), name = 'DSConv1_classifier')(P3_U)
  classifier = tf.keras.layers.BatchNormalization()(classifier)
  #classifier = tf.keras.activations.relu(classifier)
  classifier = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(classifier)

  classifier = tf.keras.layers.SeparableConv2D(64, (3, 3), padding='same', strides = (1, 1), name = 'DSConv2_classifier')(classifier)
  classifier = tf.keras.layers.BatchNormalization()(classifier)
  #classifier = tf.keras.activations.relu(classifier)
  classifier = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(classifier)

  
  #classifier = conv_block(classifier, 'conv', num_classes, (1, 1), strides=(1, 1), padding='same', relu=False)
  classifier = tf.keras.layers.Conv2D(20, 1, 1, padding='same', activation=None,
                                     kernel_regularizer=keras.regularizers.l2(0.00004), 
                                     bias_regularizer=keras.regularizers.l2(0.00004))(classifier)
  
  classifier = tf.keras.layers.Dropout(0.35)(classifier)

  classifier = tf.keras.layers.UpSampling2D((8, 8))(classifier)

  classifier = tf.dtypes.cast(classifier, tf.float32)
  classifier = tf.keras.activations.softmax(classifier)

  FANet = tf.keras.Model(inputs = input_layer , outputs = classifier, name = 'FANet')

  return FANet
