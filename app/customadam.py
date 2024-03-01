from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras import backend as K
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import training_ops
import tensorflow as tf

class CustomAdam(Optimizer):
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False, name='CustomAdam', **kwargs):
        super(CustomAdam, self).__init__(name, **kwargs)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self.epsilon = epsilon or K.epsilon()
        self.amsgrad = amsgrad

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'm')
            self.add_slot(var, 'v')
            if self.amsgrad:
                self.add_slot(var, 'vhat')

    @tf_utils.shape_type_conversion
    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        beta_1_t = self._get_hyper('beta_1', var_dtype)
        beta_2_t = self._get_hyper('beta_2', var_dtype)
        epsilon_t = tf.convert_to_tensor(self.epsilon, var_dtype)
        one = tf.constant(1, var_dtype)
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')

        m_t = (beta_1_t * m) + (one - beta_1_t) * grad
        v_t = (beta_2_t * v) + (one - beta_2_t) * K.square(grad)

        var_t = var - lr_t * m_t / (K.sqrt(v_t) + epsilon_t)

        var_update = var.assign(var_t)

        return control_flow_ops.group(*[var_update, m_t, v_t])

    def _resource_apply_sparse(self, grad, var, indices):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        beta_1_t = self._get_hyper('beta_1', var_dtype)
        beta_2_t = self._get_hyper('beta_2', var_dtype)
        epsilon_t = tf.convert_to_tensor(self.epsilon, var_dtype)
        one = tf.constant(1, var_dtype)
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')

        m_t = (beta_1_t * m) + (one - beta_1_t) * grad
        v_t = (beta_2_t * v) + (one - beta_2_t) * K.square(grad)

        var_t = var - lr_t * m_t / (K.sqrt(v_t) + epsilon_t)

        with K.control_dependencies([var_t]):
            var_update = self._resource_scatter_add(var, indices, tf.gather(var - var_t, indices))

        return control_flow_ops.group(*[var_update, m_t, v_t])

    def get_config(self):
        config = {
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'epsilon': self.epsilon,
            'amsgrad': self.amsgrad,
        }
        base_config = super(CustomAdam, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
