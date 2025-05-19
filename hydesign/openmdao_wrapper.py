from openmdao.core.explicitcomponent import ExplicitComponent
import time

class ComponentWrapper(ExplicitComponent):
    def __init__(self, inputs, outputs, function, gradient_function=None,
                 additional_inputs=None, additional_outputs=None, partial_options=None,
                 **kwargs):
        """
        Parameters
        ----------
        inputs : list of dict
            dicts have the following required keys: ['name'] and optional keys and default values:
                dict(val=1.0, shape=None, units=None, desc='', tags=None,
                     shape_by_conn=False, copy_shape=None, compute_shape=None,
                     require_connection=False, distributed=None, primal_name=None)
        outputs : list of dict
            dicts have the following required keys: ['name'] and optional keys and default values:
                dict(val=1.0, shape=None, units=None, res_units=None, desc='',
                     lower=None, upper=None, ref=1.0, ref0=0.0, res_ref=None, tags=None,
                     shape_by_conn=False, copy_shape=None, compute_shape=None, distributed=None,
                     primal_name=None)
        function : callable
            pure python function. needs to return in the format output1, output2, {additional outputs}
        gradient_function : callable, optional
            pure python gradient function. The default is None. needs to return in the format [[d_output1/d_input1, d_output1/d_input2,...], [d_output2/d_input1, ...] ...]
        additional_inputs : list of dict, optional
            same as inputs but constant so gradients for this will not be calculated. The default is None.
        additional_outputs : list of dict, optional
            same as outputs but constant so gradients for this will not be calculated. The default is None.
        partial_options : list of dict, optional
            dicts have the following optional keys and default values:
                dict(dependent=True, rows=None, cols=None, val=None,
                     step=None, form=None, step_calc=None, minimum_step=None)
        **kwargs : dict
            kwargs.

        Returns
        -------
        None.

        """
        self.inputs = inputs
        self.outputs = outputs
        self.function = function
        self.gradient_function = gradient_function
        self.additional_inputs = additional_inputs
        self.additional_outputs = additional_outputs
        self.partial_options = partial_options
        self.kwargs = kwargs
        
        self.input_keys = [i['name'] for i in self.inputs]
        self.additional_input_keys = [i['name'] for i in self.additional_inputs]
        self.all_input_keys = self.input_keys + self.additional_input_keys
        self.output_keys = [o['name'] for o in self.outputs]

        self.n_func_eval = 0
        self.func_time_sum = 0
        self.n_grad_eval = 0
        self.grad_time_sum = 0
        
    
    def setup(self):
        for inp in self.inputs:
            self.add_input(**inp)
        if self.additional_inputs is not None:
            for a_inp in self.additional_inputs:
                self.add_input(**a_inp)
        for out in self.outputs:
            self.add_output(**out)
        if self.additional_outputs is not None:
            for a_out in self.additional_outputs:
                self.add_output(**a_out)
        if self.partial_options is None:
            partial_options = len(self.outputs) * [{}]
        else:
            partial_options = self.partial_options
        if self.gradient_function is None:
            method = 'fd'
        else:
            method = 'exact'
        for out, po in zip(self.outputs, partial_options):
            self.declare_partials(out['name'], [i['name'] for i in self.inputs], method=method, **po)

    @property
    def counter(self):
        counter = float(self.n_func_eval)
        if self.grad_time_sum > 0 and self.func_time_sum > 0 and self.n_grad_eval > 0 and self.n_func_eval > 0:
            ratio = ((self.grad_time_sum / self.n_grad_eval) /
                     (self.func_time_sum / self.n_func_eval))
            counter += self.n_grad_eval * max(ratio, 1)
        else:
            counter += self.n_grad_eval
        return int(counter)


    def compute(self, inputs, outputs):
        """Compute cost model"""
        t = time.time()
        if self.additional_outputs is not None:
            res, additional_output = self.function(**{x: inputs[x] for x in self.all_input_keys})
            for k, v in additional_output.items():
                outputs[k] = v
        else:
            res = self.function(**{x: inputs[x] for x in self.all_input_keys})
        if not isinstance(res, list):
            res = [res]
        for o, r in zip(self.output_keys, res):
            outputs[o] = r
        self.func_time_sum += time.time() - t
        self.n_func_eval += 1

    def compute_partials(self, inputs, J):
        if hasattr(self, 'skip_linearize'):
            if self.skip_linearize:
                return

        t = time.time()
        if self.gradient_function is not None:
            for k, d_out_d_k in zip(self.input_keys,
                                  self.gradient_function(**{x: inputs[x] for x in self.all_input_keys})):
                if d_out_d_k is not None:
                    if not isinstance(d_out_d_k, list):
                        d_out_d_k = [d_out_d_k]
                    for o, d in zip(self.output_keys, d_out_d_k):
                        J[o, k] = d
        self.grad_time_sum += time.time() - t
        self.n_grad_eval += 1
