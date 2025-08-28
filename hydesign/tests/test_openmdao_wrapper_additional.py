# -*- coding: utf-8 -*-
"""
Additional tests for ComponentWrapper class in openmdao_wrapper.py

Created for increasing test coverage
"""
import numpy as np
import openmdao.api as om
import pytest

from hydesign.openmdao_wrapper import ComponentWrapper


def test_component_wrapper_initialization():
    """Test ComponentWrapper initialization with various input configurations."""
    # Test basic initialization
    comp = ComponentWrapper(
        inputs=[('x', {'val': 0.0})],
        outputs=[('y', {'val': 0.0})],
        function=lambda x: x + 1
    )
    
    assert len(comp.inputs) == 1
    assert len(comp.outputs) == 1
    assert comp.function is not None
    assert comp.gradient_function is None
    

def test_component_wrapper_with_additional_inputs():
    """Test ComponentWrapper with additional inputs."""
    def func(x, constant):
        return x + constant
    
    comp = ComponentWrapper(
        inputs=[('x', {'val': 0.0})],
        outputs=[('y', {'val': 0.0})],
        function=func,
        additional_inputs=[('constant', {'val': 5.0})]
    )
    
    prob = om.Problem()
    prob.model.add_subsystem('comp', comp)
    prob.setup()
    prob.set_val('comp.x', 2.0)
    prob.set_val('comp.constant', 3.0)
    prob.run_model()
    
    result = prob.get_val('comp.y')
    assert result == 5.0  # 2.0 + 3.0


def test_component_wrapper_with_additional_outputs():
    """Test ComponentWrapper with additional outputs."""
    def func(x):
        return x * 2, {'extra_output': x * 3}
    
    comp = ComponentWrapper(
        inputs=[('x', {'val': 0.0})],
        outputs=[('y', {'val': 0.0})],
        function=func,
        additional_outputs=[('extra_output', {'val': 0.0})]
    )
    
    prob = om.Problem()
    prob.model.add_subsystem('comp', comp)
    prob.setup()
    prob.set_val('comp.x', 4.0)
    prob.run_model()
    
    assert prob.get_val('comp.y') == 8.0  # 4.0 * 2
    assert prob.get_val('comp.extra_output') == 12.0  # 4.0 * 3


def test_component_wrapper_multiple_inputs_outputs():
    """Test ComponentWrapper with multiple inputs and outputs."""
    def func(x, y):
        return x + y, x * y
    
    comp = ComponentWrapper(
        inputs=[('x', {'val': 0.0}), ('y', {'val': 0.0})],
        outputs=[('sum', {'val': 0.0}), ('product', {'val': 0.0})],
        function=func
    )
    
    prob = om.Problem()
    prob.model.add_subsystem('comp', comp)
    prob.setup()
    prob.set_val('comp.x', 3.0)
    prob.set_val('comp.y', 4.0)
    prob.run_model()
    
    assert prob.get_val('comp.sum') == 7.0  # 3.0 + 4.0
    assert prob.get_val('comp.product') == 12.0  # 3.0 * 4.0


def test_component_wrapper_counter_property():
    """Test the counter property of ComponentWrapper."""
    def func(x):
        return x * 2
    
    comp = ComponentWrapper(
        inputs=[('x', {'val': 0.0})],
        outputs=[('y', {'val': 0.0})],
        function=func
    )
    
    prob = om.Problem()
    prob.model.add_subsystem('comp', comp)
    prob.setup()
    
    # Initial counter should be 0
    assert comp.counter == 0
    
    # Run the model to increment function evaluations
    prob.set_val('comp.x', 1.0)
    prob.run_model()
    
    # Counter should have increased
    assert comp.counter >= 1


def test_component_wrapper_with_partial_options():
    """Test ComponentWrapper with partial options."""
    def func(x):
        return x ** 2
    
    partial_options = [{'method': 'fd', 'step': 1e-6}]
    
    comp = ComponentWrapper(
        inputs=[('x', {'val': 0.0})],
        outputs=[('y', {'val': 0.0})],
        function=func,
        partial_options=partial_options
    )
    
    prob = om.Problem()
    prob.model.add_subsystem('comp', comp)
    prob.setup()
    prob.set_val('comp.x', 3.0)
    prob.run_model()
    
    assert prob.get_val('comp.y') == 9.0  # 3.0 ** 2


def test_component_wrapper_skip_linearize():
    """Test ComponentWrapper with skip_linearize attribute."""
    def func(x):
        return x + 1
    
    def grad(x):
        return [1.0]
    
    comp = ComponentWrapper(
        inputs=[('x', {'val': 0.0})],
        outputs=[('y', {'val': 0.0})],
        function=func,
        gradient_function=grad
    )
    
    # Set skip_linearize attribute
    comp.skip_linearize = True
    
    prob = om.Problem()
    prob.model.add_subsystem('comp', comp)
    prob.setup()
    prob.set_val('comp.x', 5.0)
    prob.run_model()
    
    assert prob.get_val('comp.y') == 6.0


def test_component_wrapper_with_array_inputs():
    """Test ComponentWrapper with array inputs and outputs."""
    def func(x):
        return np.sum(x), np.mean(x)
    
    comp = ComponentWrapper(
        inputs=[('x', {'val': np.zeros(3)})],
        outputs=[('total', {'val': 0.0}), ('average', {'val': 0.0})],
        function=func
    )
    
    prob = om.Problem()
    prob.model.add_subsystem('comp', comp)
    prob.setup()
    prob.set_val('comp.x', [1.0, 2.0, 3.0])
    prob.run_model()
    
    assert prob.get_val('comp.total') == 6.0  # 1 + 2 + 3
    assert prob.get_val('comp.average') == 2.0  # (1 + 2 + 3) / 3


def test_component_wrapper_single_tuple_inputs():
    """Test ComponentWrapper with single-element tuple inputs."""
    def func(x):
        return x * 2
    
    # Test with single-element tuples (should be converted to proper format)
    comp = ComponentWrapper(
        inputs=[('x',)],  # Single element tuple
        outputs=[('y',)], # Single element tuple  
        function=func
    )
    
    # Should not raise an error during initialization
    assert len(comp.inputs) == 1
    assert len(comp.outputs) == 1
    assert len(comp.inputs[0]) == 2  # Should have been expanded with empty dict


def test_component_wrapper_timing():
    """Test ComponentWrapper timing functionality."""
    def slow_func(x):
        # Simulate some computation time
        import time
        time.sleep(0.001)  # 1ms delay
        return x * 2
    
    comp = ComponentWrapper(
        inputs=[('x', {'val': 0.0})],
        outputs=[('y', {'val': 0.0})],
        function=slow_func
    )
    
    prob = om.Problem()
    prob.model.add_subsystem('comp', comp)
    prob.setup()
    prob.set_val('comp.x', 1.0)
    prob.run_model()
    
    # Check that timing was recorded
    assert comp.n_func_eval >= 1
    assert comp.func_time_sum > 0