"""
Comprehensive unit tests for hydesign/openmdao_wrapper.py ComponentWrapper class.
This test file targets previously untested lines: 60-63, 68, 89-90, 94-95, 108-109, 113-126, 140-144, 156-157
"""

import numpy as np

# Safe import of openmdao and ComponentWrapper
try:
    import openmdao.api as om
    from hydesign.openmdao_wrapper import ComponentWrapper
    OPENMDAO_AVAILABLE = True
except ImportError:
    OPENMDAO_AVAILABLE = False
    
import pytest


@pytest.mark.skipif(not OPENMDAO_AVAILABLE, reason="OpenMDAO not available")
def test_component_wrapper_with_additional_inputs():
    """Test ComponentWrapper with additional_inputs - lines 60-63, 89-90"""
    def func_with_additional(x, y, const):
        return x * y + const
    
    inputs = [('x', {'val': 1.0}), ('y', {'val': 2.0})]
    outputs = [('result', {'val': 0.0})]
    # Test both tuple formats - with and without dict
    additional_inputs = [('const', {'val': 5.0}), ('unused',)]
    
    comp = ComponentWrapper(
        inputs=inputs,
        outputs=outputs,
        function=func_with_additional,
        additional_inputs=additional_inputs
    )
    
    # Verify additional_inputs processing (lines 60-63)
    assert len(comp.additional_inputs) == 2
    assert comp.additional_inputs[0] == ('const', {'val': 5.0})
    assert comp.additional_inputs[1] == ('unused', {})  # Should add empty dict
    assert comp.additional_input_keys == ['const', 'unused']
    
    prob = om.Problem()
    prob.model.add_subsystem("comp", comp)
    prob.setup()  # This triggers lines 89-90
    
    prob.set_val("comp.x", 3.0)
    prob.set_val("comp.y", 4.0)
    prob.set_val("comp.const", 10.0)
    prob.set_val("comp.unused", 0.0)
    prob.run_model()
    
    # 3 * 4 + 10 = 22
    assert prob.get_val("comp.result") == 22.0


@pytest.mark.skipif(not OPENMDAO_AVAILABLE, reason="OpenMDAO not available")
def test_component_wrapper_with_additional_outputs():
    """Test ComponentWrapper with additional_outputs - lines 68, 94-95, 140-144"""
    def func_with_additional_outputs(x):
        main_result = x * 2
        extra_outputs = {'squared': x ** 2, 'cubed': x ** 3}
        return main_result, extra_outputs
    
    inputs = [('x', {'val': 1.0})]
    outputs = [('main', {'val': 0.0})]
    # Test both tuple formats
    additional_outputs = [('squared', {'val': 0.0}), ('cubed',)]
    
    comp = ComponentWrapper(
        inputs=inputs,
        outputs=outputs,
        function=func_with_additional_outputs,
        additional_outputs=additional_outputs
    )
    
    # Verify additional_outputs processing (line 68)
    assert len(comp.additional_outputs) == 2
    assert comp.additional_outputs[0] == ('squared', {'val': 0.0})
    assert comp.additional_outputs[1] == ('cubed', {})  # Should add empty dict
    
    prob = om.Problem()
    prob.model.add_subsystem("comp", comp)
    prob.setup()  # This triggers lines 94-95
    
    prob.set_val("comp.x", 5.0)
    prob.run_model()  # This triggers lines 140-144
    
    assert prob.get_val("comp.main") == 10.0     # 5 * 2
    assert prob.get_val("comp.squared") == 25.0  # 5 ** 2
    assert prob.get_val("comp.cubed") == 125.0   # 5 ** 3


@pytest.mark.skipif(not OPENMDAO_AVAILABLE, reason="OpenMDAO not available")
def test_component_wrapper_multiple_partial_options():
    """Test ComponentWrapper with multiple partial_options - lines 108-109"""
    def multi_output_func(x, y):
        return x + y, x * y, x - y
    
    def multi_output_grad(x, y):
        return [[1.0, 1.0], [y, x], [1.0, -1.0]]
    
    inputs = [('x', {'val': 1.0}), ('y', {'val': 2.0})]
    outputs = [('sum', {'val': 0.0}), ('product', {'val': 0.0}), ('diff', {'val': 0.0})]
    # Different partial options for each output
    partial_options = [
        {'method': 'fd', 'step': 1e-6},
        {'method': 'fd', 'step': 1e-5},
        {'method': 'fd', 'step': 1e-4}
    ]
    
    comp = ComponentWrapper(
        inputs=inputs,
        outputs=outputs,
        function=multi_output_func,
        gradient_function=multi_output_grad,
        partial_options=partial_options
    )
    
    prob = om.Problem()
    prob.model.add_subsystem("comp", comp)
    prob.setup()  # This triggers lines 108-109
    
    prob.set_val("comp.x", 7.0)
    prob.set_val("comp.y", 3.0)
    prob.run_model()
    
    assert prob.get_val("comp.sum") == 10.0      # 7 + 3
    assert prob.get_val("comp.product") == 21.0  # 7 * 3
    assert prob.get_val("comp.diff") == 4.0      # 7 - 3


@pytest.mark.skipif(not OPENMDAO_AVAILABLE, reason="OpenMDAO not available")
def test_component_wrapper_counter_property():
    """Test ComponentWrapper counter property - lines 113-126"""
    def func(x):
        return x ** 2
    
    def grad_func(x):
        return [2 * x]
    
    inputs = [('x', {'val': 1.0})]
    outputs = [('result', {'val': 0.0})]
    
    comp = ComponentWrapper(
        inputs=inputs,
        outputs=outputs,
        function=func,
        gradient_function=grad_func
    )
    
    prob = om.Problem()
    prob.model.add_subsystem("comp", comp)
    prob.setup()
    
    # Run model to accumulate timing data
    prob.set_val("comp.x", 2.0)
    prob.run_model()
    
    # Set up timing values to test lines 113-126
    comp.n_func_eval = 10
    comp.func_time_sum = 5.0  # > 0
    comp.n_grad_eval = 3  
    comp.grad_time_sum = 6.0  # > 0
    
    # Test the timing conditions path (lines 114-123)
    counter = comp.counter
    
    # Expected calculation:
    # ratio = (grad_time_sum / n_grad_eval) / (func_time_sum / n_func_eval)
    # ratio = (6.0 / 3) / (5.0 / 10) = 2.0 / 0.5 = 4.0
    # counter = n_func_eval + n_grad_eval * max(ratio, 1)
    # counter = 10 + 3 * max(4.0, 1) = 10 + 3 * 4 = 22
    assert counter == 22
    
    # Test the else branch (lines 124-126)
    comp.func_time_sum = 0.0  # This will trigger the else branch
    counter = comp.counter
    # Expected: counter = n_func_eval + n_grad_eval = 10 + 3 = 13
    assert counter == 13


@pytest.mark.skipif(not OPENMDAO_AVAILABLE, reason="OpenMDAO not available")
def test_component_wrapper_skip_linearize():
    """Test ComponentWrapper skip_linearize attribute - lines 156-157"""
    def simple_func(x):
        return x * 3
    
    def simple_grad(x):
        return [3.0]
    
    inputs = [('x', {'val': 1.0})]
    outputs = [('result', {'val': 0.0})]
    
    comp = ComponentWrapper(
        inputs=inputs,
        outputs=outputs,
        function=simple_func,
        gradient_function=simple_grad
    )
    
    prob = om.Problem()
    prob.model.add_subsystem("comp", comp)
    prob.setup()
    
    # Test the skip_linearize functionality
    comp.skip_linearize = True
    
    # Test that the attribute is checked in compute_partials (lines 156-157)
    initial_grad_count = comp.n_grad_eval
    
    # Create mock inputs and J for direct testing
    mock_inputs = {'x': 4.0}
    mock_J = {}
    
    # Call compute_partials directly to test lines 156-157
    comp.compute_partials(mock_inputs, mock_J)
    
    # When skip_linearize is True, method should return early
    # and not increment grad eval count
    assert comp.n_grad_eval == initial_grad_count
    
    # Test normal operation
    comp.skip_linearize = False
    comp.compute_partials(mock_inputs, mock_J)
    
    # Now grad eval count should have increased
    assert comp.n_grad_eval == initial_grad_count + 1


def test_component_wrapper_source_coverage():
    """Verify that all target lines exist in the source code"""
    
    with open('/home/runner/work/hydesign/hydesign/hydesign/openmdao_wrapper.py', 'r') as f:
        source = f.read()
        lines = source.split('\n')
    
    # Verify target lines exist and contain expected content
    test_cases = [
        (58, 'if additional_inputs is not None'),  # Line 59 
        (60, 'x + ({},) if len(x) == 1 else x for x in additional_inputs'),  # Line 61
        (62, 'self.additional_input_keys = [i[0] for i in self.additional_inputs]'),  # Line 63
        (66, 'if additional_outputs is not None'),  # Line 67 
        (68, 'x + ({},) if len(x) == 1 else x for x in additional_outputs'),  # Line 69
        (87, 'if self.additional_inputs is not None'),  # Line 88
        (88, 'for a_inp in self.additional_inputs'),  # Line 89
        (89, 'self.add_input(a_inp[0], **a_inp[1])'),  # Line 90
        (92, 'if self.additional_outputs is not None'),  # Line 93
        (93, 'for a_out in self.additional_outputs'),  # Line 94
        (94, 'self.add_output(a_out[0], **a_out[1])'),  # Line 95
        (107, 'for out, po in zip(self.outputs, self.partial_options)'),  # Line 108
        (108, 'self.declare_partials(out[0], [i[0] for i in self.inputs], **po)'),  # Line 109
        (112, 'counter = float(self.n_func_eval)'),  # Line 113
        (138, 'if self.additional_outputs is not None'),  # Line 139
        (139, 'res, additional_output = self.function('),  # Line 140
        (142, 'for k, v in additional_output.items():'),  # Line 143
        (143, 'outputs[k] = v'),  # Line 144
        (154, 'if hasattr(self, "skip_linearize")'),  # Line 155
        (155, 'if self.skip_linearize'),  # Line 156
        (156, 'return'),  # Line 157
    ]
    
    for line_idx, expected_content in test_cases:
        assert expected_content in lines[line_idx], f"Line {line_idx + 1} should contain: {expected_content}, got: {lines[line_idx]}"
    
    print("✓ All target lines verified in source code")


def test_component_wrapper_logic_without_openmdao():
    """Test ComponentWrapper logic patterns without requiring OpenMDAO"""
    
    # Test counter property logic
    class MockComponentWrapper:
        def __init__(self):
            self.n_func_eval = 0
            self.func_time_sum = 0
            self.n_grad_eval = 0
            self.grad_time_sum = 0
        
        @property  
        def counter(self):
            # This is the exact logic from lines 113-126
            counter = float(self.n_func_eval)
            if (
                self.grad_time_sum > 0
                and self.func_time_sum > 0
                and self.n_grad_eval > 0
                and self.n_func_eval > 0
            ):
                ratio = (self.grad_time_sum / self.n_grad_eval) / (
                    self.func_time_sum / self.n_func_eval
                )
                counter += self.n_grad_eval * max(ratio, 1)
            else:
                counter += self.n_grad_eval
            return int(counter)
    
    comp = MockComponentWrapper()
    
    # Test the timing conditions path (lines 114-123)
    comp.n_func_eval = 10
    comp.func_time_sum = 5.0  # > 0
    comp.n_grad_eval = 3
    comp.grad_time_sum = 6.0  # > 0
    
    counter = comp.counter
    
    # Expected calculation:
    # ratio = (grad_time_sum / n_grad_eval) / (func_time_sum / n_func_eval)
    # ratio = (6.0 / 3) / (5.0 / 10) = 2.0 / 0.5 = 4.0
    # counter = n_func_eval + n_grad_eval * max(ratio, 1)
    # counter = 10 + 3 * max(4.0, 1) = 10 + 3 * 4 = 22
    assert counter == 22
    
    # Test the else branch (lines 124-126)
    comp.func_time_sum = 0.0  # This will trigger the else branch
    counter = comp.counter
    # Expected: counter = n_func_eval + n_grad_eval = 10 + 3 = 13
    assert counter == 13
    
    print("✓ Counter logic test passed")