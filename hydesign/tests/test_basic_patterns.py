# -*- coding: utf-8 -*-
"""
Test module for basic Python functionality and simple utilities

Created for increasing test coverage - focused on basic Python patterns
"""
import os
import tempfile
import json
import pytest


def test_file_operations():
    """Test basic file operations that might be used in the codebase."""
    # Test temporary file creation and writing
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        test_data = {'test': 'data', 'numbers': [1, 2, 3]}
        json.dump(test_data, f)
        temp_file = f.name
    
    try:
        # Test reading the file back
        with open(temp_file, 'r') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == test_data
        assert loaded_data['test'] == 'data'
        assert loaded_data['numbers'] == [1, 2, 3]
        
    finally:
        os.unlink(temp_file)


def test_directory_operations():
    """Test directory operations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test directory exists
        assert os.path.exists(temp_dir)
        assert os.path.isdir(temp_dir)
        
        # Test creating subdirectory
        sub_dir = os.path.join(temp_dir, 'subdir')
        os.makedirs(sub_dir)
        assert os.path.exists(sub_dir)
        
        # Test creating file in subdirectory
        test_file = os.path.join(sub_dir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write('test content')
        
        assert os.path.exists(test_file)
        
        # Test reading file content
        with open(test_file, 'r') as f:
            content = f.read()
        
        assert content == 'test content'


def test_string_operations():
    """Test string operations commonly used in configuration parsing."""
    # Test string formatting
    template = "Location: {lat}, {lon} at {alt}m elevation"
    result = template.format(lat=55.0, lon=12.0, alt=100)
    expected = "Location: 55.0, 12.0 at 100m elevation"
    assert result == expected
    
    # Test string splitting and joining
    path_parts = ['hydesign', 'tests', 'test_file.py']
    full_path = os.path.join(*path_parts)
    assert 'hydesign' in full_path
    assert 'test_file.py' in full_path
    
    # Test string cleaning operations
    dirty_string = "  test_value  \n"
    clean_string = dirty_string.strip()
    assert clean_string == "test_value"


def test_list_operations():
    """Test list operations that might be used in data processing."""
    # Test list creation and manipulation
    data = [1, 2, 3, 4, 5]
    
    # Test list slicing
    first_half = data[:3]
    assert first_half == [1, 2, 3]
    
    second_half = data[3:]
    assert second_half == [4, 5]
    
    # Test list comprehension
    squared = [x**2 for x in data]
    assert squared == [1, 4, 9, 16, 25]
    
    # Test filtering
    even_numbers = [x for x in data if x % 2 == 0]
    assert even_numbers == [2, 4]


def test_dict_operations():
    """Test dictionary operations for configuration handling."""
    # Test dict creation and access
    config = {
        'latitude': 55.0,
        'longitude': 12.0,
        'altitude': 100.0,
        'settings': {
            'verbose': True,
            'max_iterations': 1000
        }
    }
    
    assert config['latitude'] == 55.0
    assert config['settings']['verbose'] is True
    
    # Test dict updating
    updates = {'latitude': 56.0, 'new_param': 'test'}
    config.update(updates)
    
    assert config['latitude'] == 56.0
    assert config['new_param'] == 'test'
    
    # Test dict get with defaults
    value = config.get('missing_key', 'default_value')
    assert value == 'default_value'
    
    existing_value = config.get('longitude', 'default')
    assert existing_value == 12.0


def test_exception_handling():
    """Test exception handling patterns."""
    # Test ValueError raising and catching
    def validate_positive(value):
        if value <= 0:
            raise ValueError("Value must be positive")
        return value
    
    # Test valid case
    result = validate_positive(5)
    assert result == 5
    
    # Test exception case
    with pytest.raises(ValueError) as excinfo:
        validate_positive(-1)
    
    assert "must be positive" in str(excinfo.value)


def test_function_defaults():
    """Test function default parameter handling."""
    def process_config(required_param, optional_param=None, flag=True):
        result = {'required': required_param, 'flag': flag}
        if optional_param is not None:
            result['optional'] = optional_param
        return result
    
    # Test with minimal parameters
    result1 = process_config('test')
    assert result1 == {'required': 'test', 'flag': True}
    
    # Test with all parameters
    result2 = process_config('test', 'optional_value', False)
    assert result2 == {'required': 'test', 'optional': 'optional_value', 'flag': False}
    
    # Test with keyword arguments
    result3 = process_config('test', flag=False)
    assert result3 == {'required': 'test', 'flag': False}


def test_iteration_patterns():
    """Test iteration patterns commonly used in data processing."""
    # Test enumerate
    items = ['a', 'b', 'c']
    indexed_items = [(i, item) for i, item in enumerate(items)]
    assert indexed_items == [(0, 'a'), (1, 'b'), (2, 'c')]
    
    # Test zip
    keys = ['x', 'y', 'z']
    values = [1, 2, 3]
    pairs = list(zip(keys, values))
    assert pairs == [('x', 1), ('y', 2), ('z', 3)]
    
    # Test dict creation from zip
    result_dict = dict(zip(keys, values))
    assert result_dict == {'x': 1, 'y': 2, 'z': 3}


def test_type_checking():
    """Test type checking patterns."""
    def process_input(value):
        if isinstance(value, str):
            return value.upper()
        elif isinstance(value, (int, float)):
            return value * 2
        elif isinstance(value, list):
            return len(value)
        else:
            return None
    
    assert process_input('test') == 'TEST'
    assert process_input(5) == 10
    assert process_input(2.5) == 5.0
    assert process_input([1, 2, 3]) == 3
    assert process_input({}) is None


def test_class_basics():
    """Test basic class patterns."""
    class SimpleConfig:
        def __init__(self, name, value=None):
            self.name = name
            self.value = value
            self._private = 'hidden'
        
        def get_info(self):
            return f"{self.name}: {self.value}"
        
        @property
        def is_valid(self):
            return self.value is not None
    
    # Test initialization
    config = SimpleConfig('test_config', 42)
    assert config.name == 'test_config'
    assert config.value == 42
    
    # Test method
    info = config.get_info()
    assert info == 'test_config: 42'
    
    # Test property
    assert config.is_valid is True
    
    # Test with None value
    empty_config = SimpleConfig('empty')
    assert empty_config.is_valid is False