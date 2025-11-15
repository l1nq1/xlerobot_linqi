#!/usr/bin/env python
"""
Simple verification script to check multi-dataset implementation.
This script performs static checks without requiring full dependencies.
"""

import ast
import sys
from pathlib import Path

def check_file_syntax(filepath):
    """Check if a Python file has valid syntax."""
    try:
        with open(filepath, 'r') as f:
            ast.parse(f.read())
        return True, "OK"
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

def check_class_has_method(filepath, class_name, method_name):
    """Check if a class has a specific method or property."""
    with open(filepath, 'r') as f:
        tree = ast.parse(f.read())
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    if item.name == method_name:
                        # Check if it's a property
                        for decorator in item.decorator_list:
                            if isinstance(decorator, ast.Name) and decorator.id == 'property':
                                return True, f"Found @property {method_name}"
                        return True, f"Found method {method_name}"
    return False, f"Method/property {method_name} not found in class {class_name}"

def check_no_raise_not_implemented(filepath, context="MultiLeRobotDataset"):
    """Check that a specific NotImplementedError is not raised."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    if "MultiLeRobotDataset isn't supported" in content and "raise NotImplementedError" in content:
        # Check if it's actually raising
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if "MultiLeRobotDataset isn't supported" in line and "raise NotImplementedError" in line:
                return False, f"Still raises NotImplementedError at line {i+1}"
    return True, "NotImplementedError check passed"

def check_validation_removed(filepath):
    """Check that the validation preventing multi-dataset is removed."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'isinstance(self.dataset.repo_id, list)' in line:
            # Check if the next few lines have raise NotImplementedError
            for j in range(i, min(i+3, len(lines))):
                if 'raise NotImplementedError' in lines[j] and 'LeRobotMultiDataset' in lines[j]:
                    return False, f"Validation still raises NotImplementedError at line {j+1}"
    return True, "Validation check passed"

def main():
    """Run all verification checks."""
    root = Path(__file__).parent
    src = root / "src" / "lerobot"
    
    checks = [
        ("Factory syntax", lambda: check_file_syntax(src / "datasets" / "factory.py")),
        ("Dataset syntax", lambda: check_file_syntax(src / "datasets" / "lerobot_dataset.py")),
        ("Train config syntax", lambda: check_file_syntax(src / "configs" / "train.py")),
        ("MultiLeRobotDataset has meta property", 
         lambda: check_class_has_method(src / "datasets" / "lerobot_dataset.py", 
                                       "MultiLeRobotDataset", "meta")),
        ("Factory doesn't raise NotImplementedError", 
         lambda: check_no_raise_not_implemented(src / "datasets" / "factory.py")),
        ("Train config validation removed", 
         lambda: check_validation_removed(src / "configs" / "train.py")),
    ]
    
    print("=" * 70)
    print("Multi-Dataset Implementation Verification")
    print("=" * 70)
    
    all_passed = True
    for check_name, check_func in checks:
        passed, message = check_func()
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {check_name}")
        if not passed:
            print(f"        {message}")
            all_passed = False
        else:
            print(f"        {message}")
    
    print("=" * 70)
    if all_passed:
        print("✓ All verification checks passed!")
        print("\nMulti-dataset loading is now enabled.")
        print("See MULTI_DATASET_USAGE.md for usage instructions.")
        return 0
    else:
        print("✗ Some verification checks failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())

