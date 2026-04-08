import ast
import astunparse
import re
from typing import List, Dict, Tuple, Optional
from collections import Counter

class DataQualityFilter:
    """
    Data quality filter for evaluating and filtering the quality of test assertions
    """
    
    def __init__(self):
        # Define simple patterns that typically indicate low-quality data
        self.simple_patterns = [
            # Simple string literal comparisons
            r"assert\s+['\"][^'\"]*['\"]\s*==\s*\w+",
            # Simple numeric literal comparisons
            r"assert\s+\d+\s*==\s*\w+",
            # Simple boolean literal comparisons
            r"assert\s+(True|False)\s*==\s*\w+",
            # Simple None literal comparisons
            r"assert\s+None\s*==\s*\w+",
            # Simple variable name comparisons (both are simple identifiers)
            r"assert\s+\b[a-zA-Z_]\w*\b\s*==\s*\b[a-zA-Z_]\w*\b",
            # Simple repr comparisons (e.g., repr(Is(5)) == '5')
            r"assert\s+repr\([^)]+\)\s*==\s*['\"][^'\"]*['\"]",
            # Simple str comparisons (e.g., str(5) == '5')
            r"assert\s+str\([^)]+\)\s*==\s*['\"][^'\"]*['\"]",
        ]
        
        # Define high-quality patterns that typically indicate meaningful data
        self.quality_patterns = [
            # Function call on the left side
            r"assert\s+\w+\([^)]*\)\s*==\s*",
            # Method call on the left side
            r"assert\s+\w+\.\w+\([^)]*\)\s*==\s*",
            # Attribute access on the left side
            r"assert\s+\w+\.\w+\s*==\s*",
            # Index access on the left side
            r"assert\s+\w+\[[^\]]*\]\s*==\s*",
            # Slice operation on the left side
            r"assert\s+\w+\[[^\]]*:[^\]]*\]\s*==\s*",
            # Mathematical operations on the left side
            r"assert\s+[^=]*[\+\-\*/%][^=]*\s*==\s*",
            # Logical operations on the left side
            r"assert\s+[^=]*(and|or|not)[^=]*\s*==\s*",
            # Comparison operations on the left side
            r"assert\s+[^=]*[<>!=][^=]*\s*==\s*",
        ]
    
    def analyze_assertion_complexity(self, assert_node: ast.Assert) -> Dict[str, any]:
        """
        Analyze the complexity of an assertion statement
        """
        if not isinstance(assert_node.test, ast.Compare):
            return {"complexity_score": 0, "is_quality": False, "reason": "Not a comparison"}
        
        if not assert_node.test.ops or not isinstance(assert_node.test.ops[0], ast.Eq):
            return {"complexity_score": 0, "is_quality": False, "reason": "Not an equality comparison"}
        
        left_side = assert_node.test.left
        right_side = assert_node.test.comparators[0]
        
        # Analyze the left side (the expression being tested)
        left_complexity = self._analyze_expression_complexity(left_side)
        
        # Analyze the right side (the expected value)
        right_complexity = self._analyze_expression_complexity(right_side)
        
        # Calculate total complexity score
        total_complexity = left_complexity + right_complexity
        
        # Determine whether it's high-quality data
        is_quality = self._is_quality_assertion(left_side, right_side, total_complexity)
        
        return {
            "complexity_score": total_complexity,
            "left_complexity": left_complexity,
            "right_complexity": right_complexity,
            "is_quality": is_quality,
            "reason": self._get_quality_reason(left_side, right_side, total_complexity)
        }
    
    def _analyze_expression_complexity(self, node: ast.AST) -> int:
        """
        Analyze the complexity of an expression
        """
        if node is None:
            return 0
        
        complexity = 0
        
        # Basic node types
        if isinstance(node, (ast.Name, ast.Constant, ast.Num, ast.Str)):
            complexity = 1
        elif isinstance(node, ast.Attribute):
            complexity = 2
        elif isinstance(node, ast.Call):
            complexity = 3
            # Recursively analyze arguments
            for arg in node.args:
                complexity += self._analyze_expression_complexity(arg)
        elif isinstance(node, ast.Subscript):
            complexity = 3
            complexity += self._analyze_expression_complexity(node.value)
            complexity += self._analyze_expression_complexity(node.slice)
        elif isinstance(node, ast.BinOp):
            complexity = 2
            complexity += self._analyze_expression_complexity(node.left)
            complexity += self._analyze_expression_complexity(node.right)
        elif isinstance(node, ast.UnaryOp):
            complexity = 2
            complexity += self._analyze_expression_complexity(node.operand)
        elif isinstance(node, ast.Compare):
            complexity = 2
            complexity += self._analyze_expression_complexity(node.left)
            for comparator in node.comparators:
                complexity += self._analyze_expression_complexity(comparator)
        elif isinstance(node, ast.BoolOp):
            complexity = 2
            for value in node.values:
                complexity += self._analyze_expression_complexity(value)
        elif isinstance(node, ast.List):
            complexity = 2
            for elt in node.elts:
                complexity += self._analyze_expression_complexity(elt)
        elif isinstance(node, ast.Tuple):
            complexity = 2
            for elt in node.elts:
                complexity += self._analyze_expression_complexity(elt)
        elif isinstance(node, ast.Dict):
            complexity = 3
            for key, value in zip(node.keys, node.values):
                complexity += self._analyze_expression_complexity(key)
                complexity += self._analyze_expression_complexity(value)
        
        return complexity
    
    def _is_quality_assertion(self, left_side: ast.AST, right_side: ast.AST, total_complexity: int) -> bool:
        """
        Determine whether an assertion is high-quality data
        Focuses on assertion patterns rather than complexity
        """
        # 1. Avoid obvious simple pattern matches
        assert_str = astunparse.unparse(ast.Assert(ast.Compare(left_side, [ast.Eq()], [right_side])))
        if self._matches_simple_patterns(assert_str):
            return False
        
        # 2. Check if it follows meaningful testing patterns
        # High-quality patterns include:
        # - Comparing function call results
        # - Comparing method call results
        # - Comparing attribute accesses
        # - Comparing index/slice operations
        # - Comparing mathematical operation results
        # - Comparing logical operation results
        
        # 3. Check if the left side contains meaningful operations
        left_has_meaningful_operation = self._has_meaningful_operation(left_side)
        
        # 4. Check if the right side is not too simple (to avoid trivial guesses)
        right_is_not_too_simple = self._is_not_too_simple(right_side)
        
        return left_has_meaningful_operation and right_is_not_too_simple
    
    def _has_meaningful_operation(self, node: ast.AST) -> bool:
        """
        Check if the expression contains meaningful operations
        Such operations usually indicate that the test is verifying the result of some computation or processing
        """
        # Directly check current node
        if isinstance(node, ast.Call):
            return True  # Function call
        elif isinstance(node, ast.Attribute):
            return True  # Method call or attribute access
        elif isinstance(node, ast.Subscript):
            return True  # Indexing or slicing
        elif isinstance(node, ast.BinOp):
            return True  # Arithmetic operation
        elif isinstance(node, ast.UnaryOp):
            return True  # Unary operation
        elif isinstance(node, ast.Compare):
            return True  # Comparison operation
        elif isinstance(node, ast.BoolOp):
            return True  # Logical operation
        elif isinstance(node, (ast.List, ast.Tuple, ast.Dict)):
            return True  # Container operations
        
        # Recursively check child nodes
        for child in ast.walk(node):
            if isinstance(child, (ast.Call, ast.Attribute, ast.Subscript, ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp)):
                return True
        
        return False
    
    def _is_not_too_simple(self, node: ast.AST) -> bool:
        """
        Check if the right-hand expression is not too simple
        Mainly filters out overly trivial literals
        """
        if isinstance(node, (ast.Constant, ast.Num, ast.Str)):
            # Check if it's a simple literal
            if isinstance(node, ast.Constant):
                value = node.value
            elif isinstance(node, ast.Num):
                value = node.n
            elif isinstance(node, ast.Str):
                value = node.s
            else:
                value = None
            
            # Filter out overly simple literals
            if isinstance(value, (int, float)):
                # Numbers: filter out very simple ones like 0, 1, -1
                if value in [0, 1, -1, 2, -2]:
                    return False
            elif isinstance(value, str):
                # Strings: filter out empty or very short strings
                if len(value) <= 1:
                    return False
                # Filter out obvious simple strings
                if value.lower() in ['true', 'false', 'none', 'null', 'yes', 'no']:
                    return False
        
        elif isinstance(node, ast.Name):
            # Variable names: filter out overly simple identifiers
            if node.id in ['True', 'False', 'None', 'null', 'yes', 'no']:
                return False
        
        return True
    
    def _matches_simple_patterns(self, assert_str: str) -> bool:
        """
        Check if the assertion string matches any simple patterns
        """
        for pattern in self.simple_patterns:
            if re.search(pattern, assert_str, re.IGNORECASE):
                return True
        return False
    
    def _get_quality_reason(self, left_side: ast.AST, right_side: ast.AST, total_complexity: int) -> str:
        """
        Get the reason for quality evaluation
        """
        assert_str = astunparse.unparse(ast.Assert(ast.Compare(left_side, [ast.Eq()], [right_side])))
        if self._matches_simple_patterns(assert_str):
            return "Matches simple pattern"
        
        if not self._has_meaningful_operation(left_side):
            return "Left side lacks meaningful operations"
        
        if not self._is_not_too_simple(right_side):
            return "Right side too simple"
        
        return "High quality assertion"
    
    def filter_test_function(self, func_node: ast.FunctionDef) -> List[Dict[str, any]]:
        """
        Filter assertions in a test function and return high-quality assertion data
        """
        quality_asserts = []
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.Assert):
                analysis = self.analyze_assertion_complexity(node)
                if analysis["is_quality"]:
                    quality_asserts.append({
                        "assert_node": node,
                        "analysis": analysis
                    })
        
        return quality_asserts

def create_quality_filter():
    """
    Create an instance of the data quality filter
    """
    return DataQualityFilter()