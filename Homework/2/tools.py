import numpy as np
from scipy import integrate, optimize
from typing import Dict, Any, List, Optional
import sympy as sp

class NumericalTools:
    """数值计算工具类"""
    
    @staticmethod
    def _parse_expression(expression: str, variable: str = 'x'):
        """解析数学表达式，支持自然常数e等符号"""
        try:
            x = sp.symbols(variable)
            # 定义常用的数学函数和常数
            local_dict = {
                'e': sp.E,
                'pi': sp.pi,
                'exp': sp.exp,
                'log': sp.log,
                'sin': sp.sin,
                'cos': sp.cos,
                'tan': sp.tan,
                'sqrt': sp.sqrt
            }
            
            # 替换常见的数学表示法
            cleaned_expression = expression.replace('^', '**')
            
            # 使用 sympify 解析表达式
            expr = sp.sympify(cleaned_expression, locals=local_dict)
            return expr, x
        except Exception as e:
            raise ValueError(f"表达式解析错误: {str(e)}")
    
    @staticmethod
    def numerical_derivative(function: str, point: float, variable: str = 'x', h: float = 1e-5) -> Dict[str, Any]:
        """
        数值求导 - 使用中心差分法
        """
        try:
            # 解析函数表达式
            expr, x = NumericalTools._parse_expression(function, variable)
            f = sp.lambdify(x, expr, 'numpy')
            
            # 中心差分法计算导数
            derivative = (f(point + h) - f(point - h)) / (2 * h)
            
            return {
                "success": True,
                "function": function,
                "point": point,
                "derivative_value": float(derivative),
                "method": "中心差分法",
                "step_size": h,
                "explanation": f"函数 {function} 在 {variable}={point} 处的数值导数为 {derivative:.6f}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"数值求导错误: {str(e)}",
                "suggestion": "请使用有效的数学表达式，如 'exp(x)', 'x**2', 'sin(x)'"
            }
    
    @staticmethod
    def numerical_integration(function: str, lower: float, upper: float, variable: str = 'x') -> Dict[str, Any]:
        """
        数值积分 - 使用 scipy 的 quad 方法
        """
        try:
            # 解析函数表达式
            expr, x = NumericalTools._parse_expression(function, variable)
            f = sp.lambdify(x, expr, 'numpy')
            
            # 数值积分
            result, error = integrate.quad(f, lower, upper)
            
            return {
                "success": True,
                "function": function,
                "interval": [lower, upper],
                "integral_value": float(result),
                "estimated_error": float(error),
                "method": "自适应积分法",
                "explanation": f"∫[{lower}, {upper}] {function} d{variable} = {result:.6f} (误差估计: {error:.2e})"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"数值积分错误: {str(e)}",
                "suggestion": "请使用有效的数学表达式，如 'exp(x)', 'x**2', 'sin(x)'"
            }
    
    @staticmethod
    def solve_equation_numerical(equation: str, variable: str = 'x', initial_guess: Optional[float] = None, 
                               method: str = 'root') -> Dict[str, Any]:
        """
        方程数值求解
        """
        try:
            x = sp.symbols(variable)
            
            # 处理方程格式
            if '=' in equation:
                left, right = equation.split('=')
                left_expr, _ = NumericalTools._parse_expression(left, variable)
                right_expr, _ = NumericalTools._parse_expression(right, variable)
                expr = left_expr - right_expr
            else:
                expr, _ = NumericalTools._parse_expression(equation, variable)
            
            f = sp.lambdify(x, expr, 'numpy')
            
            if method == 'root' and initial_guess is not None:
                # 使用 scipy.optimize.root
                solution = optimize.root(f, initial_guess)
                if solution.success:
                    root = float(solution.x[0])
                    return {
                        "success": True,
                        "equation": equation,
                        "solution": root,
                        "initial_guess": initial_guess,
                        "method": "root",
                        "explanation": f"方程 {equation} 的数值解为 {variable} = {root:.6f} (初始猜测: {initial_guess})"
                    }
                else:
                    return {
                        "success": False,
                        "error": f"求解失败: {solution.message}"
                    }
                    
            elif method == 'fsolve' and initial_guess is not None:
                # 使用 scipy.optimize.fsolve
                root = optimize.fsolve(f, initial_guess)[0]
                return {
                    "success": True,
                    "equation": equation,
                    "solution": float(root),
                    "initial_guess": initial_guess,
                    "method": "fsolve",
                    "explanation": f"方程 {equation} 的数值解为 {variable} = {root:.6f} (初始猜测: {initial_guess})"
                }
                    
            else:
                # 如果没有提供初始猜测，尝试符号求解
                symbolic_solutions = sp.solve(expr, x)
                if symbolic_solutions:
                    numeric_solutions = [float(sol.evalf()) for sol in symbolic_solutions if sol.is_real]
                    return {
                        "success": True,
                        "equation": equation,
                        "solutions": numeric_solutions,
                        "method": "符号求解",
                        "explanation": f"方程 {equation} 的解为: {', '.join([f'{variable} = {sol:.6f}' for sol in numeric_solutions])}"
                    }
                else:
                    return {
                        "success": False,
                        "error": "无法找到解，请提供初始猜测值"
                    }
                    
        except Exception as e:
            return {
                "success": False,
                "error": f"方程求解错误: {str(e)}"
            }

# 工具调用映射表
TOOLS = {
    "numerical_derivative": {
        "function": NumericalTools.numerical_derivative,
        "description": "数值求导 - 计算函数在特定点的导数值",
        "parameters": {
            "function": "函数表达式，如 'exp(x)', 'x**2', 'sin(x)'",
            "point": "求导点的数值",
            "variable": "变量名，默认为 'x'",
            "h": "可选，差分步长，默认为 1e-5"
        }
    },
    "numerical_integration": {
        "function": NumericalTools.numerical_integration,
        "description": "数值积分 - 计算函数在区间上的定积分",
        "parameters": {
            "function": "函数表达式，如 'exp(x)', 'x**2', 'sin(x)'",
            "lower": "积分下限",
            "upper": "积分上限", 
            "variable": "变量名，默认为 'x'"
        }
    },
    "solve_equation_numerical": {
        "function": NumericalTools.solve_equation_numerical,
        "description": "方程数值求解 - 求方程的数值解",
        "parameters": {
            "equation": "方程表达式，如 'x**2 - 4 = 0'",
            "variable": "变量名，默认为 'x'",
            "initial_guess": "可选，初始猜测值",
            "method": "可选，求解方法: 'root' 或 'fsolve'"
        }
    }
}