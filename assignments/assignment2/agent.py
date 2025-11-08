import os
import json
import requests
from dotenv import load_dotenv
from typing import Dict, Any, List
from tools import TOOLS

load_dotenv()

class NumericalAgent:
    """数值计算 Agent"""
    
    def __init__(self, api_key: str = None, model: str = "deepseek-chat"):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量")
            
        self.model = model
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        self.conversation_history = []
        
        # 系统提示词 - 专注于数值计算
        self.system_prompt = """你是一个专业的数值计算助手，专门处理三种数值计算问题：

可用工具：
1. numerical_derivative - 数值求导
  参数: function(函数表达式，使用'exp(x)'表示e^x, 'x**2'表示x²), point(求导点), variable(变量，默认x)

2. numerical_integration - 数值积分  
  参数: function(函数表达式，使用'exp(x)'表示e^x, 'x**2'表示x²), lower(下限), upper(上限), variable(变量)

3. solve_equation_numerical - 方程数值求解
  参数: equation(方程), variable(变量), initial_guess(可选，初始猜测)

重要提示：对于自然指数函数e^x，请使用'exp(x)'而不是'e^x'

使用指南：
- 仔细分析用户问题，选择正确的工具
- 确保使用正确的数学表达式格式
- 对于方程求解，尽量让用户提供初始猜测值

输出格式：
{
    "thought": "分析问题和选择工具的理由",
    "tool_calls": [
        {
            "tool_name": "工具名称",
            "parameters": {
                "参数1": "值1",
                "参数2": "值2"
            }
        }
    ],
    "final_answer": "如果没有工具调用，直接回答"
}

示例：
用户：计算e^x在x=1处的导数
{
    "thought": "用户要求计算e^x在x=1处的导数，需要使用exp(x)来表示e^x函数",
    "tool_calls": [
        {
            "tool_name": "numerical_derivative",
            "parameters": {
                "function": "exp(x)",
                "point": 1,
                "variable": "x"
            }
        }
    ],
    "final_answer": ""
}

用户：计算e^x从0到1的积分
{
    "thought": "用户要求计算e^x在[0,1]区间的积分，需要使用exp(x)来表示e^x函数",
    "tool_calls": [
        {
            "tool_name": "numerical_integration",
            "parameters": {
                "function": "exp(x)",
                "lower": 0,
                "upper": 1,
                "variable": "x"
            }
        }
    ],
    "final_answer": ""
}"""
    
    def call_deepseek_api(self, messages: List[Dict]) -> Dict[str, Any]:
        """调用 DeepSeek API"""
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 2048
        }
        
        try:
            response = requests.post(self.base_url, headers=self.headers, json=data, timeout=60)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": f"API调用失败: {str(e)}"}
    
    def execute_tool(self, tool_name: str, parameters: Dict) -> Dict[str, Any]:
        """执行工具调用"""
        if tool_name not in TOOLS:
            return {"error": f"未知工具: {tool_name}"}
        
        try:
            tool_func = TOOLS[tool_name]["function"]
            return tool_func(**parameters)
        except Exception as e:
            return {"error": f"工具执行错误: {str(e)}"}
    
    def parse_agent_response(self, response_text: str) -> Dict[str, Any]:
        """解析 Agent 的响应"""
        try:
            cleaned_text = response_text.strip()
            # 处理可能的代码块标记
            if cleaned_text.startswith('```json'):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.endswith('```'):
                cleaned_text = cleaned_text[:-3]
            
            return json.loads(cleaned_text)
        except json.JSONDecodeError:
            return {
                "thought": "响应解析失败",
                "tool_calls": [],
                "final_answer": response_text
            }
    
    def format_tool_results(self, tool_results: List[Dict]) -> str:
        """格式化工具调用结果，便于 Agent 理解"""
        formatted_results = []
        
        for result in tool_results:
            tool_name = result["tool"]
            parameters = result["parameters"]
            output = result["result"]
            
            if output.get("success"):
                if tool_name == "numerical_derivative":
                    formatted_results.append(
                        f"求导结果: {output['explanation']}\n"
                        f"函数: {output['function']}\n"
                        f"求导点: {output['point']}\n"
                        f"导数值: {output['derivative_value']:.6f}\n"
                        f"方法: {output['method']}"
                    )
                elif tool_name == "numerical_integration":
                    formatted_results.append(
                        f"积分结果: {output['explanation']}\n"
                        f"函数: {output['function']}\n"
                        f"积分区间: [{output['interval'][0]}, {output['interval'][1]}]\n"
                        f"积分值: {output['integral_value']:.6f}\n"
                        f"误差估计: {output['estimated_error']:.2e}"
                    )
                elif tool_name == "solve_equation_numerical":
                    if 'solution' in output:
                        formatted_results.append(
                            f"方程求解结果: {output['explanation']}\n"
                            f"方程: {output['equation']}\n"
                            f"解: {output['solution']:.6f}\n"
                            f"方法: {output['method']}"
                        )
                    else:
                        formatted_results.append(
                            f"方程求解结果: {output['explanation']}\n"
                            f"方程: {output['equation']}\n"
                            f"所有解: {output['solutions']}\n"
                            f"方法: {output['method']}"
                        )
            else:
                formatted_results.append(f"工具 {tool_name} 执行失败: {output.get('error', '未知错误')}")
        
        return "\n\n".join(formatted_results)
    
    def process_query(self, user_query: str) -> str:
        """处理用户查询"""
        # 构建消息
        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.conversation_history,
            {"role": "user", "content": user_query}
        ]
        
        # 调用 API
        api_response = self.call_deepseek_api(messages)
        
        if "error" in api_response:
            return f"❌ API错误: {api_response['error']}"
        
        # 获取 Agent 响应
        agent_response_text = api_response["choices"][0]["message"]["content"]
        print(f"🔍 Agent原始响应: {agent_response_text}")  # 调试信息
        
        agent_response = self.parse_agent_response(agent_response_text)
        
        # 记录对话
        self.conversation_history.append({"role": "user", "content": user_query})
        
        # 处理工具调用
        if agent_response.get("tool_calls"):
            print(f"🔧 检测到工具调用: {agent_response['tool_calls']}")  # 调试信息
            
            tool_results = []
            for tool_call in agent_response["tool_calls"]:
                tool_name = tool_call["tool_name"]
                parameters = tool_call["parameters"]
                
                print(f"🔧 执行工具: {tool_name}, 参数: {parameters}")  # 调试信息
                
                # 执行工具
                result = self.execute_tool(tool_name, parameters)
                tool_results.append({
                    "tool": tool_name,
                    "parameters": parameters,
                    "result": result
                })
            
            # 如果有工具调用结果，进行第二轮处理
            if tool_results:
                print(f"🔧 工具执行结果: {tool_results}")  # 调试信息
                
                # 格式化工具结果
                tool_context = self.format_tool_results(tool_results)
                
                second_round_messages = [
                    {"role": "system", "content": "你是一个数学助手，请基于数值计算结果给用户提供清晰易懂的答案。"},
                    {"role": "user", "content": user_query},
                    {"role": "assistant", "content": agent_response_text},
                    {"role": "user", "content": f"这是数值计算的结果：\n\n{tool_context}\n\n请基于这些计算结果，给用户提供一个完整、清晰的答案，包括计算结果和必要的解释。"}
                ]
                
                final_response = self.call_deepseek_api(second_round_messages)
                if "error" not in final_response:
                    final_answer = final_response["choices"][0]["message"]["content"]
                else:
                    final_answer = f"最终处理错误: {final_response['error']}"
            else:
                final_answer = agent_response.get("final_answer", agent_response_text)
        else:
            # 直接使用最终答案
            final_answer = agent_response.get("final_answer", agent_response_text)
        
        # 记录助手回复
        self.conversation_history.append({"role": "assistant", "content": final_answer})
        
        return final_answer
    
    def clear_history(self):
        """清空对话历史"""
        self.conversation_history.clear()
        print("对话历史已清空！")

def main():
    """主函数"""
    agent = NumericalAgent()
    
    print("=" * 50)
    print("💻 DeepSeek 数值计算 Agent")
    print("=" * 50)
    print("支持的功能：")
    print("  • 数值求导")
    print("  • 数值积分") 
    print("  • 方程数值求解")
    print("输入 'clear' 清空历史，'quit' 退出")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\n❓ 输入问题: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 再见！")
                break
            elif user_input.lower() == 'clear':
                agent.clear_history()
                continue
            elif user_input == '':
                continue
            
            print("🤔 计算中...")
            response = agent.process_query(user_input)
            print(f"🔬 计算结果: {response}")
            
        except KeyboardInterrupt:
            print("\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 错误: {str(e)}")

if __name__ == "__main__":
    main()