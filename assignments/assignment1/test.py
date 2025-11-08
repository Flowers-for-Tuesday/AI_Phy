import os
import requests
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError("请在 .env 文件中设置 DEEPSEEK_API_KEY")

class DeepSeekChat:
    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        self.conversation_history = []
    
    def send_message(self, message: str) -> str:
        """发送消息并获取回复"""
        # 添加用户消息到历史
        self.conversation_history.append({"role": "user", "content": message})
        
        # 准备请求数据
        data = {
            "model": self.model,
            "messages": self.conversation_history,
            "stream": False,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(self.base_url, headers=self.headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            assistant_message = result["choices"][0]["message"]["content"]
            
            # 添加助手回复到历史
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            
            return assistant_message
            
        except requests.exceptions.RequestException as e:
            return f"❌ 请求错误: {str(e)}"
        except KeyError:
            return "❌ 解析响应失败"

# 创建聊天实例
chat = DeepSeekChat(api_key=api_key)

print("🤖 DeepSeek，启动！（输入 q 退出）\n")

while True:
    user_input = input("👤 You：")
    if user_input.lower() in {"exit", "quit", "q"}:
        print("👋 Goodbye！")
        break

    response = chat.send_message(user_input)
    print(f"🤖 DeepSeek：{response}\n")