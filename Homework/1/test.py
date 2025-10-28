import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("请在 .env 文件中设置 GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

chat = client.chats.create(model="gemini-2.5-flash")

print("🤖 Gemini，启动！（输入 q 退出）\n")

while True:
    user_input = input("👤 You：")
    if user_input.lower() in {"exit", "quit","q"}:
        print("👋 Goodbye！")
        break

    response = chat.send_message(user_input)

    print(f"🤖 Genimi：{response.text}\n")
