"""
T1 · 不用框架,手搓一个 agent loop

目的:亲眼看清 agent 到底是什么东西。

剧透:它本质就是 ——「一个 while 循环 + 一张工具映射表」。
写完这份代码再回头看 T0 的 create_agent,你就能说清它到底替你做了什么。

跑法:
    cd /d/VibeCoding/langgraph-lab
    ./.venv/Scripts/python 02-raw-agent-loop/agent_loop.py

注意:这里不用 LangChain,直接用 openai SDK 打智谱的兼容端点。
"""

import json
import os
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("MODEL_API_KEY"),
    base_url=os.getenv("MODEL_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
)
MODEL = os.getenv("MODEL_NAME", "glm-4.7-flash")


# ========== 1. 工具:就是普通的 Python 函数 ==========


def get_current_time() -> str:
    """返回当前本地日期时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def add(a: float, b: float) -> float:
    """两个数字相加。"""
    return a + b


def count_chars(text: str) -> int:
    """统计一段文本的字数(按字符数,去掉首尾空格)。"""
    return len(text.strip())


# ========== 2. 工具清单:手写 JSON Schema ==========
# 👀 对比 T0:框架里这份 schema 是从「函数签名 + docstring」自动生成的
#    这里你得一个一个字段手敲 —— 这就是框架替你省的第一件事

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "返回当前本地日期和时间",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "把两个数字相加,返回结果",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "第一个加数"},
                    "b": {"type": "number", "description": "第二个加数"},
                },
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_chars",
            "description": "统计一段文本的字符数",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "要统计的文本"},
                },
                "required": ["text"],
            },
        },
    },
]

# ========== 3. 工具名 → 函数 的映射表 ==========
# 👀 这是框架替你省的第二件事:它自动维护这张表。手写的话忘了同步就会报错。

TOOL_MAP = {
    "get_current_time": get_current_time,
    "add": add,
    "count_chars": count_chars,
}


# ========== 4. 这就是 agent loop 本体 ==========


def run_agent(question: str, max_turns: int = 10) -> str:
    messages = [
        {
            "role": "system",
            "content": "你是一个简洁的助手。涉及时间或计算时必须调用工具,不要凭记忆回答。",
        },
        {"role": "user", "content": question},
    ]

    for turn in range(1, max_turns + 1):
        print(f"\n---------- 第 {turn} 轮 ----------")

        # ① 把当前全部消息发给模型
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            temperature=0,
        )
        msg = resp.choices[0].message

        # ⚠️ 关键:模型的这条回复也必须塞回历史
        #    否则下一轮模型不知道自己刚才说过什么、调过什么
        messages.append(msg)

        # ② 分支:它是要调工具,还是已经能给最终答案?
        if not msg.tool_calls:
            print("[终止条件达成] 模型没再要求调工具 → 循环结束")
            return msg.content or ""

        # ③ 要调工具 → 逐个执行,把结果作为 tool 消息塞回去
        #    ⚠️ 注意:这里是 for 循环「逐个跑」,并不是真并发。
        #       模型在同一轮里一次发起多个调用 ≠ 它们同时执行。
        #       实测:3 个各睡 1 秒的工具,for 循环 3.00s,ThreadPoolExecutor 1.00s。
        #       详见 README「并行工具调用:拆成三层才看得清」
        for call in msg.tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments or "{}")
            print(f"[模型要求] {name}({args})")

            fn = TOOL_MAP.get(name)
            result = fn(**args) if fn else f"错误:不存在名为 {name} 的工具"
            print(f"[工具返回] {result}")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,  # 必须配对,模型靠它知道这是哪次调用的结果
                    "content": str(result),
                }
            )

    return "⚠️ 达到最大轮数仍未收敛,已强制停止(防止死循环烧钱)"


if __name__ == "__main__":
    question = "现在几点?帮我算 1234 + 5678,顺便数一下「人工智能」这几个字有几个字符"
    print(f">>> 提问:{question}")

    answer = run_agent(question)
    print(f"\n===== 最终回答 =====\n{answer}")
