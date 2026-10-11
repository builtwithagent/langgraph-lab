"""
探测:parallel_tool_calls 这个参数到底管的是什么?

背景:OpenAI 在 2024-06 加了 parallel_tool_calls 参数(默认 true),
     官方原话:「set parallel_tool_calls to false, which ensures exactly zero or one tool is called」
     —— 注意它管的是 **模型一次返回几个调用**,不是「你的代码怎么执行」。

本脚本验证智谱的 OpenAI 兼容端点认不认这个参数。

跑法:
    cd /d/VibeCoding/langgraph-lab
    ./.venv/Scripts/python 02-raw-agent-loop/parallel_tool_calls_probe.py
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

client = OpenAI(
    api_key=os.getenv("MODEL_API_KEY"),
    base_url=os.getenv("MODEL_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
)
MODEL = os.getenv("MODEL_NAME", "glm-4.7-flash")


def task_a() -> str:
    """执行任务 A。"""
    return "A 完成"


def task_b() -> str:
    """执行任务 B。"""
    return "B 完成"


def task_c() -> str:
    """执行任务 C。"""
    return "C 完成"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": f"task_{n}",
            "description": f"执行任务 {n.upper()}",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }
    for n in ("a", "b", "c")
]

TOOL_MAP = {"task_a": task_a, "task_b": task_b, "task_c": task_c}
SYSTEM = "你是一个简洁的助手。需要执行任务时必须调用工具。"
QUESTION = "请依次执行任务 A、任务 B、任务 C,三个都要执行,然后告诉我结果。"

# 免费模型并发限制紧 —— 给所有请求加最小间隔,避免整轮被 429 打穿
MIN_INTERVAL = 7.0
_last_call = [0.0]


def throttled_create(kwargs, retries: int = 4):
    """带节流 + 退避的模型调用。返回 None 表示彻底失败。"""
    gap = time.time() - _last_call[0]
    if gap < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - gap)

    for i in range(retries):
        try:
            resp = client.chat.completions.create(**kwargs)
            _last_call[0] = time.time()
            return resp
        except Exception as e:
            body = str(e)
            if "429" not in body:
                # 非限流错误(比如参数不支持)直接暴露出来
                print(f"    [非限流错误] {body[:200]}")
                return None
            if i == retries - 1:
                return None
            wait = 5 * (i + 1)
            print(f"    [限流] 等 {wait}s...")
            time.sleep(wait)
    return None


def run_once(parallel_flag, retries: int = 4):
    """跑一次完整的 agent loop,返回(每轮调用数, 是否报错)"""
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": QUESTION},
    ]
    turn_calls = []

    for turn in range(1, 8):
        kwargs = dict(model=MODEL, messages=messages, tools=TOOLS, temperature=0)
        if parallel_flag is not None:
            kwargs["parallel_tool_calls"] = parallel_flag

        resp = throttled_create(kwargs)
        if resp is None:
            return turn_calls or None, "连续限流失败"

        msg = resp.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            return turn_calls, None

        turn_calls.append(len(msg.tool_calls))
        for call in msg.tool_calls:
            fn = TOOL_MAP.get(call.function.name)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": fn() if fn else "无此工具",
                }
            )

    return turn_calls, "未收敛"


if __name__ == "__main__":
    print("=" * 64)
    print("探测 parallel_tool_calls:它管的是「模型一次给几个调用」")
    print("=" * 64)
    print(f"模型:{MODEL}")
    print(f"问题:{QUESTION}\n")

    configs = [
        ("不传这个参数", None),
        ("parallel_tool_calls=True", True),
        ("parallel_tool_calls=False", False),
    ]

    for label, flag in configs:
        print(f"--- {label} ---")
        for i in range(2):
            calls, err = run_once(flag)
            if err:
                print(f"  第 {i + 1} 次:❌ {err}")
                continue
            print(f"  第 {i + 1} 次:每轮调用数 = {calls}   共 {sum(calls)} 个工具")
        print()

    print("=" * 64)
    print("读法:")
    print("  False 那一组如果稳定出现 [1,1,1] → 参数生效,模型被强制一次只给一个")
    print("  如果三组表现没差别 → 该端点忽略了这个参数(开源/兼容端点的常见情况)")
    print("=" * 64)
