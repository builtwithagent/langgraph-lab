"""
T1 延伸实验:「并行工具调用」到底并行在哪?

把这件事拆成三层,分别验证:

  第一层 决策层 —— 模型在一条消息里给了几个 tool_call?
                   → 同一个问题跑 3 次,统计每轮的调用数(不需要人干预,看模型自己)
  第二层 执行层 —— 给定同一批 tool_call,你的代码是真并发还是排队跑?
                   → 手动构造 3 个 call,对比串行 / 并发耗时(隔离掉模型,只看代码)
  第三层 依赖约束 —— 后一个工具要用前一个的输出,还能并行吗?
                   → 制造数据依赖,看模型会不会被拆成两轮

跑法:
    cd /d/VibeCoding/langgraph-lab
    ./.venv/Scripts/python 02-raw-agent-loop/parallel_experiment.py
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

client = OpenAI(
    api_key=os.getenv("MODEL_API_KEY"),
    base_url=os.getenv("MODEL_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
)
MODEL = os.getenv("MODEL_NAME", "glm-4.7-flash")


# ========== 工具 A/B/C:三个互不依赖的「慢工具」 ==========
# 各睡 1 秒 —— 用耗时就能反推它们是并发跑的还是排队跑的


def slow_task_a() -> str:
    """执行任务 A,大约需要 1 秒。"""
    time.sleep(1)
    return "A 完成"


def slow_task_b() -> str:
    """执行任务 B,大约需要 1 秒。"""
    time.sleep(1)
    return "B 完成"


def slow_task_c() -> str:
    """执行任务 C,大约需要 1 秒。"""
    time.sleep(1)
    return "C 完成"


# ========== 工具 D/E:一对「有数据依赖」的工具 ==========
# pack_clothes 的 temp 只能来自 get_weather 的返回值 —— 结构上就不可能并行


def get_weather(city: str) -> dict:
    """查询指定城市当前天气。"""
    return {"city": city, "weather": "晴", "temp": 18}


def pack_clothes(temp: int) -> dict:
    """根据气温推荐穿什么。"""
    if temp >= 25:
        advice = "短袖"
    elif temp >= 15:
        advice = "薄外套"
    else:
        advice = "厚外套"
    return {"temp": temp, "advice": advice}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "slow_task_a",
            "description": "执行任务 A",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "slow_task_b",
            "description": "执行任务 B",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "slow_task_c",
            "description": "执行任务 C",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询城市当前天气",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "城市名"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pack_clothes",
            "description": "根据气温推荐穿什么",
            "parameters": {
                "type": "object",
                "properties": {
                    "temp": {"type": "integer", "description": "气温(摄氏度)"},
                },
                "required": ["temp"],
            },
        },
    },
]

TOOL_MAP = {
    "slow_task_a": slow_task_a,
    "slow_task_b": slow_task_b,
    "slow_task_c": slow_task_c,
    "get_weather": get_weather,
    "pack_clothes": pack_clothes,
}

SYSTEM = "你是一个简洁的助手。需要执行任务或查询数据时必须调用工具。"


def call_tool(call):
    """执行一个 tool_call,返回 (call, 结果)"""
    fn = TOOL_MAP.get(call.function.name)
    args = json.loads(call.function.arguments or "{}")
    result = fn(**args) if fn else f"错误:没有工具 {call.function.name}"
    return call, result


def to_tool_message(call, result):
    content = (
        json.dumps(result, ensure_ascii=False) if isinstance(result, dict) else str(result)
    )
    return {"role": "tool", "tool_call_id": call.id, "content": content}


def ask(messages, retries: int = 3):
    """带指数退避的模型调用(免费模型高峰期限流)"""
    for i in range(retries):
        try:
            return client.chat.completions.create(
                model=MODEL, messages=messages, tools=TOOLS, temperature=0
            )
        except Exception as e:
            if i == retries - 1:
                raise
            wait = 2**i
            print(f"    [重试] {type(e).__name__},{wait}s 后再试")
            time.sleep(wait)


def run_agent(question: str, concurrent: bool, verbose: bool = True, max_turns: int = 10) -> dict:
    """手搓 agent loop。concurrent 决定同一轮里的多个工具怎么执行。"""
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]
    turn_calls = []  # 每轮模型要了几个工具

    for turn in range(1, max_turns + 1):
        resp = ask(messages)
        msg = resp.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            if verbose:
                print(f"    第 {turn} 轮:无工具调用 → 终止")
            return {"answer": msg.content or "", "turn_calls": turn_calls}

        n = len(msg.tool_calls)
        turn_calls.append(n)
        names = [c.function.name for c in msg.tool_calls]
        if verbose:
            print(f"    第 {turn} 轮:要求调用 {n} 个 → {names}")

        t = time.perf_counter()
        if concurrent:
            with ThreadPoolExecutor(max_workers=n) as pool:
                results = list(pool.map(call_tool, msg.tool_calls))
        else:
            results = [call_tool(c) for c in msg.tool_calls]
        elapsed = time.perf_counter() - t

        if verbose:
            print(f"    └ {n} 个工具实际执行耗时 {elapsed:.2f}s")

        for call, result in results:
            messages.append(to_tool_message(call, result))

    return {"answer": "⚠️ 达到最大轮数仍未收敛", "turn_calls": turn_calls}


def fake_call(idx: int, name: str):
    """手动造一个 tool_call 对象 —— 绕过模型,单独测执行层"""
    return SimpleNamespace(
        id=f"call_{idx}", function=SimpleNamespace(name=name, arguments="{}")
    )


if __name__ == "__main__":
    SEP = "=" * 62

    # ---------------------------------------------------------------
    print(SEP)
    print("第二层 · 执行层:同一批 tool_call,串行 vs 并发")
    print("(完全绕过模型 —— 手动构造 3 个调用,直接测代码)")
    print(SEP)

    calls = [fake_call(1, "slow_task_a"), fake_call(2, "slow_task_b"), fake_call(3, "slow_task_c")]

    t = time.perf_counter()
    for c in calls:
        call_tool(c)
    serial = time.perf_counter() - t

    t = time.perf_counter()
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(call_tool, calls))
    parallel = time.perf_counter() - t

    print(f"  3 个各睡 1 秒的工具:")
    print(f"    for 循环逐个跑(=.venv/agent_loop.py 的写法)  {serial:.2f}s")
    print(f"    ThreadPoolExecutor 并发                       {parallel:.2f}s")
    print(f"  → 结论:同轮多调用只是「一次发起」,真并发得自己写。差 {serial - parallel:.1f}s")

    # ---------------------------------------------------------------
    print()
    print(SEP)
    print("第一层 · 决策层:同一问题跑 3 次,模型每次要几个工具?")
    print(SEP)

    q1 = "请依次执行任务 A、任务 B、任务 C,三个都要执行,然后告诉我结果。"
    patterns = []
    for i in range(3):
        print(f"\n  --- 第 {i + 1} 次 ---")
        r = run_agent(q1, concurrent=False)
        patterns.append(r["turn_calls"])
        print(f"    每轮调用数:{r['turn_calls']}")

    print(f"\n  三次的模式:{patterns}")
    if len({tuple(p) for p in patterns}) > 1:
        print("  → 结论:prompt 完全一样,temperature=0,模型给出的并行结构却不同。")
        print("     「一轮里调几个工具」是模型的决定,不是你的代码决定的。")
    else:
        print("  → 三次一致。但这份稳定只是这次的表现,不能当保证 —— 换模型/换问法都可能变。")

    # ---------------------------------------------------------------
    print()
    print(SEP)
    print("第三层 · 依赖约束:后一个工具要用前一个的输出")
    print(SEP)
    print("\n  问题:北京现在气温多少?根据温度推荐一下穿什么。")
    r = run_agent(
        "北京现在气温多少?根据温度推荐一下穿什么。",
        concurrent=True,
    )
    print(f"  每轮调用数:{r['turn_calls']}")
    print("  → 两轮、每轮 1 个。pack_clothes 的入参 temp 只能来自上一步,"
          "结构上并行不了。")
    print(f"\n  最终回答:{r['answer']}")
