"""
T1 延伸:四类错误 × 四种处理 —— 在同一个手搓 loop 里各写一遍

配套笔记:ai-agent-notes/notes/02-工具调用的并发与安全.md 第六节

  ① 瞬时错误(429 / 超时)    谁修:系统     → 工具层重试,模型根本不知道出过错
  ② 工具执行失败 / 参数错     谁修:模型     → 把错误当 Observation 塞回去,它自己改
  ③ 缺信息 / 需要授权         谁修:人       → 停下来问,拿到答复再继续
  ④ 未预期的 bug             谁修:开发者   → 不捕获,让它抛出来

【这个 demo 的核心思想】
用「异常类型」把四类错误分开。执行器只捕获它明确知道该怎么处理的那两个:

    except NeedHumanInput      → 知道怎么办(问人)
    except ToolExecutionError  → 知道怎么办(还给模型)
    → 其余一律不捕获,让它冒泡(那是第 ④ 类,该开发者看)

这正好对应 PEP 8 的建议:捕获异常时写明具体类型,别用裸 except。

⚠️ 关于 ③ 手搓版的局限
手搓 loop 没有 checkpointer,做不了真正的「暂停 → 几小时后从断点恢复」。
这里用「阻塞式问人 + 记住授权」模拟同样的语义。
LangGraph 里的正规实现是 interrupt(),而它必须配 checkpointer(T4 → T5 的因果关系)。

跑法:
    cd /d/VibeCoding/langgraph-lab
    ./.venv/Scripts/python 02-raw-agent-loop/error_handling_demo.py        # 跑全部
    ./.venv/Scripts/python 02-raw-agent-loop/error_handling_demo.py 1 3    # 只跑场景 1、3
"""

import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

client = OpenAI(
    api_key=os.getenv("MODEL_API_KEY"),
    base_url=os.getenv("MODEL_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
)
MODEL = os.getenv("MODEL_NAME", "glm-4.7-flash")

MIN_INTERVAL = 6.0
_last_call = [0.0]


# ==================== 四类错误:用异常类型区分 ====================
#
# ①②③ 我明确知道该怎么处理,所以给它们各自的类型。
# ④ 不给类型 —— KeyError / TypeError 这些「本来就不该发生」的,不在此列。


class TransientError(Exception):
    """① 瞬时错误:重试就可能好(429、超时、503...)"""


class ToolExecutionError(Exception):
    """② 工具执行失败 / 参数不对:把消息还给模型,它自己能改"""


class NeedHumanInput(Exception):
    """③ 缺信息 / 需要授权:只能问人"""

    def __init__(self, question: str):
        self.question = question
        super().__init__(question)


# ==================== 各场景的工具 ====================

# ---- ① 瞬时错误:一个「前两次必然失败」的接口 ----
_flaky = {"calls": 0}


def fetch_quote(symbol: str) -> str:
    """查询股票最新价。"""
    _flaky["calls"] += 1
    if _flaky["calls"] <= 2:
        raise TransientError(f"上游返回 429 Too Many Requests(第 {_flaky['calls']} 次)")
    return f"{symbol} 最新价 1520.00"


# ---- ② 参数错:对格式挑刺的接口 ----
def get_price(symbol: str) -> str:
    """查询股票最新价。symbol 必须是 6 位数字代码。"""
    if not re.fullmatch(r"\d{6}", symbol or ""):
        raise ToolExecutionError(
            f"股票代码 '{symbol}' 格式不对 —— 必须是 6 位数字,例如 600519。请修正后重新调用。"
        )
    return "1520.00"


# ---- ③ 需要授权:有副作用的接口 ----
_approved = set()


def place_order(symbol: str, shares: int) -> str:
    """提交一个买入订单(有副作用,执行前需要人工授权)。"""
    key = f"{symbol}:{shares}"
    if key not in _approved:
        # 不在这里直接问 —— 抛出去,交给循环统一处理「需要人」这件事
        raise NeedHumanInput(f"即将买入 {symbol} {shares} 股,确认执行吗?")
    return f"已提交订单:{symbol} {shares} 股"


# ---- ④ 未预期的 bug:故意写错的一行 ----
def get_my_position() -> str:
    """查询我的持仓。"""
    account = {"cash": 100000}  # 注意:这里根本没有 "positions" 键
    return f"持仓:{account['positions']}"  # KeyError —— 这就是「不该被处理」的那类


def tool_schema(name: str, desc: str, params: dict | None = None) -> dict:
    """手写 JSON Schema 太啰嗦,demo 里用个小助手生成(参数说明还是逐个手填)"""
    props, required = {}, []
    for pname, (ptype, pdesc) in (params or {}).items():
        props[pname] = {"type": ptype, "description": pdesc}
        required.append(pname)
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


SCENARIOS = {
    "1": {
        "title": "① 瞬时错误 → 系统修(工具层重试,模型无感)",
        "question": "600519 现在什么价?直接查一下。",
        "tools": [tool_schema("fetch_quote", "查询股票最新价", {"symbol": ("string", "股票代码")})],
        "tool_map": {"fetch_quote": fetch_quote},
        "expect": "fetch_quote 内部失败两次、第三次成功;模型全程不知道出过错",
    },
    "2": {
        "title": "② 参数错 → 模型修(错误还给模型,它自己改)",
        "question": "贵州茅台(SH600519)现在什么价?",
        "tools": [tool_schema("get_price", "查询股票最新价", {"symbol": ("string", "股票代码")})],
        "tool_map": {"get_price": get_price},
        "expect": "模型很可能先传 'SH600519',格式错 → 错误回到它手里 → 它改成 600519 重试",
    },
    "3": {
        "title": "③ 需要授权 → 人修(停下来问,拿到答复再继续)",
        "question": "帮我买 600519,100 股。",
        "tools": [
            tool_schema(
                "place_order",
                "提交买入订单",
                {"symbol": ("string", "股票代码"), "shares": ("integer", "股数")},
            )
        ],
        "tool_map": {"place_order": place_order},
        "expect": "抛出「需要人工」→ 循环问人 → 拿到 y → 重跑这一次调用 → 成功",
    },
    "4": {
        "title": "④ 未预期的 bug → 开发者修(不捕获,让它炸)",
        "question": "我现在持仓是什么?",
        "tools": [tool_schema("get_my_position", "查询我的持仓")],
        "tool_map": {"get_my_position": get_my_position},
        "expect": "KeyError 直接冒泡到最外层 —— 前三类都有人接,这一类没人接,这就对了",
    },
}


# ==================== 模型调用(带节流,免费模型限流严重) ====================


def ask(messages, tools, retries: int = 4):
    gap = time.time() - _last_call[0]
    if gap < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - gap)

    for i in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL, messages=messages, tools=tools, temperature=0
            )
            _last_call[0] = time.time()
            return resp
        except Exception as e:
            if "429" not in str(e):
                raise
            if i == retries - 1:
                raise
            wait = 5 * (i + 1)
            print(f"      [限流] 等 {wait}s...")
            time.sleep(wait)


def call_with_retry(fn, args, max_attempts: int = 3):
    """① 瞬时错误在这一层解决 —— 重试成功的话,模型根本不知道出过错"""
    for attempt in range(1, max_attempts + 1):
        try:
            return fn(**args)
        except TransientError as e:
            if attempt == max_attempts:
                raise
            wait = 1.0 * (2 ** (attempt - 1))
            print(f"      [系统重试] 第 {attempt} 次失败({e}),等 {wait:.0f}s 再试")
            time.sleep(wait)


def ask_human(question: str) -> str:
    """③ 问人。真实项目里这里就是 LangGraph 的 interrupt —— 图会停在原地等人。

    demo 里自动回 y,避免卡在 input() 上;要看真实交互就取消下面的注释。
    """
    print(f"      [需要人工] {question}")
    # answer = input("      >>> 你的决定: ")
    answer = "y"
    print(f"      [人工答复] {answer}(demo 自动应答)")
    return answer


def execute_tool(call, tool_map):
    """四类错误的分流中心 —— 整个 demo 最重要的就是这一段"""
    fn = tool_map[call.function.name]
    args = json.loads(call.function.arguments or "{}")

    try:
        # ① 瞬时错误 → 在这里重试掉,模型看不见
        result = call_with_retry(fn, args)
        return str(result)

    except NeedHumanInput as e:
        # ③ 需要人 → 问,拿到答复后重跑这次调用
        answer = ask_human(e.question)
        if answer.strip().lower() in ("y", "yes", "是", "批准"):
            _approved.add(f"{args.get('symbol')}:{args.get('shares')}")
            return str(fn(**args))
        return "人工拒绝:这笔操作没有执行"

    except ToolExecutionError as e:
        # ② 参数错 / 工具失败 → 原样还给模型,让它下一轮自己改
        print(f"      [还给模型] {e}")
        return f"工具执行失败:{e}"

    # ④ 故意不写 except —— KeyError / TypeError 直接冒泡出去
    #    写个 except Exception 把它们也接住,就等于把 bug 伪装成「工具失败」


# ==================== agent loop ====================


def run_agent(question: str, tools, tool_map, max_turns: int = 8) -> str:
    messages = [
        {"role": "system", "content": "你是一个简洁的助手。涉及行情或交易时必须调用工具,不要凭记忆回答。"},
        {"role": "user", "content": question},
    ]

    for turn in range(1, max_turns + 1):
        resp = ask(messages, tools)
        msg = resp.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            print(f"    第 {turn} 轮:无工具调用 → 终止")
            return msg.content or ""

        names = [c.function.name for c in msg.tool_calls]
        print(f"    第 {turn} 轮:模型要求调用 {names}")

        for call in msg.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            print(f"      [调用] {call.function.name}({args})")
            content = execute_tool(call, tool_map)
            print(f"      [结果] {content}")
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": content}
            )

    return "⚠️ 达到最大轮数仍未收敛"


def run_scenario(key: str):
    sc = SCENARIOS[key]
    print("\n" + "=" * 66)
    print(f"场景 {key}:{sc['title']}")
    print(f"预期:{sc['expect']}")
    print("=" * 66)
    print(f"  提问:{sc['question']}")

    try:
        answer = run_agent(sc["question"], sc["tools"], sc["tool_map"])
        print(f"\n  ✅ 最终回答:{answer}")

    except Exception as e:
        # ④ 未预期的错误跑到这里 —— 注意:我们只是让 demo 能继续跑完,
        #    并没有"处理"它。真实项目里你就该在这里被打断,然后去修代码。
        print(f"\n  ❌ 未预期错误 → 第 ④ 类,由开发者修")
        print(f"     类型:{type(e).__name__}")
        print(f"     信息:{e!r}")
        print(f"     正确处理:看完整 traceback 定位问题,修代码。")
        print(f"     错误做法:在这里 except 住、返回一句「工具调用失败」——")
        print(f"               那样你永远不知道代码有 bug。")
        print("\n  ---- 完整 traceback ----")
        traceback.print_exc()


if __name__ == "__main__":
    keys = sys.argv[1:] or list(SCENARIOS.keys())
    print(f"模型:{MODEL}")
    print(f"要跑的场景:{keys}")
    for k in keys:
        if k not in SCENARIOS:
            print(f"未知场景 {k},跳过")
            continue
        run_scenario(k)

    print("\n" + "=" * 66)
    print("四类错误的处理方式,一句话总结:")
    print("  ① 瞬时  → 重试掉,别让模型知道")
    print("  ② 执行失败 → 还给模型,让它自己改")
    print("  ③ 缺信息 → 停下来问人")
    print("  ④ bug   → 别接,让它炸给开发者看")
    print("=" * 66)
