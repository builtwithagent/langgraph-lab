"""
T2 · 把 T1 的 agent 装进状态图 —— 看框架到底收走了什么

同一件事,两种写法对照:

    T1 手搓(02-raw-agent-loop/agent_loop.py)
        ~40 行循环体:自己维护 messages、自己写 while、自己判终止、自己执行工具、自己配 tool_call_id

    T2 状态图(本文件)
        只剩「声明结构」:谁是节点、谁连谁、什么时候分支

跑法:
    cd /d/VibeCoding/langgraph-lab
    ./.venv/Scripts/python 03-first-graph/02_agent_as_graph.py
"""

import os
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import RetryPolicy
from pydantic import SecretStr

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

LINE = "=" * 66


# ==================== 1. 工具:用 @tool 装饰器 ====================
# 👀 对比 T1:那边要手敲 JSON Schema 的每个字段;
#    这边写个 docstring 就够了 —— schema 是自动从「函数签名 + docstring」生成的


@tool
def get_current_time() -> str:
    """返回当前本地日期和时间,格式 YYYY-MM-DD HH:MM:SS。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def add(a: float, b: float) -> float:
    """把两个数字相加,返回结果。用于需要精确计算的场合。"""
    return a + b


@tool
def divide(a: float, b: float) -> float:
    """把 a 除以 b。b 不能为 0。"""
    if b == 0:
        # 故意抛异常 —— 用来验证「工具失败会变成 ToolMessage 还给模型」
        raise ZeroDivisionError("除数不能为 0")
    return a / b


TOOLS = [get_current_time, add, divide]


# ==================== 2. 模型 ====================

_api_key = os.getenv("MODEL_API_KEY")
if not _api_key:
    raise RuntimeError("没找到 MODEL_API_KEY —— 请检查 .env 文件")

model = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "glm-4.7-flash"),
    base_url=os.getenv("MODEL_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
    api_key=SecretStr(_api_key),
    temperature=0,
)
model_with_tools = model.bind_tools(TOOLS)


# ==================== 3. 节点 ====================


def call_model(state: MessagesState) -> dict:
    """agent 节点:把消息交给模型,拿回它的回复。

    对比 T1:那边这一步是 `resp = client.chat.completions.create(...)`
            + `messages.append(msg)`,两件事。
    这里只要返回 {"messages": [...]} ——
    **MessagesState 自带 add_messages reducer,框架自己帮你 append。**
    """
    time.sleep(4)  # 给免费模型留点间隔,免得被限流
    return {"messages": [model_with_tools.invoke(state["messages"])]}


# ==================== 4. 组装图 ====================

builder = StateGraph(MessagesState)

# agent 节点挂上重试策略 —— 瞬时错误自动退避重试,不用自己写
# ⚠️ LangGraph 默认不重试,必须显式挂 RetryPolicy 才有
builder.add_node(
    "agent",
    call_model,
    retry_policy=RetryPolicy(max_attempts=3, initial_interval=1.0, backoff_factor=2.0),
)

# ToolNode:自动执行模型要求的工具
#   handle_tool_errors=True → 工具抛异常时,不打断流程,
#   而是把错误包成 ToolMessage(status="error") 还回给模型,让它自己处理
#   (对应笔记里的第 ② 类错误:工具执行失败 → 模型修)
builder.add_node("tools", ToolNode(TOOLS, handle_tool_errors=True))

builder.add_edge(START, "agent")

# ⬇️ tools_condition 就是 T1 里那句 `if not msg.tool_calls` ——
#    框架把它封装成了一个现成的路由函数
builder.add_conditional_edges(
    "agent", tools_condition, {"tools": "tools", "__end__": END}
)
builder.add_edge("tools", "agent")  # 工具跑完 → 回到 agent

graph = builder.compile()


if __name__ == "__main__":
    print(LINE)
    print("图结构")
    print(LINE)
    print("  START → agent ─┬─(要调工具)→ tools → 回到 agent")
    print("                 └─(不用工具)→ END")
    print()
    print("  自动生成的描述:")
    print(graph.get_graph().draw_mermaid())

    questions = [
        "现在几点?顺便算一下 1234 + 5678。",
        "帮我算 10 除以 0 等于几?",
    ]

    for q in questions:
        print()
        print(LINE)
        print(f"提问:{q}")
        print(LINE)

        result = graph.invoke({"messages": [HumanMessage(q)]})

        # 👀 把每条消息按角色打出来 —— 你会看到跟 T1 里一模一样的那套流转
        # 注意:tool_calls 只有 AIMessage 有,HumanMessage / ToolMessage 没有,
        #      所以要用 getattr 兜一下(这正是类型化消息比裸 dict 好的地方)
        for m in result["messages"]:
            kind = type(m).__name__
            calls = getattr(m, "tool_calls", None)
            if calls:
                desc = [f"{c['name']}({c['args']})" for c in calls]
                print(f"  [{kind}] 要求调用工具:{desc}")
            elif m.content:
                text = str(m.content).replace("\n", " ")
                print(f"  [{kind}] {text[:180]}")

        print()
        print(f"  ✅ 最终回答:{result['messages'][-1].content}")

    print()
    print(LINE)
    print("对照 T1:框架替你收走了什么")
    print(LINE)
    print("  T1 你要自己写              →  T2 谁来做")
    print("  --------------------------     ------------------------------")
    print("  手写 JSON Schema           →  @tool 装饰器从签名自动生成")
    print("  手写 TOOL_MAP 映射表        →  ToolNode(TOOLS) 自动维护")
    print("  messages.append(...)       →  MessagesState 的 add_messages reducer")
    print("  tool_call_id 手动配对       →  ToolNode 自动配对")
    print("  while + if not tool_calls  →  tools_condition 现成的路由函数")
    print("  并行执行多个工具            →  ToolNode 默认并发")
    print("  try/except 工具异常         →  handle_tool_errors=True")
    print("  自己写重试逻辑              →  RetryPolicy 挂在节点上")
    print("  自己加 max_turns 防死循环    →  recursion_limit(默认 25 步)")
