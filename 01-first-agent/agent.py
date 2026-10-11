"""
T0 · 第一个 agent

目标:跑通「模型 + 工具」的最小闭环 —— 让模型自己决定要不要调工具。

跑法(Git Bash):
    cd /d/VibeCoding/langgraph-lab
    ./.venv/Scripts/python 01-first-agent/agent.py

要点(这是 T0 真正要理解的东西):
  1. 函数的 docstring 就是「工具说明书」—— 模型靠它决定调不调用、传什么参数
  2. create_agent 底层跑在 LangGraph 上,这一步先不管它,能跑通就行
  3. LangSmith 开了的话,去后台能看到这次运行的完整 trace
"""

import os
from datetime import datetime

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

load_dotenv()


# ---------- 工具函数 ----------
# 注意:注释/docstring 不是给我看的,是给模型看的,写清楚它才知道什么时候用


def get_current_time() -> str:
    """返回当前本地日期和时间,格式 YYYY-MM-DD HH:MM:SS。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def add(a: float, b: float) -> float:
    """把两个数字相加,返回结果。用于需要精确计算的场合。"""
    return a + b


# ---------- 模型 ----------
# 默认:智谱 GLM-4.7-Flash —— 免费、200K 上下文、支持工具调用、国内直连
# 走的是 OpenAI 兼容协议,所以换任何供应商都只改 .env 三个变量,代码不用动
_api_key = os.getenv("MODEL_API_KEY")
if not _api_key:
    raise RuntimeError("没找到 MODEL_API_KEY —— 请检查 .env 文件")

model = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "glm-4.7-flash"),
    base_url=os.getenv("MODEL_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
    # ⚠️ 用 SecretStr 包一层,别直接传 str:
    #    ChatOpenAI 的类型标注是 SecretStr|None|Callable,不含 str,
    #    直接传字符串 Pylance 会标红(虽然运行时 pydantic 会自动转换、照样能跑)
    api_key=SecretStr(_api_key),
    temperature=0,
)


# ---------- Agent ----------
agent = create_agent(
    model=model,
    tools=[get_current_time, add],
    system_prompt=(
        "你是一个简洁的助手。"
        "涉及时间或精确计算时,必须调用工具,不要凭记忆回答。"
    ),
)


if __name__ == "__main__":
    question = "现在几点?顺便算一下 1234 + 5678 等于多少"

    print(f"\n>>> 提问:{question}\n")

    result = agent.invoke({"messages": [{"role": "user", "content": question}]})

    # 打印完整消息链 —— 重点看「中间发生了什么」
    for msg in result["messages"]:
        msg.pretty_print()
