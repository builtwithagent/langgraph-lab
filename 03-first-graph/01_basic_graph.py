"""
T2 · 第一个 StateGraph —— 先不碰 LLM,纯本地把「图」的零件摸一遍

从 T1 的 while 循环 到 LangGraph 的状态图,换掉的到底是什么:

    T1 手搓:  messages 列表(你自己维护) + while 循环(你自己写) + if 判断(你自己写)
    T2 框架:  State(声明式)           + Node(纯函数)        + Edge / 条件边(声明式)

一句话:**把「怎么流转」从命令式代码,变成声明式的结构。**

图的零件总共就四个:
    ① State   —— 一块共享内存,所有节点都能读、都能写
    ② Node    —— 一个普通函数:收 state,返回「我要更新哪些字段」
    ③ Edge    —— 固定的下一步
    ④ 条件边   —— 由函数决定下一步走哪 —— **这是图里唯一能做分支的地方**


================================================================
📐 本文件两个示例的图例(先看图,再看代码)
================================================================

  示例一(最小图):一条直线,没有分支

      START --> [ a ] --> [ b ] --> END

      三个节点共用同一块 State:

          State = { question: str, answer: str, trail: list[str] }
                     (a 只读)        (b 读写)      (靠 reducer 累加)

      一句话:节点不是「函数调用」,是「对同一块内存的两轮改写」。
      a 返回 {"trail": [...]},b 返回 {"answer": ..., "trail": [...]} ——
      都是 partial update,谁都没交回整个 state。

  示例二(条件边):一条岔路

      START --> [ read_amount ] --> [ judge ] --+- (高额) --> [ high_priority ] --> END
                                                 |
                                                 +- (普通) --> [ normal        ] --> END

      注意 judge 自己不做判断,它只把结论写进 state(verdict);
      真正「决定走哪」的是 route() —— 这在 LangGraph 里叫条件边。
      对比 T1:它就是手写 while 循环里那个 if,只不过被声明出来了。


================================================================
📎 知识点补齐(卡住了就点开,全部实测可访问)
================================================================

  本文件只用到 5 个新概念,逐个给入口:

  [1] State / Node / Edge / 图的整体心智模型  ← 最该先看这个
      · LangGraph 官方 Graph API 文档
        https://docs.langchain.com/oss/python/langgraph/graph-api
      · 「用图的方式思考」(官方概念篇,比 API 文档好懂)
        https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph

  [2] reducer(Annotated + operator.add)—— 本文件最容易看漏的点
      · 上面那篇 Graph API 的 State / Reducers 小节
      · Python 官方 `Annotated` 说明
        https://docs.python.org/3/library/typing.html#typing.Annotated
      · Python 官方 `operator.add` 说明
        https://docs.python.org/3/library/operator.html#operator.add

  [3] TypedDict(State 为什么声明成它)
      · Python 官方文档
        https://docs.python.org/3/library/typing.html#typing.TypedDict

  [4] 条件边 + 第三个参数 path_map
      · Graph API 文档里的 Conditional edges 小节(同上 URL)

  [5] Mermaid(文件末尾会打印图的 Mermaid 源码)
      · 语法速查 https://mermaid.js.org/syntax/flowchart.html
      · 在线预览 https://mermaid.live/
        (把末尾打印出来的源码粘进去,就能看到图长什么样)

  想跟着视频过一遍(中文,B站,2026-10-11 用 B 站 API 核验过标题):
      · 19 分钟快速入门与原理剖析 —— 最短,先建立概念
        https://www.bilibili.com/video/BV1HVFLzfEeS
      · 尚硅谷 LangGraph 教程 —— 体系化,分 P 讲
        https://www.bilibili.com/video/BV1z3NY66EY1
      · 官方免费课程(英文,有字幕;讲得比大多数中文教程细)
        https://academy.langchain.com/courses/intro-to-langgraph


跑法:
    cd /d/VibeCoding/langgraph-lab
    ./.venv/Scripts/python 03-first-graph/01_basic_graph.py
"""

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

LINE = "=" * 66


# ==================================================================
# 第一段:最小的图 —— State + Node + Edge,就三样
# ==================================================================
#
#   图长这样(一条直线,没有分支):
#
#       START --> [ a ] --> [ b ] --> END
#
#   跑的时候,状态是这样一步步被填满的
#   (Q = 输入的问题,A = b 算出来的回答):
#
#       invoke 时           node_a 之后          node_b 之后
#       question = Q        question = Q        question = Q
#       answer   = ""       answer   = ""       answer   = A
#       trail    = []       trail    = [a]      trail    = [a, b]
#
#   👆 a 只改 trail;b 改 answer + trail —— 谁都没交回整个 state。
#
#   ⚠️ 两个节点都往 trail 里写,结果却是两句话都在 ——
#      这不是默认行为,是 trail 声明了 reducer(operator.add)才有的。
#      没有 reducer,后写的会直接盖掉先写的,最后只剩 ['node_b 跑过了']。


class MinimalState(TypedDict):
    """① State:图的「共享内存」。

    注意它是 TypedDict —— 只是个类型说明,运行时就是个普通 dict。
    每个节点读它拿输入、返回 dict 表示「要改哪几个字段」。
    """

    question: str
    answer: str
    # Annotated[..., operator.add] = reducer:
    # 默认行为是「后面覆盖前面」,加了 reducer 变成「累加」——
    # 这是 LangGraph 里一个很关键的概念,消息列表就是这么处理的
    trail: Annotated[list[str], operator.add]


def node_a(state: MinimalState) -> dict:
    """② Node:一个普通函数。收 state,返回「要更新的字段」。"""
    print(f"  [node_a] 读到 question = {state['question']!r}")
    return {"trail": ["node_a 跑过了"]}


def node_b(state: MinimalState) -> dict:
    print(f"  [node_b] 读到 trail = {state['trail']}")
    return {"answer": f"回答:{state['question']} —— 42", "trail": ["node_b 跑过了"]}


def build_minimal():
    """③ Edge:固定的下一步。START / END 是两个特殊标记。"""
    builder = StateGraph(MinimalState)
    builder.add_node("a", node_a)          # 注册节点
    builder.add_node("b", node_b)
    builder.add_edge(START, "a")           # 起点 → a
    builder.add_edge("a", "b")             # a → b
    builder.add_edge("b", END)             # b → 终点
    return builder.compile()               # ⚠️ 必须 compile,不 compile 不能 invoke


# ==================================================================
# 第二段:条件边 —— 图里唯一的分支来源
# ==================================================================
#
#   图长这样(judge 之后岔成两条路):
#
#       START --> [ read_amount ] --> [ judge ] --+--(高额)--> [ high_priority ] --> END
#                                                 |
#                                                 +--(普通)--> [ normal        ] --> END
#
#   state 在各节点的变化:
#
#       invoke 时       read_amount 之后    judge 之后         high_priority / normal 之后
#       order_id = A    order_id = A       order_id = A       order_id = A
#       amount   = 0    amount   = 1299.0  (不变)             (不变)
#       verdict  = ""   verdict  = ""      verdict = 高优先级  verdict = 高优先级
#       trail    = []   trail = [读金额]    trail = [..判定]    trail = [..→分配客服]
#
#   👆 judge 自己不做分支,它只把结论写进 state(verdict 字段)。
#      真正决定走哪的是 route() —— 这个函数就是条件边。
#
#   对比 T1:route() 就是你手写 while 循环里那个 if。
#            区别是:if 藏在代码里,条件边被「声明」进了图结构 ——
#            所以 get_graph().draw_mermaid() 能把分支画出来。

# 模拟的订单库:订单号 -> 金额
FAKE_ORDERS = {"A1001": 1299.0, "A1002": 89.0}


class ReviewState(TypedDict):
    order_id: str
    amount: float
    verdict: str
    trail: Annotated[list[str], operator.add]


def read_amount(state: ReviewState) -> dict:
    amount = FAKE_ORDERS.get(state["order_id"], 0.0)
    print(f"  [read_amount] 订单 {state['order_id']} 金额 {amount} 元")
    return {"amount": amount, "trail": [f"读取金额 {amount} 元"]}


def judge(state: ReviewState) -> dict:
    verdict = "高优先级" if state["amount"] >= 1000 else "普通工单"
    print(f"  [judge] 判定结果:{verdict}")
    return {"verdict": verdict, "trail": [f"判定:{verdict}"]}


def route(state: ReviewState) -> str:
    """④ 条件边 = 一个函数,返回「下一个节点的名字」。

    整个图里,只有这一类函数能做分支判断。
    对比 T1:这相当于你手写 while 循环里的那个 if。
    """
    return "high_priority" if state["verdict"] == "高优先级" else "normal"


def high_priority(state: ReviewState) -> dict:
    print("  [high_priority] 分配人工客服,2 小时内响应")
    return {"trail": ["→ 分配人工客服"]}


def normal(state: ReviewState) -> dict:
    print("  [normal] 走自助处理流程")
    return {"trail": ["→ 走自助流程"]}


def build_review():
    builder = StateGraph(ReviewState)
    builder.add_node("read_amount", read_amount)
    builder.add_node("judge", judge)
    builder.add_node("high_priority", high_priority)
    builder.add_node("normal", normal)

    builder.add_edge(START, "read_amount")
    builder.add_edge("read_amount", "judge")

    # ⬇️ 条件边:不写死下一步,交给 route() 决定
    #
    # 第三个参数 path_map 是「可能的分支」的显式声明。
    # 不传也能跑(靠 route 的返回值直接找同名节点),但有一个副作用:
    #   → 图结构可视化画不出分支,只会画一条 judge → END 的假边。
    # 传上之后,draw_mermaid() / 架构图就能正确显示两条分支了。
    # 👉 习惯:条件边一律把 path_map 写上。
    builder.add_conditional_edges(
        "judge",
        route,
        {"high_priority": "high_priority", "normal": "normal"},
    )

    builder.add_edge("high_priority", END)
    builder.add_edge("normal", END)
    return builder.compile()


if __name__ == "__main__":
    # ---------------- 第一段 ----------------
    print(LINE)
    print("第一段:最小的图(STATE + NODE + EDGE)")
    print(LINE)
    g1 = build_minimal()
    print("  图结构:")
    print("    START → a → b → END")
    print()
    result = g1.invoke({"question": "生命、宇宙以及一切的答案是什么?", "trail": []})
    print()
    print(f"  最终 state:{result}")
    print("  👆 注意 trail 是 ['node_a 跑过了', 'node_b 跑过了'] ——")
    print("     两个节点都往同一个字段写,靠 reducer 累加起来了,没有互相覆盖。")

    # ---------------- 第二段 ----------------
    print()
    print(LINE)
    print("第二段:条件边(图里唯一的分支来源)")
    print(LINE)
    print("  图结构:")
    print("    START → read_amount → judge ─┬─(高额)─→ high_priority → END")
    print("                                  └─(普通)─→ normal        → END")
    g2 = build_review()

    for order_id in ("A1001", "A1002"):
        print()
        print(f"  ---- 输入 {order_id} ----")
        out = g2.invoke({"order_id": order_id, "trail": []})
        print(f"  结果:verdict={out['verdict']},trail={out['trail']}")

    # ---------------- 把图结构导成 Mermaid ----------------
    print()
    print(LINE)
    print("图的自动描述(可以直接贴进 README / 架构图工具)")
    print(LINE)
    print(g2.get_graph().draw_mermaid())
