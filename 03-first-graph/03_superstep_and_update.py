"""
T2 补充实验 · superstep 到底是什么 + 节点的返回值能不能给全量

回答两个问题,**全程不调模型**(纯本地图,秒出):
    实验 1  superstep 的边界在哪?一个节点执行一次就算一个 superstep 吗?
    实验 2  recursion_limit 数的到底是什么?
    实验 3  节点返回 partial update vs 返回完整 state,差在哪?
    实验 4  返回值的两个边界情况

跑法:
    cd /d/VibeCoding/langgraph-lab
    ./.venv/Scripts/python 03-first-graph/03_superstep_and_update.py
"""

import operator
import time
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

LINE = "=" * 66
SLEEP = 0.6  # 每个节点睡这么久,用来"称量"哪些节点是同时跑的


class Trail(TypedDict):
    """一个最小的 state:只有一个带 reducer 的字段。"""

    trail: Annotated[list[str], operator.add]


def _mk_node(name: str):
    """造一个节点:睡一会,然后往 trail 里追一条自己的名字。"""

    def node(state: Trail) -> dict:
        time.sleep(SLEEP)
        return {"trail": [name]}

    return node


def _build(edges: list[tuple], nodes: list[str]):
    builder = StateGraph(Trail)
    for n in nodes:
        builder.add_node(n, _mk_node(n))
    for src, dst in edges:
        builder.add_edge(src, dst)
    return builder.compile()


# ==================================================================
# 实验 1 + 2:superstep 的边界 & recursion_limit 数什么
# ==================================================================
#
#   三种图的形状,每种都是「3 个节点、每个睡 0.6 秒」,总耗时说明一切:
#
#     A  START --> a,b,c 三个一起扇出       -> 1 个 superstep(3 个并发)
#     B  START --> fanout --> a,b,c         -> 2 个 superstep
#     C  START --> a --> b --> c(纯串行)   -> 3 个 superstep
#
#   然后用 recursion_limit 反推 superstep 个数做交叉验证。


GRAPH_A = ("A: START 一次性扇出 3 个", [(START, "a"), (START, "b"), (START, "c"),
                                       ("a", END), ("b", END), ("c", END)], ["a", "b", "c"])

GRAPH_B = ("B: fanout 之后再扇出 3 个", [(START, "fan"), ("fan", "a"), ("fan", "b"), ("fan", "c"),
                                        ("a", END), ("b", END), ("c", END)], ["fan", "a", "b", "c"])

GRAPH_C = ("C: 纯串行 3 个", [(START, "a"), ("a", "b"), ("b", "c"), ("c", END)], ["a", "b", "c"])


def experiment_1_and_2():
    print(LINE)
    print("实验 1:superstep 的边界 ≠ 一个节点")
    print(LINE)
    print(f"  三种图都是 3 个节点,每个节点睡 {SLEEP}s")
    print()

    graphs = {}
    costs = {}
    for label, edges, nodes in (GRAPH_A, GRAPH_B, GRAPH_C):
        g = _build(edges, nodes)
        graphs[label] = g
        t0 = time.perf_counter()
        g.invoke({"trail": []})
        costs[label] = time.perf_counter() - t0
        print(f"  {label}")
        print(f"      耗时 {costs[label]:.2f}s")
    print()
    print("  👆 读法:")
    print(f"     A {costs[GRAPH_A[0]]:.2f}s  → 3 个节点挤在【同一个 superstep】里并发跑")
    print(f"     B {costs[GRAPH_B[0]]:.2f}s  → fanout 独占 1 步,然后 a/b/c 在下一步并发")
    print(f"     C {costs[GRAPH_C[0]]:.2f}s  → 3 个节点分散在【3 个 superstep】,只能排队")
    print()
    print("  ⚠️ 所以「一个节点执行一次 = 一个 superstep」只在串行时成立。")
    print("     准确说法:一个 superstep = 图中被同时激活的一批节点,")
    print("     这批节点全部跑完后,state 才会统一更新一次(屏障)。")

    print()
    print(LINE)
    print("实验 2:recursion_limit 数的到底是什么")
    print(LINE)

    cases = [
        ("A 扇出 3 个(1 个 superstep)", "A: START 一次性扇出 3 个", [1, 2]),
        ("C 串行 3 个(3 个 superstep)", "C: 纯串行 3 个", [3, 4]),
    ]
    for label, key, limits in cases:
        for lim in limits:
            try:
                graphs[key].invoke({"trail": []}, config={"recursion_limit": lim})
                print(f"  {label}  recursion_limit={lim}  ->  跑完了")
            except Exception as exc:  # noqa: BLE001 - 就是要把异常类型打出来
                print(f"  {label}  recursion_limit={lim}  ->  {type(exc).__name__}")
    print()
    print("  👆 读法:A 需要 2,C 需要 4 —— 都比「节点执行的步数」多 1。")
    print("     多出来的那 1 步,是最后「走到 END」的那一步,它同样计数。")
    print("     公式:recursion_limit >= 节点执行的 superstep 数 + 1")
    print()
    print("  ⚠️ 但如果是【死循环】(永远到不了 END),上限就是能白跑多少步:")
    print("     自环图 limit=25 -> 节点实际执行 25 次,然后抛错。")
    print("     换算到 agent 图(agent/tools 两步一轮):默认 25 ≈ 12 轮工具调用。")


# ==================================================================
# 实验 3 + 4:节点的返回值
# ==================================================================
#
#   问:节点入参是完整 state,那返回值能不能也给个完整 state?
#
#   答:语法上可以,但【语义上危险】——
#   返回值不是「覆盖整个 state」,而是「逐字段交给各个字段自己的 reducer」。
#       无 reducer 的字段:你给什么就是什么(给全量 = 给部分)
#       有 reducer 的字段:你给的值会被【合并】到现有值上 -> 给全量就重复了


def _run_one(fn, label: str):
    builder = StateGraph(Trail)
    builder.add_node("n1", _mk_node("n1"))
    builder.add_node("probe", fn)
    builder.add_edge(START, "n1")
    builder.add_edge("n1", "probe")
    builder.add_edge("probe", END)
    result = builder.compile().invoke({"trail": []})
    print(f"  {label}")
    print(f"      trail = {result['trail']}")


def experiment_3_and_4():
    print()
    print(LINE)
    print("实验 3:返回 partial  vs  返回完整 state")
    print(LINE)
    print("  前置:n1 先跑,trail 已经是 ['n1']")
    print()

    _run_one(lambda s: {"trail": ["n2"]}, "① probe 只返回要改的字段(partial):")
    print()
    _run_one(lambda s: {"trail": s["trail"] + ["n2"]}, "② probe 返回完整 state(trail 带上历史):")
    print()
    print("  👆 读法:")
    print("     ① ['n1', 'n2']       —— 期望结果")
    print("     ② ['n1', 'n1', 'n2'] —— ⚠️ n1 被算了两次!历史被 reducer 又加了一遍")
    print()
    print("  原因:trail 有 operator.add 这个 reducer。节点返回 ['n1','n2'] 时,")
    print("  reducer 做的是「现有值 + 你返回的值」,不是「用你的值替换」。")
    print("  所以【给得越多 ≠ 越保险,而是重复越多】。")
    print()
    print("  结论:入参必须是完整 state(框架保证),")
    print("        返回值【永远只写你要改的字段】—— 这是唯一安全的写法。")

    print()
    print(LINE)
    print("实验 4:返回值的两个边界情况")
    print(LINE)
    print()

    _run_one(lambda s: {"trail": ["x"], "bogus": 123}, "① 返回一个 state 里没声明的 key:")
    print()
    _run_one(lambda s: {}, "② 返回空 dict:")
    print()
    print("  👆 读法:")
    print("     ① 多余的 key 被【静默丢弃】,不报错也不写进 state —— 但也别指望它能扩展 state")
    print("     ② 空 dict 完全合法,表示「这一步不改任何字段」")


# ==================================================================
# 实验 5:「界」到底指什么 —— 让节点自己报告它读到的 state
# ==================================================================
#
#   并行时 a、b 都读到空 trail  -> 互相看不见,读的是同一份快照
#   串行时 b 读到了 a 写的东西  -> 界在每一步之间都发生过一次
#
#   这就是「界」的实证:这一批节点全部停笔之后,才允许把结果写进 state。


class BarrierState(TypedDict):
    trail: Annotated[list[str], operator.add]


def _reporter(name: str):
    """节点不只写,还把「我读到的 state」打出来。"""

    def node(state: BarrierState) -> dict:
        print(f"       {name} 读到的 trail = {state['trail']}")
        return {"trail": [f"{name} 写的"]}

    return node


def _barrier_graph(edges: list) -> object:
    builder = StateGraph(BarrierState)
    for n in ("a", "b"):
        builder.add_node(n, _reporter(n))
    for src, dst in edges:
        builder.add_edge(src, dst)
    return builder.compile()


def experiment_5_barrier():
    print()
    print(LINE)
    print("实验 5:「界」到底指什么")
    print(LINE)

    print()
    print("  ① 并行:START -> a | b -> END")
    result = _barrier_graph([(START, "a"), (START, "b"), ("a", END), ("b", END)]).invoke({"trail": []})
    print(f"       最终 trail = {result['trail']}")
    print("       👆 a、b 都读到空 —— 互相看不见,因为读的是同一份旧 state")

    print()
    print("  ② 串行:START -> a -> b -> END")
    result = _barrier_graph([(START, "a"), ("a", "b"), ("b", END)]).invoke({"trail": []})
    print(f"       最终 trail = {result['trail']}")
    print("       👆 b 读到了 a 写的 —— 因为 a 交卷后,界已经发生过一次")

    print()
    print("  结论:界 = 「这一批节点全部停笔之后,才允许把结果写进 state」的那个时刻。")
    print("        串行时界照样存在,只是每步只有 1 个节点,所以你察觉不到。")


if __name__ == "__main__":
    experiment_1_and_2()
    experiment_3_and_4()
    experiment_5_barrier()
    print()
    print(LINE)
    print("一句话总结")
    print(LINE)
    print("  · 一个 superstep = 一次 state 更新所对应的那一批节点")
    print("  · recursion_limit 数的是 superstep,且「走到 END」那一步也要占 1 格")
    print("  · 入参永远是完整 state;返回值只能是 partial —— 给全量会触发 reducer 重复累加")
    print("  · 「界」= 一批节点全部停笔后才写 state;它保证并行结果不依赖谁跑得快")
