"""
T3 · 01 自定义 State Schema —— reducer 到底怎么工作

T2 用的是内置的 `MessagesState`,它就一个字段。但真实项目的 state 往往不止一个字段,
而且不同字段的合并规则还不一样。这一步先把 schema 本身摸透,**不调模型**:

    ① MessagesState 拆开长什么样
    ② 自定义多字段 schema:每个字段可以有自己的 reducer
    ③ 一个有 reducer、一个没有,行为差在哪
    ④ add_messages 的四条合并规则(实测)
    ⑤ RemoveMessage:怎么"删掉"一条消息
    ⑥–⑨ reducer 的判定机制 / 签名规则 / add_messages vs operator.add
    ⑩ ⚠️ 想清空 messages 时,`= []` 和 `return []` 为什么全都【不生效且不报错】

跑法:
    cd /d/VibeCoding/langgraph-lab
    ./.venv/Scripts/python 04-state-memory/01_state_schema.py
"""

import operator
from inspect import signature
from typing import Annotated, TypedDict, get_args, get_origin, get_type_hints

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, RemoveMessage
from langgraph.graph import END, START, MessagesState, StateGraph, add_messages
from langgraph.graph.message import REMOVE_ALL_MESSAGES

LINE = "=" * 66


# ==================================================================
# ① MessagesState 拆开长什么样
# ==================================================================
#
# 它就是「一个字段 + 一个 reducer」而已 —— 不是什么黑魔法。
# 你在 T2 里 `return {"messages": [...]}`,框架用的是 add_messages 去合并。


def part_1_messages_state():
    print(LINE)
    print("① MessagesState 拆开是什么")
    print(LINE)
    print(f"  父类:{[c.__name__ for c in MessagesState.__mro__]}")
    print(f"  字段:{list(MessagesState.__annotations__)}")
    ann = MessagesState.__annotations__["messages"]
    print(f"  messages 的注解:{ann}")
    print()
    print("  👆 读法:MessagesState = { messages: Annotated[list, add_messages] }")
    print("     就一个字段 + 一个 reducer。其余全是框架帮你省的事。")


# ==================================================================
# ② 自定义多字段 schema
# ==================================================================
#
# 三种字段,三种合并规则:
#     messages   -> add_messages(按 id upsert)
#     steps      -> operator.add(累加)
#     summary    -> 没有 reducer(直接覆盖)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    steps: Annotated[list[str], operator.add]
    summary: str  # <- 没有 Annotated,就是最普通的字段:后写覆盖先写


def node_one(state: AgentState) -> dict:
    print(f"    第一个节点读到 steps = {state['steps']}")
    return {
        "steps": ["读了输入"],
        "summary": "第一版摘要",
        "messages": [AIMessage("我先看看", id="a1")],
    }


def node_two(state: AgentState) -> dict:
    print(f"    第二个节点读到 steps = {state['steps']}")
    print(f"    第二个节点读到 summary = {state['summary']!r}")
    return {"steps": ["干了点活"], "summary": "第二版摘要(覆盖掉了上一版)"}


def part_2_and_3_custom_schema():
    print()
    print(LINE)
    print("② 自定义多字段 schema + ③ 有/无 reducer 的差异")
    print(LINE)
    print("  schema: messages(reducer: add_messages) / steps(reducer: +) / summary(无 reducer)")
    print()

    builder = StateGraph(AgentState)
    builder.add_node("one", node_one)
    builder.add_node("two", node_two)
    builder.add_edge(START, "one")
    builder.add_edge("one", "two")
    builder.add_edge("two", END)
    graph = builder.compile()

    result = graph.invoke({"messages": [HumanMessage("开始", id="h1")], "steps": [], "summary": ""})

    print()
    print("  最终 state:")
    print(f"    messages = {[type(m).__name__ for m in result['messages']]}")
    print(f"    steps    = {result['steps']}")
    print(f"    summary  = {result['summary']!r}")
    print()
    print("  👆 三个字段,三种表现:")
    print("     messages —— 追加了(a1 进去了,原来的 h1 还在)")
    print("     steps    —— 累加了(两个节点各写一条,都在)")
    print("     summary  —— 只有第二版,第一版被覆盖 —— 因为它没有 reducer")
    print()
    print("  ⚠️ 注意 steps 的坑:两个节点各返回 ['读了输入'] 这种『单元素列表』,")
    print("     而不是把历史带上。因为 reducer 负责拼,你只管交自己那一份。")


# ==================================================================
# ④ add_messages 的四条合并规则
# ==================================================================
#
# 这一步不建图,直接调纯函数 —— 它就是「旧列表 + 新更新 -> 新列表」。


def _dump(title: str, before: list, update: list) -> None:
    after = add_messages(before, update)
    print(f"  {title}")
    print(f"      {len(before)} 条  +  {len(update)} 条更新  ->  {len(after)} 条")
    for m in after:
        print(f"          [{type(m).__name__:>12} id={m.id}] {str(m.content)[:34]}")


def part_4_add_messages_rules():
    print()
    print(LINE)
    print("④ add_messages 的四条合并规则(实测)")
    print(LINE)
    print()

    h1 = HumanMessage("第一问", id="m1")
    a1 = AIMessage("第一答", id="m1-2")

    _dump("规则 1:全新 id  ->  追加", [h1, a1], [HumanMessage("第二问", id="m2")])
    print()
    _dump("规则 2:重复 id  ->  就地替换那一格", [h1, a1], [AIMessage("第一答(改过了)", id="m1-2")])
    print()
    _dump("规则 3:RemoveMessage(id=..)  ->  删掉那一条", [h1, a1], [RemoveMessage(id="m1")])
    print()
    _dump("规则 4:一次给多条  ->  按上面规则逐条处理", [h1], [a1, HumanMessage("第二问", id="m2")])
    print()
    print("  👆 所以 add_messages 不是 `list +=`,而是「按 id 的 upsert + 支持删除」。")
    print("     好处:改一条消息不用重写整个列表 —— 直接把改过的那条丢回去就行。")


# ==================================================================
# ⑤ RemoveMessage:三种删除姿势
# ==================================================================


def part_5_remove():
    print()
    print(LINE)
    print("⑤ RemoveMessage:删除消息的三种姿势")
    print(LINE)
    print()

    msgs = [
        HumanMessage("问 1", id="m1"),
        AIMessage("答 1", id="m2"),
        HumanMessage("问 2", id="m3"),
        AIMessage("答 2", id="m4"),
    ]
    print(f"  初始 4 条:{[m.id for m in msgs]}")
    print()

    kept = add_messages(msgs, [RemoveMessage(id="m1")])
    print(f"  · 删单条  RemoveMessage(id='m1')        -> {[m.id for m in kept]}")

    kept = add_messages(msgs, [RemoveMessage(id=m.id) for m in msgs[:2]])
    print(f"  · 删多条  [RemoveMessage(id=..) x2]     -> {[m.id for m in kept]}")

    kept = add_messages(msgs, [RemoveMessage(id=REMOVE_ALL_MESSAGES)])
    print(f"  · 清空    id=REMOVE_ALL_MESSAGES        -> {[m.id for m in kept]}")
    print()
    print("  👆 REMOVE_ALL_MESSAGES 是 langgraph 给的一个特殊常量,专门用来『一键清空』。")
    print("     它有个硬约束:用它的那一次更新里,不能混进别的非删除消息。")


# ==================================================================
# ⑥ 为什么 Annotated[..., operator.add] 就能当 reducer?
# ==================================================================
#
# 分三层看:
#
#   Python 侧
#       Annotated[list[str], operator.add] 是一个【真实对象】,
#       它把「类型」和「附加元数据」一起装着:
#           get_args(...)   -> (list[str], operator.add)
#           .__metadata__   -> (operator.add,)
#       ⚠️ 但 get_type_hints() 默认会把元数据【剥掉】——
#          默认拿到的是光秃秃的 list[str],必须 include_extras=True。
#
#   框架侧(langgraph/graph/state.py)
#       :1825   type_hints = get_type_hints(schema, include_extras=True)
#       :1904   _is_field_binop() 对每个字段做:
#                 1. 有 __metadata__ 吗?
#                 2. 有的话,取【最后一个】元数据 meta[-1]
#                 3. 它 callable 吗?
#                 4. 它的签名正好两个位置参数吗?  不是 -> 直接报错
#                 -> 全通过:包成 BinaryOperatorAggregate(typ, meta[-1])
#       上面这关没过 -> 退化成 LastValue,也就是「覆盖」
#
#   结论
#       reducer 不是什么特殊语法,就是挂在类型上的一个普通函数。
#       框架只干一件事:把它取出来,按 (旧值, 新值) 调用。


def part_6_the_mechanism():
    print()
    print(LINE)
    print("⑥ Annotated 的元数据是怎么被读出来的")
    print(LINE)
    print()

    alias = Annotated[list[str], operator.add]
    print(f"  Annotated[list[str], operator.add] 本身 = {alias}")
    print(f"      type(...)      -> {type(alias).__name__}")
    print(f"      get_origin(...)-> {get_origin(alias)}")
    print(f"      get_args(...)  -> {get_args(alias)}")
    print(f"      .__metadata__  -> {alias.__metadata__}")
    print()
    print("  ⚠️ 关键的坑:get_type_hints() 默认把元数据剥掉")
    print(f"      默认             -> {get_type_hints(_DemoSchema)}")
    print(f"      include_extras   -> {get_type_hints(_DemoSchema, include_extras=True)}")
    print()
    print("  👆 所以框架必须写 get_type_hints(schema, include_extras=True),")
    print("     否则 reducer 信息在取注解那一步就丢光了。")
    print("     源码位置:langgraph/graph/state.py:1825")


# ==================================================================
# ⑦ reducer 的「契约」:能被当 reducer 的函数长什么样
# ==================================================================


def my_reducer(existing: list, new: list) -> list:
    """一个自定义 reducer —— 框架会照用它,不认 operator.add 这个具体函数。"""
    print(f"      >>> reducer 被调用:existing={existing!r}  new={new!r}")
    return existing + new


class CustomReducerSchema(TypedDict):
    items: Annotated[list[str], my_reducer]


def part_7_reducer_contract():
    print()
    print(LINE)
    print("⑦ reducer 的契约:(旧值, 新值) -> 新值")
    print(LINE)
    print()

    builder = StateGraph(CustomReducerSchema)
    builder.add_node("n1", lambda s: {"items": ["a"]})
    builder.add_node("n2", lambda s: {"items": ["b"]})
    builder.add_edge(START, "n1")
    builder.add_edge("n1", "n2")
    builder.add_edge("n2", END)
    result = builder.compile().invoke({"items": []})

    print(f"  最终 items = {result['items']}")
    print()
    print("  👆 三个观察:")
    print("     1. 框架【照用】了我自己写的函数 —— 它不认 operator.add 这个名字,")
    print("        只要签名对(两个位置参数),就当成 reducer")
    print("     2. 参数顺序是 (existing, new) = (旧值, 新值)")
    print("     3. 开局还有一次 existing=[], new=[] —— 初始 state 也是走 reducer 进去的")

    print()
    print("  签名写错会怎样(只收 1 个参数):")
    try:
        StateGraph(_BadSchema).add_node("n", lambda s: {}).compile()
        print("      没报错(不应该)")
    except Exception as exc:  # noqa: BLE001
        print(f"      {type(exc).__name__}: {exc}")
    print()
    print("  👆 校验发生在【建图/编译时】,不是运行时 —— 写错立刻炸,不会拖到线上。")
    print("     源码:同文件 _is_field_binop(),签名校验就在那儿。")


class _DemoSchema(TypedDict):
    """只用来演示 get_type_hints 的差异。"""

    plain: str
    with_reducer: Annotated[list[str], operator.add]


class _BadSchema(TypedDict):
    """故意放一个签名错误的 reducer。"""

    items: Annotated[list[str], lambda x: x]  # noqa: E731


# ==================================================================
# ⑧ 「签名对」是哪种签名?—— 只数位置参数的个数
# ==================================================================
#
# 源码里(_is_field_binop):
#     sig = signature(meta[-1])
#     params = list(sig.parameters.values())
#     sum(p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in params) == 2
#
# 注意三个「不看」:
#     · 不看参数【名字】—— 叫 a b,还是 existing new,都行
#     · 不看【类型注解】—— 写不写 :list 都不影响判定
#     · 不看函数体
# 只看一件事:这个函数能不能接受两个位置参数。


def _scan(fn, label: str) -> None:
    params = list(signature(fn).parameters.values())
    count = sum(p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in params)
    verdict = "可以当 reducer" if count == 2 else "建图时报错"
    print(f"  {label:30} 位置参数={count}  ->  {verdict}")
    print(f"      {[f'{p.name}:{p.kind.name}' for p in params]}")


def _sig_no_annotation(existing, new):
    return existing + new


def _sig_with_default(existing, new=None):
    return existing


def _sig_star_args(*args):
    return args


def _sig_keyword_only(existing, *, new):
    return existing


def _sig_renamed(a, b):
    return (a or []) + (b or [])


def part_8_signature_rule():
    print()
    print(LINE)
    print("⑧ 「签名对」到底是哪种签名")
    print(LINE)
    print()
    _scan(_sig_no_annotation, "def f(existing, new)")
    _scan(_sig_with_default, "def f(existing, new=None)")
    _scan(_sig_renamed, "def f(a, b)  名字完全不同")
    _scan(_sig_star_args, "def f(*args)")
    _scan(_sig_keyword_only, "def f(existing, *, new)")
    print()
    print("  👆 读法:框架只数「位置参数的个数是不是 2」。")
    print("     · 名字随便叫、类型注解可以不写 —— 都通过")
    print("     · *args(可变位置)算 0 个;纯 keyword-only 只算 1 个 —— 都过不了")
    print("     · 所以「签名」= 两个位置参数,不是「两个带类型注解的参数」")
    print()
    print("  💡 顺带回答「框架不认 operator.add 这个名字」:")
    print("     源码里从头到尾没有出现过 operator.add 这个名字。")
    print("     它只做「取元数据 -> 是不是 callable -> 数参数个数」。")
    print("     所以 operator.add、add_messages、你自己写的函数,在框架眼里是【同一种东西】。")


# ==================================================================
# ⑨ operator.add 与 add_messages:同一份输入,两种结果
# ==================================================================
#
# 两者【不是】谁优谁劣,而是语义完全不同:
#     operator.add   = 纯拼接(堆叠)。它不懂消息,只知道「加上去」
#     add_messages   = 懂消息的合并。它认识 id,会替换、会删除
#
# 而且 —— add_messages 本身就是「一个普通的 reducer」,
# 和你自己写的 my_reducer 地位完全一样,只是官方替你写好了一个。
# 这就是「任何符合签名的函数都能当 reducer」最好的证明。


def _brief(msgs: list) -> str:
    parts = [f"{type(m).__name__}:{m.id}={str(m.content)[:10]}" for m in msgs]
    return "[" + ", ".join(parts) + "]"


def part_9_add_messages_vs_operator_add():
    print()
    print(LINE)
    print("⑨ operator.add vs add_messages:同一份输入")
    print(LINE)
    print()

    msgs = [HumanMessage("问1", id="m1"), AIMessage("答1", id="m2")]
    same_id = [HumanMessage("问1(改过了)", id="m1")]
    delete = [RemoveMessage(id="m1")]

    print(f"  起点(2 条):{_brief(msgs)}")
    print()
    print("  场景 A —— 把一个【已存在的 id】再交一次(AI 重写了自己的回答)")
    print(f"      operator.add  -> {_brief(operator.add(msgs, same_id))}")
    print(f"      add_messages  -> {_brief(add_messages(msgs, same_id))}")
    print()
    print("  场景 B —— 交一张【删除指令】")
    print(f"      operator.add  -> {_brief(operator.add(msgs, delete))}")
    print(f"      add_messages  -> {_brief(add_messages(msgs, delete))}")
    print()
    print("  👆 场景 A:operator.add 交出了【重复的 m1】;add_messages 把 m1 就地换掉了。")
    print("     场景 B:operator.add 把 RemoveMessage 当成一条【普通消息】堆进了列表 ——")
    print("             它不会报错,只会静默地错下去。add_messages 才是真删掉。")
    print()
    print("  ⚠️ 实战含义:messages 字段如果手滑写成 `Annotated[list, operator.add]`,")
    print("     症状是「消息删不掉、改不掉、越滚越长」,而且【不报错】—— 很难查。")
    print()
    print("  共同点也有:两者都遵守同一个契约 —— (旧值, 新值) -> 新值。")
    print("  区别只在【怎么合并】:一个只会加,一个懂 id。")
    print()
    print("  💡 所以 add_messages 并不是什么框架特权 ——")
    print("     它就是官方替你写好的一个 reducer,和你自己写的 my_reducer 地位相同。")


# ==================================================================
# ⑩ 想清空 messages:为什么 `= []` 和 `return []` 都不行
# ==================================================================
def part_10_clear_trap():
    print("\n" + "=" * 66)
    print("⑩ 想清空 messages,为什么不能赋值 [] 也不能 return []")
    print("=" * 66)

    seed = [HumanMessage(content="你好", id="m1"), AIMessage(content="在的", id="m2")]
    print("""
  【先看 reducer 本身,不涉及图】
    add_messages(旧列表, 新列表) 的语义是「把新列表【合并】进旧列表」。
    那如果新列表是空的呢? —— 合并一个空列表 = 什么都没发生。
""")
    got = add_messages(seed, [])
    print(f"      旧列表              : {_brief(seed)}  ({len(seed)} 条)")
    print(f"      add_messages(旧, []) -> {_brief(got)}  ({len(got)} 条)")
    print("      👆 一条都没少。**这就是「返回空列表清不掉」的全部根因。**")

    def _mk(node_fn):
        b = StateGraph(MessagesState)
        b.add_node("n", node_fn)
        b.add_edge(START, "n")
        return b.compile()

    def case_a(state):
        return {"messages": []}

    def case_b(state):
        state["messages"] = []
        return {}

    def case_c(state):
        state["messages"] = []
        return dict(state)

    def case_d(state):
        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}

    seed_state = {"messages": [HumanMessage(content="你好", id="s1"),
                               AIMessage(content="在的", id="s2")]}

    print("\n  【再看图里:三种「直觉写法」 vs 一种正确写法】")
    print(f"    起始 state: {_brief(seed_state['messages'])}  (2 条)\n")
    cases = [
        ("A. return {\"messages\": []}", case_a),
        ("B. state[\"messages\"] = []; return {}", case_b),
        ("C. state[\"messages\"] = []; return dict(state)", case_c),
        ("D. return {\"messages\": [RemoveMessage(REMOVE_ALL)]}", case_d),
    ]
    for label, fn in cases:
        out = _mk(fn).invoke(seed_state)
        n = len(out["messages"])
        mark = "✅ 清空了" if n == 0 else f"❌ 没清掉,还是 {n} 条"
        print(f"    {label:<50} {mark}")

    print("""
  ── 为什么 A 不行:节点返回值是「更新指令」,不是「最终状态」────────
     你 `return {"messages": []}` 的意思是:
         "请往 messages 里【加】0 条消息"
     而不是:
         "请把 messages 【设成】空列表"

     框架拿到这个 dict,逐字段交给该字段的 reducer:
         add_messages(旧列表, [])  ->  旧列表原样返回
     所以**不报错、也不生效** —— 静默失效。

  ── 为什么 B 不行:节点里改的 state 是「当次读出来的快照」───────
     节点函数拿到的那份 state,是框架从 channel 读出来交给你的。
     你在里面 `state["messages"] = []`,改的是【你手里这份】,
     框架在节点返回后并**不会**把它写回去 —— 它只看你 return 了什么。
     而你 return 的是 `{}`(空更新),所以图里的 state 一点没动。

  ── 为什么 C 也不行:这条最反直觉 ─────────────────────────
     `state["messages"] = []` 之后再 `return dict(state)`,
     看起来是"我把改好的整个 state 交回去了"。
     但**框架仍然把它当更新指令**,还是走 `add_messages(旧, [])` ——
     依然是"加 0 条",依然什么都没发生。

     记住这句话:
     **节点的返回值永远是"我要改哪些字段、改成什么增量",永远不是"新的 state 长这样"。**

  ── 那正确的表达方式是什么 ───────────────────────────────
     在 reducer 的世界里,**"想清空"的表达方式是「发一条删除指令」**,
     不是"交一个空列表":

         from langgraph.graph.message import REMOVE_ALL_MESSAGES
         return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}

     add_messages 认得这个特殊的 id,于是一条条把列表清空。

     同理,想删某几条,也得点名 id:
         return {"messages": [RemoveMessage(id=m.id) for m in state["messages"][:-2]]}

  ── 💡 一句话总结 ────────────────────────────────────────
     在 LangGraph 里,**你不是在"操作 state",你是在"提交变更"**。
     清空 = 提交一条"全删"变更;而不是提交一个空 state。
     搞混这两件事,就会得到一堆**不报错的错误代码**。
""")


if __name__ == "__main__":
    part_1_messages_state()
    part_2_and_3_custom_schema()
    part_4_add_messages_rules()
    part_5_remove()
    part_6_the_mechanism()
    part_7_reducer_contract()
    part_8_signature_rule()
    part_9_add_messages_vs_operator_add()
    part_10_clear_trap()
