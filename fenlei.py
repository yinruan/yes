# yes
"""
5.3.3.2 意图识别 + 不同处理链路
=================================
核心流程：
  用户输入 → 意图识别 Chain（PromptTemplate+模型+JsonOutputParser）
           → RouterChain 根据意图分发
           → 文本创作 Chain（写作类：诗歌/散文/故事等）
           → 数学计算 Chain（计算类：数学表达式求值）
           → 代码生成 Chain（编程类：生成代码片段）
           → 知识问答 Chain（百科类：回答知识性问题）
           → 反馈结果

使用 LangGraph 构建带条件边的工作流图，
使用 LangChain Tool 封装计算工具和代码执行工具。
"""

import ast
import math
import operator
import re
from typing import Annotated, Literal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ─────────────────────────────────────────────
# 1. 工具定义（LangChain Tool）
# ─────────────────────────────────────────────

@tool
def calculator(expression: str) -> str:
    """
    安全地计算数学表达式。
    支持：+、-、*、/、**、//、%、括号、sqrt、sin、cos、pi、e 等。
    示例：calculator("(1 + 2) * 3")  →  "9"
    """
    # 白名单安全解析
    allowed_names = {
        "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
        "tan": math.tan, "log": math.log, "log10": math.log10,
        "abs": abs, "round": round, "pi": math.pi, "e": math.e,
        "pow": math.pow, "ceil": math.ceil, "floor": math.floor,
    }
    try:
        # 仅允许安全的 AST 节点
        tree = ast.parse(expression, mode='eval')
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id not in allowed_names:
                        return f"错误：不允许调用函数 '{node.func.id}'"
        result = eval(compile(tree, '<string>', 'eval'),
                      {"__builtins__": {}}, allowed_names)
        # 格式化输出
        if isinstance(result, float) and result == int(result):
            return str(int(result))
        return str(round(result, 10))
    except ZeroDivisionError:
        return "错误：除以零"
    except Exception as e:
        return f"计算错误：{e}"


@tool
def code_formatter(code: str, language: str = "python") -> str:
    """
    对生成的代码进行基本格式标注，返回带语言标签的 Markdown 代码块。
    用于代码生成链路中的后处理。
    """
    code = code.strip()
    if not code.startswith("```"):
        return f"```{language}\n{code}\n```"
    return code


# ─────────────────────────────────────────────
# 2. 意图类型定义
# ─────────────────────────────────────────────

INTENT_TYPES = Literal[
    "text_creation",   # 文本创作：写作、诗歌、散文、故事
    "math_calculation",# 数学计算：计算表达式、数学题
    "code_generation", # 代码生成：写代码、编程
    "knowledge_qa",    # 知识问答：百科、事实性问题
    "unknown",         # 未知/其他
]


class IntentClassification(BaseModel):
    """意图分类结果"""
    intent: INTENT_TYPES = Field(description="用户意图类型")
    sub_type: str = Field(default="", description="细化的子类型，如'诗歌'/'散文'/'Python代码'等")
    key_content: str = Field(description="提炼出的核心任务内容，去掉套话")
    math_expr: str = Field(
        default="",
        description="若为数学计算意图，提取出纯数学表达式（可被 Python eval 执行），否则为空"
    )
    confidence: float = Field(description="置信度 0~1")


# ─────────────────────────────────────────────
# 3. 状态定义
# ─────────────────────────────────────────────

class RouterState(TypedDict):
    messages: Annotated[list, add_messages]
    user_input: str
    intent: str
    sub_type: str
    key_content: str
    math_expr: str
    tool_result: str       # 工具调用结果
    response: str


# ─────────────────────────────────────────────
# 4. Prompts
# ─────────────────────────────────────────────

INTENT_CLASSIFY_SYSTEM = """你是一个意图分类器，将用户输入分类到以下意图之一：

- text_creation：用户想要创作文字内容（写文章、诗歌、散文、故事、文案等）
- math_calculation：用户想要进行数学计算（含表达式、数学题、统计等）
- code_generation：用户想要生成代码（不限语言，含算法题、脚本等）
- knowledge_qa：用户在询问知识性问题（历史、科学、常识、事实等）
- unknown：无法归类的其他请求

输出严格 JSON，不要有多余文字。
"""

intent_classify_prompt = ChatPromptTemplate.from_messages([
    ("system", INTENT_CLASSIFY_SYSTEM),
    ("human", "用户输入：{user_input}\n\n请分类并输出 JSON。"),
])

# 各子 Chain 的 Prompt
TEXT_CREATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一位才华横溢的文字创作者，擅长各种体裁的写作。
创作风格：细腻、生动、富有感染力。
子类型：{sub_type}
请根据用户需求进行创作，作品要完整、有质量，字数适中（200-500字）。
"""),
    ("human", "{key_content}"),
])

MATH_CALCULATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是数学讲解老师。
计算工具已经给出了结果：{tool_result}
请结合计算结果，用清晰的步骤向用户解释计算过程（如有必要），
并给出最终答案。若结果是错误信息，请分析原因并给出正确解法。
"""),
    ("human", "{user_input}"),
])

CODE_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一位资深程序员，擅长多种编程语言。
编程语言偏好：{sub_type}（若未指定则使用 Python）
要求：
1. 代码要有注释，易于理解
2. 考虑边界情况
3. 提供简短的使用说明
"""),
    ("human", "{key_content}"),
])

KNOWLEDGE_QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个博学的知识助手，能准确回答各类知识性问题。
回答要求：
1. 内容准确，有据可查
2. 语言简洁易懂
3. 如果问题有多个维度，分点说明
4. 适当举例说明
"""),
    ("human", "{user_input}"),
])

UNKNOWN_INTENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个通用 AI 助手。
用户的请求不属于你的专项处理范围（文本创作/数学计算/代码生成/知识问答）。
请：
1. 友好地告知当前支持的功能范围
2. 尝试用通用方式回答用户问题（如果有办法的话）
3. 引导用户重新描述或换个方向
"""),
    ("human", "{user_input}"),
])


# ─────────────────────────────────────────────
# 5. LangGraph 节点
# ─────────────────────────────────────────────

def build_intent_router_graph(llm: ChatOpenAI):
    """构建意图路由图"""

    intent_parser = JsonOutputParser(pydantic_object=IntentClassification)

    # ── 节点1：意图识别 ──────────────────────────
    def intent_node(state: RouterState) -> dict:
        """使用 LLM + JsonOutputParser 识别意图"""
        user_input = state["user_input"]
        messages = intent_classify_prompt.format_messages(user_input=user_input)
        response = llm.invoke(messages)

        try:
            result = intent_parser.parse(response.content)
            intent     = result["intent"]
            sub_type   = result.get("sub_type", "")
            key_content = result.get("key_content", user_input)
            math_expr  = result.get("math_expr", "")
        except Exception:
            # 容错：正则提取
            content = response.content
            m = re.search(r'"intent"\s*:\s*"([^"]+)"', content)
            intent = m.group(1) if m else "unknown"
            sub_type = ""
            key_content = user_input
            math_expr = ""

        print(f"\n[意图识别] intent={intent}, sub_type={sub_type or '无'}")
        print(f"           key_content={key_content[:50]}")

        return {
            "intent": intent,
            "sub_type": sub_type,
            "key_content": key_content,
            "math_expr": math_expr,
            "messages": [HumanMessage(content=user_input)],
        }

    # ── 节点2：文本创作 Chain ─────────────────────
    def text_creation_node(state: RouterState) -> dict:
        messages = TEXT_CREATION_PROMPT.format_messages(
            sub_type=state.get("sub_type", "通用"),
            key_content=state["key_content"],
        )
        response = llm.invoke(messages)
        print(f"[文本创作链] 已生成，字数约 {len(response.content)}")
        return {
            "response": response.content,
            "messages": [AIMessage(content=response.content)],
        }

    # ── 节点3：数学计算 Chain（含工具调用）───────
    def math_tool_node(state: RouterState) -> dict:
        """调用 calculator 工具"""
        expr = state.get("math_expr", "").strip()
        if expr:
            tool_result = calculator.invoke({"expression": expr})
            print(f"[数学工具] 表达式={expr} → 结果={tool_result}")
        else:
            # 尝试从 key_content 中提取数字表达式
            content = state["key_content"]
            # 简单提取数字和运算符
            expr_match = re.search(r'[\d\s\+\-\*\/\(\)\.\^%]+', content)
            if expr_match:
                expr = expr_match.group().strip()
                tool_result = calculator.invoke({"expression": expr})
                print(f"[数学工具] 自动提取表达式={expr} → 结果={tool_result}")
            else:
                tool_result = "无法从问题中提取数学表达式，请直接用文字描述"
        return {"tool_result": tool_result}

    def math_answer_node(state: RouterState) -> dict:
        """基于工具结果生成解释"""
        messages = MATH_CALCULATION_PROMPT.format_messages(
            tool_result=state.get("tool_result", "无结果"),
            user_input=state["user_input"],
        )
        response = llm.invoke(messages)
        return {
            "response": response.content,
            "messages": [AIMessage(content=response.content)],
        }

    # ── 节点4：代码生成 Chain ─────────────────────
    def code_generation_node(state: RouterState) -> dict:
        messages = CODE_GENERATION_PROMPT.format_messages(
            sub_type=state.get("sub_type", "Python"),
            key_content=state["key_content"],
        )
        response = llm.invoke(messages)
        # 后处理：确保代码有格式标注
        formatted = code_formatter.invoke({
            "code": response.content,
            "language": state.get("sub_type", "python").lower()
        })
        print(f"[代码生成链] 已生成代码")
        return {
            "response": formatted if "```" in formatted else response.content,
            "messages": [AIMessage(content=response.content)],
        }

    # ── 节点5：知识问答 Chain ─────────────────────
    def knowledge_qa_node(state: RouterState) -> dict:
        messages = KNOWLEDGE_QA_PROMPT.format_messages(
            user_input=state["user_input"],
        )
        response = llm.invoke(messages)
        print(f"[知识问答链] 已回答")
        return {
            "response": response.content,
            "messages": [AIMessage(content=response.content)],
        }

    # ── 节点6：未知意图 ───────────────────────────
    def unknown_node(state: RouterState) -> dict:
        messages = UNKNOWN_INTENT_PROMPT.format_messages(
            user_input=state["user_input"],
        )
        response = llm.invoke(messages)
        return {
            "response": response.content,
            "messages": [AIMessage(content=response.content)],
        }

    # ── 路由函数 ──────────────────────────────────
    def route_intent(state: RouterState) -> str:
        intent = state.get("intent", "unknown")
        route_map = {
            "text_creation":    "text_creation",
            "math_calculation": "math_tool",
            "code_generation":  "code_generation",
            "knowledge_qa":     "knowledge_qa",
        }
        target = route_map.get(intent, "unknown")
        print(f"[路由] {intent} → {target}")
        return target

    # ─────────────────────────────────────────────
    # 6. 构建图
    # ─────────────────────────────────────────────
    graph = StateGraph(RouterState)

    # 添加节点
    graph.add_node("intent",         intent_node)
    graph.add_node("text_creation",  text_creation_node)
    graph.add_node("math_tool",      math_tool_node)       # 工具调用
    graph.add_node("math_answer",    math_answer_node)     # 生成解答
    graph.add_node("code_generation",code_generation_node)
    graph.add_node("knowledge_qa",   knowledge_qa_node)
    graph.add_node("unknown",        unknown_node)

    # 入口
    graph.add_edge(START, "intent")

    # 条件路由
    graph.add_conditional_edges(
        "intent",
        route_intent,
        {
            "text_creation":  "text_creation",
            "math_tool":      "math_tool",
            "code_generation":"code_generation",
            "knowledge_qa":   "knowledge_qa",
            "unknown":        "unknown",
        }
    )

    # 数学链路的串行子图：工具 → 答案
    graph.add_edge("math_tool", "math_answer")

    # 所有出口 → END
    for node in ["text_creation", "math_answer", "code_generation",
                 "knowledge_qa", "unknown"]:
        graph.add_edge(node, END)

    return graph.compile()


# ─────────────────────────────────────────────
# 7. 运行器
# ─────────────────────────────────────────────

class IntentRouter:
    """意图路由器，封装 LangGraph 图"""

    def __init__(self, llm: ChatOpenAI):
        self.graph = build_intent_router_graph(llm)

    def run(self, user_input: str) -> str:
        """处理用户输入，返回对应链路的结果"""
        print(f"\n{'='*55}")
        print(f"用户输入：{user_input}")

        initial_state: RouterState = {
            "messages": [],
            "user_input": user_input,
            "intent": "",
            "sub_type": "",
            "key_content": "",
            "math_expr": "",
            "tool_result": "",
            "response": "",
        }

        result = self.graph.invoke(initial_state)
        response = result.get("response", "处理失败，请重试。")

        print(f"\n回复：{response[:200]}{'...' if len(response) > 200 else ''}")
        return response


# ─────────────────────────────────────────────
# 8. Demo 运行
# ─────────────────────────────────────────────

def run_demo():
    print("=" * 60)
    print("  意图识别 + 不同处理链路（LangChain + LangGraph）")
    print("=" * 60)

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
    )

    router = IntentRouter(llm)

    # 测试用例：覆盖所有意图分支
    test_cases = [
        "写一篇关于秋天的散文，要有意境",               # → 文本创作
        "帮我计算 (2**10 - 1) * 3 + sqrt(144)",         # → 数学计算（含工具）
        "计算 1 + 1 等于多少",                           # → 数学计算（简单）
        "用 Python 写一个快速排序算法",                  # → 代码生成
        "光速是多少？为什么黑洞的引力那么大？",          # → 知识问答
        "帮我订一张明天去上海的机票",                    # → 未知意图
    ]

    for query in test_cases:
        router.run(query)
        print()

    print("\n--- 交互式模式（输入 quit 退出）---")
    while True:
        try:
            user_input = input("\n你：").strip()
            if user_input.lower() in ("quit", "exit", "退出", "q"):
                print("再见！")
                break
            if not user_input:
                continue
            router.run(user_input)
        except KeyboardInterrupt:
            print("\n再见！")
            break


if __name__ == "__main__":
    run_demo()

