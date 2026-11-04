import streamlit as st
import os
import csv
import io
from datetime import datetime
from openai import OpenAI

# ==========================================
# 1. 配置区
# ==========================================
# 从 Streamlit Secrets 读取，本地测试可用环境变量
API_KEY = st.secrets.get("API_KEY", os.environ.get("API_KEY", ""))

# 模型配置（deepseek）
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"

# 千问配置
# BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# MODEL_NAME = "qwen-plus"

# 实验组学号列表
EXPERIMENTAL_GROUP = ["11", "202431109011035", "202431109011090", "202431109011057", "202431109011072", "202431109011067", "202431109011027", "202431109011026", "202431109011096", "202431109011052", "202431109011085", "202431109011068", "202431109011031", "202431109011014", "202431109011108", "202431109011095", "202431109011106", "202431109011020", "202431109011064", "202431109011056", "202431109014058", "202431109014064", "202431109011079", "202431109011101", "202431109011073", "202431109011037", "202431109011081", "202431109014045", "202431109011013", "202431109011001", "202431109011051", "202431109011069", "202431109011049", "202431109014021", "202431109011002", "202431109011050", "202431109011060", "202431109011030", "202431109011058", "202431109011017", "202431109011078", "202431109011022", "202431109014036", "202431109011011", "202431109011107", "202431109011075", "202431109014050", "202431109011008", "202431109011047", "202431109011094", "202431109011055", "202431109011036", "202431109011034", "202431109011032", "202431109011023", "202431109011044", "202431109014030", "202231109010059", "202431109011065", "202431109011016", "202431109011105", "202431109011025", "202431109011089", "202431109011087", "202431109014068", "202431109011071", "202431109011024", "202431109011046", "202431109011040", "202431109014029", "202231109010008", "202431109011029", "202431109011066", "202431109011015", "202431109011062", "202431109014057", "202431109011006", "202431109011048", "202431109011028", "202431109014013", "202431109011086", "202431109011080", "202431109011054", "202431109014040", "202431109011045", "202431109011063", "202431109011018", "202431109011061", "202431109011012" ] 

# 系统提示词
SYSTEM_PROMPT_EXPERIMENTAL = """你是一名教育技术学领域的专家级学习引导助手。你的核心任务是：在GenAI支持的人机协同学习过程中，作为反思支架，引导学生对自己的思维过程进行深度反思，同时在必要时提供恰到好处的脚手架支持。
你的角色是苏格拉底式的提问者，兼具导师的辅助功能。请根据对话进度灵活切换【规划-认知-复盘】的引导策略。

【全局交互规则】
1. 弹性输出：默认情况下保持简练（尽量控制在100-200字左右），使用口语化、友好的大白话；但如果确实涉及复杂的概念解释或需要提供框架模板时，可以适当展开，但切忌写小作文。
2. 单点提问：每次回复原则上只提出【1个】核心问题。绝对禁止一次性抛出多个无关联的问题。
3. 支架递进：提问应从“表面描述”逐渐过渡到“原因分析”，最后上升到“经验迁移”。
4. 适度代劳和半成品支架：
   在讨论初期，不要直接给出完整答案，要引导学生自己思考。
   如果学生卡壳（如回复“不知道”、“没思路”），或者双方已经理清了大致大纲，你可以提供【半成品脚手架】。
   具体做法：不要给最终结果，而是提供2-3个不同的简易模板/思路框架/大纲方向供学生选择；或者给出一个带有填空提示的骨架（如：背景是___，核心冲突是___，解决策略是___），让学生在此基础上补充完善。

【分阶段引导框架】
你需要根据当前对话的进度，灵活执行以下三个阶段的引导策略：

阶段一：任务前规划支架（触发条件：学生刚开始提问或描述任务）
核心目标：促使学生在借助AI之前，先外显自己的初步构想。
执行策略：先肯定学生的提问，然后要求他不借助任何工具，先写下自己的第一直觉或初步想法。
示例语气：“这个切入点很有意思。在我们深入探讨具体方案之前，你脑海里最原始的想法是什么？”

阶段二：任务中认知支架（触发条件：学生提出了初步想法，或向你索要具体方案）
核心目标：引发认知冲突，引导学生对比、评判和修改自己的方案。
执行策略：
  1. 如果学生的想法有漏洞：不要直接批评，可以用假设性的反面例子推演，让他自己发现问题。
  2. 如果学生完全没有头绪、强烈要求给方案：启动【半成品脚手架】。告诉他：“如果觉得很难，我们可以从这几个方向来选：A、B、C，你觉得哪个最贴近你的想法？”，或者给他一个带提示语的填空题模板让他尝试补充。
  3. 当大纲已经理清：不要让他只停留在口头讨论。给出一个精简的内容模板或结构框架，要求他将刚才讨论的内容填进去，或者对他刚提出的方案进行书面解释。

阶段三：任务后复盘支架（触发条件：对话轮次较多，或学生表示任务已基本完成）
核心目标：引导学生对比初始想法和最终结果，归纳经验并促进迁移。
执行策略：
  1. 引导学生回顾：“对比你一开始的想法，现在的方案或思路发生了哪些主要变化？”
  2. 追问这些变化的原因，以及这些经验如何迁移到其他学习任务中。

【特殊情况处理】
如果学生表现出明显的烦躁或受挫（如：“太麻烦了”、“直接告诉我答案吧”）：请立即停止连环追问，先用温和的话语安抚情绪，然后直接给出一个基础的框架模板或几个可选方案，把最后一步的“决定权”留给学生。
全程使用中文回复，语气保持温和与鼓励。"""


SYSTEM_PROMPT_CONTROL = """你是一个学习助手。学生会在任务中向你提问，请你直接、准确地回答问题，协助他们完成任务。不要主动追问学生的感受或反思。"""

# 日志文件路径（仅供后台日志记录，学生端通过按钮下载）
LOG_FILE = "experiment_log.csv"


# ==========================================
# 2. 工具函数
# ==========================================
def get_group(student_id):
    """根据学号判断组别"""
    return "实验组" if student_id in EXPERIMENTAL_GROUP else "对照组"


def init_log():
    """初始化日志文件（含表头）"""
    if not os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["时间戳", "学号", "组别", "任务轮次", "角色", "内容"])
        except Exception as e:
            st.error(f"日志文件初始化失败：{e}")


def log_message(student_id, group, task_round, role, content):
    """追加一条消息到日志，带异常捕获防止并发写崩"""
    try:
        with open(LOG_FILE, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                student_id,
                group,
                task_round,
                role,
                content
            ])
    except Exception as e:
        st.toast(f"⚠️ 日志写入失败：{e}")


def call_llm(messages, system_prompt):
    """调用大模型 API，带超时和异常处理"""
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        timeout=60.0  # 60 秒超时
    )
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=full_messages,
        stream=False
    )
    return response.choices[0].message.content


def export_current_chat():
    """把当前会话消息导出为带 BOM 的标准 CSV，解决 Excel 乱码和列错位"""
    output = io.StringIO()
    # quoting=csv.QUOTE_ALL 确保内容里有逗号也不会错列
    writer = csv.writer(output, quoting=csv.QUOTE_ALL) 
    writer.writerow(["角色", "内容"])
    for msg in st.session_state.messages:
        writer.writerow([msg["role"], msg["content"]])
    # 返回带有 BOM 头的字节流，让 Excel 双击直接完美识别中文
    return output.getvalue().encode('utf-8-sig')


# ==========================================
# 3. 页面配置与初始化
# ==========================================
st.set_page_config(page_title="学习助手", page_icon="🤖", layout="centered")
init_log()

st.title("🤖 学习助手")
st.caption("请在下方输入学号开始使用")

# 会话状态初始化
if "messages" not in st.session_state:
    st.session_state.messages = []
if "student_id" not in st.session_state:
    st.session_state.student_id = None
if "group" not in st.session_state:
    st.session_state.group = None
if "task_round" not in st.session_state:
    st.session_state.task_round = 1
if "round_archived" not in st.session_state:
    st.session_state.round_archived = False
if "ready_to_download" not in st.session_state:
    st.session_state.ready_to_download = False


# ==========================================
# 4. 登录界面
# ==========================================
if st.session_state.student_id is None:
    with st.form("login_form"):
        student_id = st.text_input("请输入你的学号：", max_chars=20)
        submitted = st.form_submit_button("开始")
        if submitted and student_id.strip():
            st.session_state.student_id = student_id.strip()
            st.session_state.group = get_group(st.session_state.student_id)
            st.rerun()
    st.stop()  # 未登录时不再往下执行


# ==========================================
# 5. 主界面（已登录）
# ==========================================
group = st.session_state.group

# 侧边栏：显示信息和操作按钮
with st.sidebar:
    st.header("📋 实验信息")
    st.write(f"**学号**：{st.session_state.student_id}")
    # st.write(f"**组别**：{group}")

    st.divider()
    st.subheader("任务轮次")
    task_round = st.selectbox(
        "当前进行到第几次任务：",
        options=[1, 2, 3],
        index=st.session_state.task_round - 1,
        format_func=lambda x: f"第 {x} 次任务",
        key="task_selector"
    )

    # 如果切换了轮次，清空对话，开始新一轮
    if task_round != st.session_state.task_round:
        st.session_state.task_round = task_round
        st.session_state.messages = []
        st.session_state.round_archived = False
        st.session_state.ready_to_download = False
        st.rerun()

    st.divider()
    st.subheader("操作")
    if st.button("🔄 结束本轮并清空对话"):
        st.session_state.messages = []
        st.session_state.round_archived = False
        st.session_state.ready_to_download = False
        st.success("已清空当前对话，可开始新一轮。")
        st.rerun()

    # 先放一个普通按钮，点击后设置标记
    if st.button("📥 下载本次对话记录"):
        st.session_state.ready_to_download = True

    # 只有标记被触发后，才渲染真正的下载按钮
    if st.session_state.get("ready_to_download", False) and st.session_state.messages:
        st.download_button(
            label="✅ 点击保存 CSV 文件",
            data=export_current_chat(),
            file_name=f"{st.session_state.student_id}_第{st.session_state.task_round}次任务.csv",
            mime="text/csv",
            key="final_download"
        )
        st.caption("下载完成后，可继续对话或切换任务轮次。")

    st.divider()
    st.caption("提示：完成一次任务后，请先下载记录，再切换下一轮。")


# 主区域
st.success(f"已登录，学号：{st.session_state.student_id}｜当前：第 {st.session_state.task_round} 次任务")

# 渲染历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 输入框
if prompt := st.chat_input("输入你的消息..."):
    # 1. 记录并显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    log_message(st.session_state.student_id, group, st.session_state.task_round, "user", prompt)

    # 2. 选择系统提示词
    if group == "实验组":
        system_prompt = SYSTEM_PROMPT_EXPERIMENTAL
    else:
        system_prompt = SYSTEM_PROMPT_CONTROL

    # 3. 调用 API（传入完整历史，修复丢消息 bug）
    with st.spinner("思考中..."):
        try:
            response = call_llm(st.session_state.messages, system_prompt)
        except Exception as e:
            response = f"⚠️ 调用失败，请检查网络或联系管理员。\n\n错误信息：{str(e)}"
            st.error(response)

    # 4. 记录 AI 回复
    st.session_state.messages.append({"role": "assistant", "content": response})
    log_message(st.session_state.student_id, group, st.session_state.task_round, "assistant", response)
    
    # 5. 重置下载状态，并强制重跑页面，确保下载按钮拿到完整的最新对话
    st.session_state.ready_to_download = False  
    st.rerun() 