import streamlit as st
import os
import csv
from datetime import datetime
from openai import OpenAI

if "ready_to_download" not in st.session_state:
    st.session_state.ready_to_download = False

# ==========================================
#  配置区
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
EXPERIMENTAL_GROUP = ["11", "", ""]

# 系统提示词
SYSTEM_PROMPT_EXPERIMENTAL = """你是一个学习引导助手。你的任务不是直接给学生答案，而是引导他们深入思考。
请严格遵守以下规则：
【规划阶段】
当学生首次提出问题时，先引导他们自己思考：
1. 请学生先写下自己的初步想法（不借助任何工具）
2. 然后针对他们的想法，提出2-3个引导性问题，帮助他们完善思路
3. 不要直接给出完整方案
【认知阶段】
在学生有了初步思路后：
1. 提供2-3个不同的方案思路，每个方案说明其优缺点
2. 要求学生比较这些方案，选择最适合的一个，并说明理由
3. 如果学生的选择明显不合理，温和地指出并引导他们重新思考
【复盘阶段】
在讨论接近结尾时：
1. 请学生回顾自己最初的方案和最终方案，找出主要变化
2. 追问这些变化的原因
3. 引导学生思考：这次学到的方法可以怎样用到其他学习任务中
全程使用中文回复，语气友好但坚持引导学生思考。
"""

SYSTEM_PROMPT_CONTROL = """你是一个学习助手。学生会在任务中向你提问，请你直接、准确地回答问题，协助他们完成任务。不要主动追问学生的感受或反思。"""

# 日志文件路径（Streamlit Cloud 重启可能丢失，建议学生用下载按钮备份）
LOG_FILE = "experiment_log.csv"


# ==========================================
# 工具函数
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
        # 不中断实验，只在页面角落提示
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
    """把当前会话消息导出为 CSV 字符串（供下载）"""
    rows = [["角色", "内容"]]
    for msg in st.session_state.messages:
        rows.append([msg["role"], msg["content"]])
    csv_str = "\n".join([",".join([f'"{c}"' for c in row]) for row in rows])
    return csv_str


# ==========================================
# 页面配置与初始化
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


# ==========================================
# 登录界面
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
# 主界面（已登录）
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
        st.rerun()

    st.divider()
    st.subheader("操作")
    if st.button("🔄 结束本轮并清空对话"):
        st.session_state.messages = []
        st.session_state.round_archived = False
        st.success("已清空当前对话，可开始新一轮。")
        st.rerun()

   # 先放一个普通按钮，点击后设置标记
    if st.button("📥 下载本次对话记录"):
        st.session_state.ready_to_download = True

    # 只有标记被触发后，才渲染真正的下载按钮
    # 此时 session_state.messages 已经包含全部完整对话
    if st.session_state.get("ready_to_download", False) and st.session_state.messages:
        st.download_button(
            label="✅ 点击保存 CSV 文件",
            data=export_current_chat(),
            file_name=f"{st.session_state.student_id}_第{st.session_state.task_round}次任务.csv",
            mime="text/csv",
            key="final_download"  # 给一个固定 key，避免重复渲染冲突
        )
        st.caption("下载完成后，可继续对话或切换任务轮次。")

    st.divider()
    st.caption("提示：完成一次任务后，请在左侧切换到下一轮。")


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

    with st.chat_message("user"):
        st.markdown(prompt)

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

    # 4. 记录并显示 AI 回复
    st.session_state.messages.append({"role": "assistant", "content": response})
    log_message(st.session_state.student_id, group, st.session_state.task_round, "assistant", response)

    with st.chat_message("assistant"):
        st.markdown(response)
    # AI 回复已追加到 messages 后
    st.session_state.ready_to_download = False  # 新消息产生，旧的下载准备作废