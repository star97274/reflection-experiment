import streamlit as st
import json
import os
import csv
import time
from datetime import datetime
from openai import OpenAI


DEEPSEEK_API_KEY = st.secrets.get("DEEPSEEK_API_KEY", os.environ.get("DEEPSEEK_API_KEY", ""))

# 实验组学号
EXPERIMENTAL_GROUP = ["11", "202431111038044", "", ""]

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
全程使用中文回复，语气友好但坚持引导学生思考。"""

SYSTEM_PROMPT_CONTROL = """你是一个友好的学习助手。学生会在任务中向你提问，请你直接、准确地回答问题，协助他们完成任务。"""


# 日志文件路径
LOG_FILE = "experiment_log.csv"

# ========== 工具函数 ==========
def get_group(student_id):
    """根据学号判断组别"""
    return "实验组" if student_id in EXPERIMENTAL_GROUP else "对照组"

def init_log():
    """初始化日志文件"""
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["时间戳", "学号", "组别", "任务轮次", "角色", "内容"])

def log_message(student_id, group, task_round, role, content):
    """记录一条消息到日志"""
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().isoformat(), student_id, group, task_round, role, content])

def call_deepseek(messages, system_prompt):
    """调用 DeepSeek API"""
    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com"
    )
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=full_messages,
        stream=False
    )
    return response.choices[0].message.content

# ========== 页面配置 ==========
st.set_page_config(page_title="学习助手", page_icon="🤖", layout="centered")
init_log()

st.title("🤖 学习助手")
st.caption("请在下方输入学号开始使用")

# ========== 会话状态初始化 ==========
if "messages" not in st.session_state:
    st.session_state.messages = []
if "student_id" not in st.session_state:
    st.session_state.student_id = None
if "task_round" not in st.session_state:
    st.session_state.task_round = 1

# ========== 学号输入 ==========
if st.session_state.student_id is None:
    with st.form("login_form"):
        student_id = st.text_input("请输入你的学号：", max_chars=20)
        submitted = st.form_submit_button("开始")
        if submitted and student_id.strip():
            st.session_state.student_id = student_id.strip()
            st.session_state.group = get_group(st.session_state.student_id)
            st.rerun()

# ========== 主界面 ==========
if st.session_state.student_id:
    group = st.session_state.group
    
    st.success(f"已登录，学号：{st.session_state.student_id}")
    
    # 任务轮次选择（学生可自行切换，或由研究者指定）
    task_round = st.selectbox(
        "当前任务轮次：",
        options=[1, 2, 3],
        format_func=lambda x: f"第 {x} 次任务",
        key="task_selector"
    )
    st.session_state.task_round = task_round
    
    # 显示历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # 输入框
    if prompt := st.chat_input("输入你的消息..."):
        # 记录用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        log_message(st.session_state.student_id, group, task_round, "user", prompt)
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 根据组别选择系统提示词
        if group == "实验组":
            system_prompt = SYSTEM_PROMPT_EXPERIMENTAL
        else:
            system_prompt = SYSTEM_PROMPT_CONTROL
        
        # 调用 API
        with st.spinner("思考中..."):
            try:
                response = call_deepseek(st.session_state.messages[:-1], system_prompt)
            except Exception as e:
                response = f"调用失败：{str(e)}"
        
        # 记录 AI 回复
        st.session_state.messages.append({"role": "assistant", "content": response})
        log_message(st.session_state.student_id, group, task_round, "assistant", response)
        
        with st.chat_message("assistant"):
            st.markdown(response)