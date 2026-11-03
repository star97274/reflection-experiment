import streamlit as st
import os
import csv
import io
import time
import requests
from datetime import datetime
from openai import OpenAI

# ==========================================
# 1. 配置区
# ==========================================
# 读取环境变量/Secrets
API_KEY = st.secrets.get("API_KEY", os.environ.get("API_KEY", ""))

# 大模型配置
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"

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

# ==========================================
# 2. 飞书 API 工具函数
# ==========================================
def get_feishu_token():
    """获取并缓存飞书 token，防止触发限流"""
    if "feishu_token" in st.session_state and st.session_state.get("feishu_token_expire", 0) > time.time():
        return st.session_state["feishu_token"]
    
    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    payload = {
        "app_id": st.secrets["FEISHU_APP_ID"],
        "app_secret": st.secrets["FEISHU_APP_SECRET"]
    }
    try:
        resp = requests.post(url, json=payload).json()
        token = resp.get("tenant_access_token")
        expire = resp.get("expire", 7200)
        
        # 提前 5 分钟过期，保证绝对安全
        st.session_state["feishu_token"] = token
        st.session_state["feishu_token_expire"] = time.time() + expire - 300
        return token
    except Exception as e:
        st.error(f"飞书鉴权失败，请检查 Secrets 配置: {e}")
        return None

def feishu_get_student_records(student_id, task_round=None):
    """从飞书拉取该学生的历史记录，按时间正序返回"""
    token = get_feishu_token()
    if not token: return []
    
    url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{st.secrets['FEISHU_APP_TOKEN']}/tables/{st.secrets['FEISHU_TABLE_ID']}/records/search"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    
    conditions = [{"field_name": "学号", "operator": "is", "value": [student_id]}]
    if task_round:
        conditions.append({"field_name": "任务轮次", "operator": "is", "value": [str(task_round)]})
    
    payload = {
        "filter": {"conjunction": "and", "conditions": conditions},
        "page_size": 500,
        "sort": [{"field_name": "时间戳", "desc": False}] # 按时间正序
    }
    
    all_messages = []
    while True:
        try:
            resp = requests.post(url, headers=headers, json=payload).json()
            data = resp.get("data", {})
            items = data.get("items", [])
            
            for item in items:
                f = item.get("fields", {})
                role = f.get("角色")
                content = f.get("内容")
                # 飞书有时会把多行文本包装成 list，这里做兼容处理
                if isinstance(role, list) and role: role = role[0].get("text", "")
                if isinstance(content, list) and content: content = content[0].get("text", "")
                
                if role and content:
                    all_messages.append({"role": str(role), "content": str(content)})
                    
            if not data.get("has_more"):
                break
            payload["page_token"] = data.get("page_token")
        except Exception as e:
            st.toast(f"拉取历史记录失败: {e}")
            break
            
    return all_messages

def feishu_add_record(student_id, group, task_round, role, content):
    """向飞书多维表格追加一条消息"""
    token = get_feishu_token()
    if not token: return
    
    url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{st.secrets['FEISHU_APP_TOKEN']}/tables/{st.secrets['FEISHU_TABLE_ID']}/records"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    
    payload = {
        "fields": {
            "学号": str(student_id),
            "组别": str(group),
            "任务轮次": str(task_round),
            "角色": str(role),
            "内容": str(content),
            "时间戳": int(time.time() * 1000) # 存入毫秒级时间戳
        }
    }
    try:
        resp = requests.post(url, headers=headers, json=payload)
        if resp.status_code != 200:
            st.toast(f"⚠️ 飞书同步失败: {resp.text}")
    except Exception as e:
        st.toast(f"⚠️ 网络异常，飞书同步失败: {e}")

def feishu_download_all_as_csv():
    """拉取飞书全部记录，生成 CSV（供研究者下载）"""
    token = get_feishu_token()
    if not token: return b""
    
    url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{st.secrets['FEISHU_APP_TOKEN']}/tables/{st.secrets['FEISHU_TABLE_ID']}/records/search"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    
    all_items = []
    page_token = None
    while True:
        payload = {"page_size": 500}
        if page_token: payload["page_token"] = page_token
        
        try:
            resp = requests.post(url, headers=headers, json=payload).json()
            data = resp.get("data", {})
            all_items.extend(data.get("items", []))
            if not data.get("has_more"): break
            page_token = data.get("page_token")
        except Exception:
            break

    output = io.StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_ALL)
    writer.writerow(["时间戳", "学号", "组别", "任务轮次", "角色", "内容"])
    
    for item in all_items:
        f = item.get("fields", {})
        # 兼容飞书返回的列表格式
        row = []
        for key in ["时间戳", "学号", "组别", "任务轮次", "角色", "内容"]:
            val = f.get(key, "")
            if isinstance(val, list) and val: val = val[0].get("text", "")
            row.append(val)
        writer.writerow(row)
        
    return output.getvalue().encode('utf-8-sig')


# ==========================================
# 3. 大模型调用工具
# ==========================================
def call_llm(messages, system_prompt):
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        timeout=60.0
    )
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=full_messages,
        stream=False
    )
    return response.choices[0].message.content


# ==========================================
# 4. 页面配置与初始化
# ==========================================
st.set_page_config(page_title="学习助手", page_icon="🤖", layout="centered")
st.title("🤖 学习助手")
st.caption("请在下方输入学号开始使用")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "student_id" not in st.session_state:
    st.session_state.student_id = None
if "group" not in st.session_state:
    st.session_state.group = None
if "task_round" not in st.session_state:
    st.session_state.task_round = 1

# ==========================================
# 5. 登录界面
# ==========================================
if st.session_state.student_id is None:
    with st.form("login_form"):
        student_id = st.text_input("请输入你的学号：", max_chars=20)
        submitted = st.form_submit_button("开始")
        if submitted and student_id.strip():
            st.session_state.student_id = student_id.strip()
            st.session_state.group = "实验组" if st.session_state.student_id in EXPERIMENTAL_GROUP else "对照组"
            # 从飞书加载历史
            with st.spinner("正在加载历史记录..."):
                st.session_state.messages = feishu_get_student_records(
                    st.session_state.student_id, st.session_state.task_round
                )
            st.rerun()
    st.stop()

# ==========================================
# 6. 主界面（已登录）
# ==========================================
group = st.session_state.group

with st.sidebar:
    st.header("📋 实验信息")
    st.write(f"**学号**：{st.session_state.student_id}")
    
    st.divider()
    st.subheader("任务轮次")
    task_round = st.selectbox(
        "当前进行到第几次任务：",
        options=[1, 2, 3],
        index=st.session_state.task_round - 1,
        format_func=lambda x: f"第 {x} 次任务",
        key="task_selector"
    )

    # 切换轮次时，重新从飞书拉取对应轮次的历史
    if task_round != st.session_state.task_round:
        st.session_state.task_round = task_round
        with st.spinner("切换任务，加载记录..."):
            st.session_state.messages = feishu_get_student_records(
                st.session_state.student_id, st.session_state.task_round
            )
        st.rerun()

    st.divider()
    st.subheader("操作")
    if st.button("🔄 清空当前页面显示"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("🔧 研究者工具")
    if st.button("📥 下载全部实验记录 (CSV)"):
        with st.spinner("正在从飞书提取所有数据..."):
            csv_data = feishu_download_all_as_csv()
            st.download_button(
                label="✅ 点击保存",
                data=csv_data,
                file_name=f"全部实验记录_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )

# 主区域
st.success(f"已登录，学号：{st.session_state.student_id}｜当前：第 {st.session_state.task_round} 次任务")

# 渲染历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 对话输入
if prompt := st.chat_input("输入你的消息..."):
    # 1. 显示并同步用户消息到飞书
    st.session_state.messages.append({"role": "user", "content": prompt})
    feishu_add_record(st.session_state.student_id, group, st.session_state.task_round, "user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 选择提示词
    system_prompt = SYSTEM_PROMPT_EXPERIMENTAL if group == "实验组" else SYSTEM_PROMPT_CONTROL

    # 3. 调用大模型
    with st.spinner("思考中..."):
        try:
            response = call_llm(st.session_state.messages, system_prompt)
        except Exception as e:
            response = f"⚠️ 调用失败：{str(e)}"
            st.error(response)

    # 4. 显示并同步 AI 回复到飞书
    st.session_state.messages.append({"role": "assistant", "content": response})
    feishu_add_record(st.session_state.student_id, group, st.session_state.task_round, "assistant", response)
    with st.chat_message("assistant"):
        st.markdown(response)
        
    st.rerun()