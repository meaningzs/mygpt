import glob
import settings
import streamlit as st

from langchain_core.messages.chat import ChatMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_teddynote.prompts import load_prompt

# API KEY 정보로드
#load_dotenv()

# python -m streamlit run main.py
st.title("📝나만의 GPT 만들기")
st.subheader("(🧸테디노트/TeddyNote님 코드 기반)")

st.session_state.api_key = settings.resolve_api_key()
if st.session_state.api_key:
    st.write(f'API키 : ...{st.session_state.api_key[-5:]}')
else:
    st.warning(
        "OpenAI API 키가 없습니다. 환경변수 OPENAI_API_KEY, "
        ".streamlit/secrets.toml 의 openai_api_key, 또는 .settings/config.json 의 api_key 중 하나를 설정하세요."
    )
main_text = st.empty()

# api_key = st.text_input("🔑 새로운 OPENAI API Key", type="password")
# save_btn = st.button("설정 저장", key="save_btn")

# if save_btn:
#    settings.save_config({"api_key": api_key})
#    st.session_state.api_key = api_key
#    st.write("설정이 저장되었습니다.")


# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성한다.
    st.session_state["messages"] = []

# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")

    # prompt_files = glob.glob("prompts/*.yaml")
    selected_prompt = st.selectbox(
        "프롬프트 선택",
        ["prompts/general.yaml", "prompts/prompt-maker.yaml"],
        index=0,
    )
    task_input = st.text_area("TASK 입력", "")


# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메시지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 체인 생성
def create_chain(prompt_filepath, task=""):
    # prompt 적용
    prompt = load_prompt(prompt_filepath, encoding="utf-8")

    # 추가 파라미터가 있으면 추가
    if task:
        prompt = prompt.partial(task=task)

    # GPT
    #llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0, api_key=st.session_state.api_key)

    # 출력 파서
    output_parser = StrOutputParser()

    # 체인 생성
    chain = prompt | llm | output_parser
    return chain


# 초기화 버튼이 눌리면...
if clear_btn:
    st.session_state["messages"] = []

# 이전 대화 기록 출력
print_messages()

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

# 만약에 사용자 입력이 들어오면...
if user_input and not st.session_state.api_key:
    st.error("API 키가 설정되지 않아 답변할 수 없습니다.")
elif user_input:
    # 사용자의 입력
    st.chat_message("user").write(user_input)
    # chain 을 생성
    chain = create_chain(selected_prompt, task=task_input)

    # 스트리밍 호출
    response = chain.stream({"question": user_input})
    with st.chat_message("assistant"):
        # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
        container = st.empty()

        ai_answer = ""
        for token in response:
            ai_answer += token
            container.markdown(ai_answer)

    # 대화기록을 저장한다.
    add_message("user", user_input)
    add_message("assistant", ai_answer)
