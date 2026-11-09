import json
import os

SETTING_DIR = "./.settings"


def load_config(filename="config.json"):
    directory = os.path.dirname(SETTING_DIR)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    try:
        filename = os.path.join(SETTING_DIR, filename)
        with open(filename, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"{filename} 파일이 존재하지 않습니다.")
        return {}


def save_config(config, filename="config.json"):
    directory = os.path.dirname(SETTING_DIR)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    current_config = load_config(filename)
    updated_config = {**current_config, **config}

    filename = os.path.join(SETTING_DIR, filename)
    with open(filename, "w") as file:
        json.dump(updated_config, file, indent=4)


def resolve_api_key():
    """OpenAI API 키를 우선순위대로 찾는다.

    1) .settings/config.json 의 api_key (사용자 저장값)
    2) 환경변수 OPENAI_API_KEY (서버/컨테이너 운영용)
    3) .streamlit/secrets.toml 의 openai_api_key
    없으면 빈 문자열을 반환한다.
    """
    config = load_config()
    if config.get("api_key"):
        return config["api_key"]

    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]

    try:
        import streamlit as st

        return st.secrets.get("openai_api_key", "") or ""
    except Exception:
        # secrets.toml 이 없으면 Streamlit 이 예외를 던진다.
        return ""
