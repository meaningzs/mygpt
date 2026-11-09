# 📝 나만의 GPT 만들기 (mygpt)

LangChain + OpenAI 기반 Streamlit 챗 앱입니다. (테디노트 코드 기반)

| 페이지 | 설명 |
| --- | --- |
| `main.py` | 프롬프트 선택형 챗봇 |
| `pages/00_HWP.py` | HWP 문서 Q&A (RAG) |
| `pages/01_PDF.py` | PDF 문서 Q&A (RAG) |
| `pages/02_Multi_Modal.py` | 이미지 멀티모달 |
| `pages/03_Database.py` | Google Sheets 연결 (secrets.toml 필요) |

## 서버 운영

### API 키 설정

다음 중 하나로 주입합니다. 우선순위는 위에서부터입니다. **키는 절대 커밋하지 마세요.**

1. `.settings/config.json` 의 `api_key`
2. 환경변수 `OPENAI_API_KEY` (서버/컨테이너 운영 권장)
3. `.streamlit/secrets.toml` 의 `openai_api_key` (`example.secrets.toml` 참고)

키가 없어도 서버는 뜨고, 화면에 경고만 표시됩니다.

### 1) Docker Compose (권장)

```bash
echo "OPENAI_API_KEY=sk-..." > .env      # .gitignore 에 포함됨
docker compose up -d --build
# http://<서버IP>:8501
```

- 포트 변경: `PORT=9000 docker compose up -d`
- 상태 확인: `docker ps` (HEALTHCHECK → `healthy`), `curl localhost:8501/_stcore/health` → `ok`
- 로그: `docker compose logs -f`
- Google Sheets 페이지를 쓰려면 `.streamlit/secrets.toml` 을 만든 뒤 `docker-compose.yml` 의 `volumes` 주석을 해제하세요.

### 2) Docker 없이 (venv)

```bash
OPENAI_API_KEY=sk-... PORT=8501 ./serve.sh
```

처음 실행 시 `.venv` 를 만들고 의존성을 설치합니다. Python 3.11 권장.
(Debian/Ubuntu 시스템 Python 3.12+ 에서는 `pandasql` 빌드가 실패할 수 있으니 3.11 venv 를 쓰세요.)

### 3) 개발

```bash
python -m streamlit run main.py
```

`.devcontainer` 로 GitHub Codespaces 에서도 바로 실행됩니다.
