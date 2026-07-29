import os
from datetime import datetime

import streamlit as st
from output_functions import render_output

LOG_PATH = "logs.txt"


def _read_recent_logs(limit=200):
    if not os.path.exists(LOG_PATH):
        return []
    # logs.txt may contain non-UTF-8 bytes on Windows; decode defensively.
    with open(LOG_PATH, "rb") as log_file:
        raw = log_file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("cp1252", errors="replace")
    lines = text.splitlines(keepends=True)
    return lines[-limit:]


def _append_chat_message(role, payload):
    st.session_state.chat_transcript.append(
        {
            "role": role,
            "payload": payload,
            "timestamp": datetime.now().strftime("%H:%M"),
        }
    )


def _render_chat_message(item):
    role = item.get("role", "assistant")
    payload = item.get("payload")
    timestamp = item.get("timestamp")
    avatar = ":material/person:" if role == "user" else ":material/smart_toy:"

    with st.chat_message(role, avatar=avatar):
        if role == "user":
            st.write(str(payload or ""))
            if timestamp:
                st.caption(timestamp)
            return

        if isinstance(payload, dict):
            render_output(payload)
            sources = payload.get("sources") if isinstance(payload.get("sources"), list) else []
            if sources:
                unique_sources = list(dict.fromkeys(str(source) for source in sources if source))
                if unique_sources:
                    st.markdown("---")
                    st.caption("Sources")
                    st.markdown(" · ".join(unique_sources))
        else:
            st.write(str(payload or ""))

        if timestamp:
            st.caption(timestamp)


def _chat_ui_header():
    st.title("Research assistant")
    st.caption("Messenger-style interface. Each question is answered independently from arXiv context only.")


def _starter_prompts_ui():
    prompts = {
        "Top 5 CS papers": "What are the top 5 papers for computer science?",
        "Recent AI papers": "What are the most recent papers on large language models?",
        "Paper summary by ID": "Tell me about paper 2406.15531",
        "Dataset used in a paper": "What dataset is used in paper 2406.15531?",
    }

    selected = st.pills(
        "Try a starter prompt",
        list(prompts.keys()),
        selection_mode="single",
    )
    if selected:
        st.session_state.pending_prompt = prompts[selected]
        st.rerun()


def main():
    from data_ingestion import init_db
    from orchestrator import handle_query

    init_db()
    st.set_page_config(page_title="Research Assistant", page_icon=":material/forum:", layout="wide")

    if "chat_transcript" not in st.session_state:
        st.session_state.chat_transcript = []
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None

    _chat_ui_header()

    with st.sidebar:
        st.subheader("Session")
        if st.button("Clear chat", width="stretch"):
            st.session_state.chat_transcript = []
            st.rerun()

        st.subheader("Logs")
        if os.path.exists(LOG_PATH):
            with open(LOG_PATH, "rb") as log_file:
                st.download_button(
                    "Download logs.txt",
                    data=log_file.read(),
                    file_name="logs.txt",
                    mime="text/plain",
                    width="stretch",
                )
        else:
            st.caption("No logs yet.")

    # Show all previous messages in this browser session.
    for item in st.session_state.chat_transcript:
        _render_chat_message(item)

    if not st.session_state.chat_transcript:
        st.info("Try asking: What are the top 5 papers for computer science?")
        _starter_prompts_ui()

    prompt = st.chat_input("Ask about papers, methods, datasets, or an arXiv ID")
    if not prompt and st.session_state.pending_prompt:
        prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None

    if prompt:
        _append_chat_message("user", prompt)
        with st.chat_message("user", avatar=":material/person:"):
            st.write(prompt)
            st.caption(datetime.now().strftime("%H:%M"))

        with st.chat_message("assistant", avatar=":material/smart_toy:"):
            with st.spinner("Searching and drafting answer..."):
                try:
                    # Intentional stateless call: every prompt is handled independently.
                    response = handle_query(prompt)
                except Exception as exc:
                    response = {
                        "messages": [{"type": "text", "content": f"Unable to process the query: {exc}"}],
                        "sources": [],
                        "meta": {},
                    }
                render_output(response)

                sources = response.get("sources") if isinstance(response.get("sources"), list) else []
                unique_sources = list(dict.fromkeys(str(source) for source in sources if source))
                if unique_sources:
                    st.markdown("---")
                    st.caption("Sources")
                    st.markdown(" · ".join(unique_sources))
                st.caption(datetime.now().strftime("%H:%M"))

        _append_chat_message("assistant", response)

    with st.expander("Recent logs", expanded=False):
        lines = _read_recent_logs()
        if not lines:
            st.caption("No logs yet.")
        else:
            st.code("".join(lines), language="text")


if __name__ == "__main__":
    main()
