import streamlit as st


def _table_like(content):
    if isinstance(content, list):
        return content
    if isinstance(content, dict):
        return [content]
    return [{"value": str(content)}]

def render_message(message):
    if isinstance(message, str):
        st.write(message)
        return

    if not isinstance(message, dict):
        st.write(message)
        return

    msg_type = message.get("type", "text")
    content = message.get("content", "")
    subtype = message.get("subtype", None)
    title = message.get("title", None)
    x_label = message.get("x_label", None)
    y_label = message.get("y_label", None)

    if isinstance(content, dict):
        if "content" in content:
            content = content["content"]
        else:
            content = str(content)
    elif isinstance(content, list):
        content = "\n".join(str(item) for item in content)

    if msg_type == "text":
        if title:
            st.subheader(title)
        st.write(content)

    elif msg_type == "table":
        try:
            if title:
                st.subheader(title)
            st.table(_table_like(content))
        except Exception:
            st.write("Invalid table format:", content)

    elif msg_type == "graph":
        try:
            data = _table_like(content)
            if title:
                st.subheader(title)

            if subtype == "bar":
                st.bar_chart(data)
            elif subtype == "scatter":
                st.scatter_chart(data)
            else:
                st.line_chart(data)

            if x_label or y_label:
                st.caption(f"{x_label or ''} vs {y_label or ''}")
        except Exception:
            st.write("Invalid graph format:", content)

    elif msg_type == "list":
        if isinstance(content, list):
            if title:
                st.subheader(title)
            for item in content:
                st.markdown(f"- {item}")
        else:
            st.write("Invalid list format:", content)

    elif msg_type == "citations":
        citations = content
        if isinstance(content, dict):
            if "items" in content:
                citations = content["items"]
            else:
                citations = list(content.values())

        rendered = []
        if isinstance(citations, dict):
            citations = list(citations.values())

        if isinstance(citations, list):
            for item in citations:
                if isinstance(item, dict):
                    rendered.append(item.get("citation") or item.get("text") or str(item))
                else:
                    rendered.append(str(item))
        else:
            rendered = [str(citations)]

        if title:
            st.subheader(title)
        st.write("Sources:")
        for citation in rendered:
            st.markdown(f"- {citation}")

    elif msg_type == "code":
        if title:
            st.subheader(title)
        st.code(content, language=subtype if subtype else "python")

    else:
        st.write("Unknown message type:", msg_type, content)


def render_output(json_response):
    messages = []
    if isinstance(json_response, dict):
        if isinstance(json_response.get("messages"), list):
            messages = json_response.get("messages", [])
        elif isinstance(json_response.get("content"), list):
            messages = json_response.get("content", [])
        elif json_response.get("content"):
            messages = [{"type": "text", "content": json_response.get("content")}] 
        elif json_response.get("answer"):
            messages = [{"type": "text", "content": json_response.get("answer")}] 
    elif isinstance(json_response, list):
        messages = json_response

    if not messages:
        st.write("No output available")
        return

    for msg in messages:
        if isinstance(msg, dict) and "messages" in msg and isinstance(msg["messages"], list):
            for nested in msg["messages"]:
                render_message(nested)
        else:
            render_message(msg)
