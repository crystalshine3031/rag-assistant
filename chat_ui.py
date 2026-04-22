import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/ask"

st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="🤖",
    layout="centered"
)

st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: white;
    }

    /* Header */
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: #9ca3af;
        font-size: 0.95rem;
    }

    /* Stats bar */
    .stats-bar {
        display: flex;
        justify-content: center;
        gap: 2rem;
        padding: 0.8rem;
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .stat-item {
        text-align: center;
    }
    .stat-number {
        font-size: 1.3rem;
        font-weight: 700;
        color: #a78bfa;
    }
    .stat-label {
        font-size: 0.7rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Chat messages */
    .user-bubble {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white;
        padding: 1rem 1.2rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.5rem 0 0.5rem 3rem;
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.3);
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .assistant-bubble {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.12);
        color: #e5e7eb;
        padding: 1rem 1.2rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.5rem 3rem 0.5rem 0;
        font-size: 0.95rem;
        line-height: 1.7;
    }
    .source-chip {
        display: inline-block;
        background: rgba(167, 139, 250, 0.15);
        border: 1px solid rgba(167, 139, 250, 0.3);
        color: #a78bfa;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        margin: 4px 3px 0 0;
    }
    .role-label {
        font-size: 0.75rem;
        color: #6b7280;
        margin-bottom: 4px;
        padding-left: 4px;
    }

    /* Input area */
    .stChatInput > div {
        background: rgba(255,255,255,0.07) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 12px !important;
    }
    .stChatInput input {
        color: white !important;
    }

    /* Spinner */
    .thinking {
        color: #a78bfa;
        font-size: 0.85rem;
        padding: 0.5rem;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Scrollbar */
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #4f46e5; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>RAG Knowledge Assistant</h1>
    <p>Ask anything about your ML textbook — powered by Groq + ChromaDB</p>
</div>
""", unsafe_allow_html=True)

# Stats bar
st.markdown("""
<div class="stats-bar">
    <div class="stat-item">
        <div class="stat-number">177</div>
        <div class="stat-label">Chunks indexed</div>
    </div>
    <div class="stat-item">
        <div class="stat-number">171</div>
        <div class="stat-label">Pages read</div>
    </div>
    <div class="stat-item">
        <div class="stat-number">Llama 3.1</div>
        <div class="stat-label">AI Model</div>
    </div>
    <div class="stat-item">
        <div class="stat-number">Live</div>
        <div class="stat-label">Status</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hi! I have read your entire ML textbook — all 171 pages. Ask me anything and I will explain it clearly with examples, just like ChatGPT would.",
        "sources": []
    })

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="role-label">You</div><div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="role-label">Assistant</div><div class="assistant-bubble">{msg["content"]}', unsafe_allow_html=True)
        if msg.get("sources"):
            chips = "".join([f'<span class="source-chip">Page {s["page"]}</span>' for s in msg["sources"]])
            st.markdown(f'{chips}</div>', unsafe_allow_html=True)
        else:
            st.markdown('</div>', unsafe_allow_html=True)

# Chat input
if question := st.chat_input("Ask about machine learning, algorithms, neural networks..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": question})
    st.markdown(f'<div class="role-label">You</div><div class="user-bubble">{question}</div>', unsafe_allow_html=True)

    # Get answer
    with st.spinner("Thinking..."):
        try:
            response = requests.post(
                API_URL,
                json={"question": question},
                timeout=30
            )
            data = response.json()

            if response.status_code == 200:
                answer = data["answer"]
                sources = data.get("sources", [])

                st.markdown(f'<div class="role-label">Assistant</div><div class="assistant-bubble">{answer}', unsafe_allow_html=True)
                if sources:
                    chips = "".join([f'<span class="source-chip">Page {s["page"]}</span>' for s in sources])
                    st.markdown(f'{chips}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('</div>', unsafe_allow_html=True)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
            else:
                st.error(f"API Error: {data.get('detail', 'Unknown error')}")

        except requests.exceptions.ConnectionError:
            st.error("FastAPI server not running. Start it with: uvicorn src.api.main:app --port 8000")
        except Exception as e:
            st.error(f"Error: {str(e)}")