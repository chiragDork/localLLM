import streamlit as st
import requests
import json
from datetime import datetime
from openai import OpenAI

# --- OpenAI Client ---
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- Local validation of user prompt ---
def validate_prompt_with_local(prompt, model, base_url):
    check_prompt = (
        f"You are a data compliance assistant.\n"
        f"Does the following prompt contain any PII, PHI, confidential, proprietary, or internal information? "
        f"Is it safe to send this prompt to a cloud-based LLM provider? Answer only YES or NO and explain.\n\n"
        f"Prompt:\n{prompt}"
    )
    try:
        response = requests.post(f"{base_url}/api/generate", json={
            "model": model,
            "prompt": check_prompt,
            "stream": False
        }, timeout=60)
        return response.json().get("response", "[No response from local model]")
    except Exception as e:
        return f"[Local validation error: {e}]"

# --- Validate cloud response locally ---
def ask_local(prompt, model, base_url):
    try:
        response = requests.post(f"{base_url}/api/generate", json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }, timeout=60)
        return response.json().get("response", "[No response from local model]")
    except Exception as e:
        return f"[Error calling local LLM: {e}]"

# --- Cloud LLM ---
def ask_cloud(query):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": query}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Error calling cloud LLM: {e}]"

# --- Logging ---
def audit_log(user_input, cloud_resp, local_resp):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "input": user_input,
        "cloud_response": cloud_resp,
        "local_validation": local_resp
    }
    with open("audit_log.json", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

# --- Streamlit UI ---
st.set_page_config(page_title="Hybrid LLM Assistant", layout="centered")
st.title("🔐 Hybrid LLM Assistant")
st.caption("Prompt is reviewed by a local model before being sent to the cloud.")

# Local model config
user_local_url = st.text_input("🖥️ Enter your local model's base URL (e.g., http://localhost:11434 or ngrok HTTPS URL):")
available_models = ["mistral", "llama3", "phi3", "custom"]
model_choice = st.selectbox("🧠 Choose your local model:", available_models)
LOCAL_MODEL = model_choice if model_choice != "custom" else st.text_input("Custom model name (e.g. my-model):")

query = st.text_input("💬 Ask a question (no sensitive info):")

# Submit
if st.button("Submit"):
    if not query.strip():
        st.warning("Please enter a question.")
        st.stop()
    if not user_local_url or not LOCAL_MODEL:
        st.warning("Please enter both model name and local model URL.")
        st.stop()

    # Step 1: Ask local model to approve prompt
    with st.spinner("🧠 Validating prompt using local LLM..."):
        validation_feedback = validate_prompt_with_local(query, LOCAL_MODEL, user_local_url)
        st.info(f"📝 Local Model Review:\n\n{validation_feedback}")

    if "no" in validation_feedback.lower():
        st.error("🚫 Local model flagged this prompt as unsafe for cloud.")
        st.stop()

    # Step 2: Send to cloud
    with st.spinner("🌐 Getting response from OpenAI..."):
        cloud_response = ask_cloud(query)

    # Step 3: Validate cloud response locally
    with st.spinner("🔍 Validating cloud response locally..."):
        review_prompt = (
            f"You are a regulatory compliance expert. Review the following cloud-generated answer:\n\n"
            f"{cloud_response}\n\n"
            f"Is it accurate, compliant, and free of hallucinations or misstatements?"
        )
        local_review = ask_local(review_prompt, LOCAL_MODEL, user_local_url)

    # Show results
    st.subheader("🌐 Cloud Response")
    st.write(cloud_response)

    st.subheader(f"🛡️ Local Model Review ({LOCAL_MODEL})")
    st.write(local_review)

    audit_log(query, cloud_response, local_review)
