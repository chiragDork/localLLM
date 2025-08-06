import streamlit as st
from openai import OpenAI
import subprocess
import json
from datetime import datetime

# --- OpenAI Client ---
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
  # Replace with your actual key

# --- Local validation of user prompt (Advanced) ---
def validate_prompt_with_local(prompt, model):
    check_prompt = (
        f"You are a data compliance assistant.\n"
        f"Does the following prompt contain any PII, PHI, confidential, proprietary, or internal information? "
        f"Is it safe to send this prompt to a cloud-based LLM provider? Answer only YES or NO and explain.\n\n"
        f"Prompt:\n{prompt}"
    )
    try:
        result = subprocess.run(['ollama', 'run', model], input=check_prompt.encode(), stdout=subprocess.PIPE)
        return result.stdout.decode().strip()
    except Exception as e:
        return f"[Local validation error: {e}]"

# --- Validate cloud response locally ---
def ask_local(prompt, model_name):
    try:
        result = subprocess.run(['ollama', 'run', model_name], input=prompt.encode(), stdout=subprocess.PIPE)
        return result.stdout.decode().strip()
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

query = st.text_input("💬 Ask a question (no sensitive info):")

# Local model config
available_models = ["mistral", "llama3", "phi3", "custom"]
model_choice = st.selectbox("🧠 Choose your local model (used for pre-check and validation):", available_models)
LOCAL_MODEL = model_choice if model_choice != "custom" else st.text_input("Custom model name (e.g. my-model):")

# Model availability check
if LOCAL_MODEL:
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if LOCAL_MODEL not in result.stdout:
            st.warning(f"⚠️ Model '{LOCAL_MODEL}' not installed.")
            st.code(f"ollama pull {LOCAL_MODEL}", language="bash")
            st.stop()
    except Exception as e:
        st.error(f"⚠️ Model check failed: {str(e)}")
        st.stop()

# Submit
if st.button("Submit"):
    if query.strip() == "":
        st.warning("Please enter a question.")
        st.stop()

    # Step 1: Ask local model to approve prompt
    with st.spinner("🧠 Validating prompt using local LLM..."):
        validation_feedback = validate_prompt_with_local(query, LOCAL_MODEL)
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
        local_review = ask_local(review_prompt, LOCAL_MODEL)

    # Show results
    st.subheader("🌐 Cloud Response")
    st.write(cloud_response)

    st.subheader(f"🛡️ Local Model Review ({LOCAL_MODEL})")
    st.write(local_review)

    audit_log(query, cloud_response, local_review)
