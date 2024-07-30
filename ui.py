import streamlit as st
import requests
from streamlit.components.v1 import iframe

# FastAPI server URL
API_URL = "http://localhost:8000"
MLFLOW_URL = "http://localhost:8080"

# A dictionary to keep track of submitted tasks and their statuses
if "submitted_tasks" not in st.session_state:
    st.session_state["submitted_tasks"] = {}
if "submitted_pueue_tasks" not in st.session_state:
    st.session_state["submitted_pueue_tasks"] = {}

# Streamlit UI
st.title("MLFlow Training Task Manager")

train_tab, resume_tab = st.tabs(["Train", "Resume"])
with train_tab:
    # Training parameters input form
    st.header("Submit a Training Task")
    run_name = st.text_input("Run Name", value="", help="optional")
    exp_name = st.text_input("Experiment Name", value="", help="optional")
    learning_rate = st.number_input(
        "Learning Rate", min_value=0.0001, max_value=1.0, value=0.01, step=0.001
    )
    epochs = st.number_input(
        "Number of Epochs", min_value=1, max_value=100, value=10, step=1
    )
    gpu_id = st.number_input(
        "GPU ID (-1 for automatic allocation)",
        min_value=-1,
        max_value=10,
        value=-1,
        step=1,
    )

    # Prepare the payload
    payload = {
        "learning_rate": learning_rate,
        "epochs": epochs,
        "gpu_id": gpu_id,
        "run_name": run_name if run_name else None,
        "exp_name": exp_name if exp_name else None,
    }

    if st.button("Submit Training Task"):
        # Send POST request to submit the training task
        response = requests.post(f"{API_URL}/train", json=payload)
        if response.status_code == 200:
            response_data = response.json()
            run_id = response_data["run_id"]
            st.success(f"Training task has been submitted. Run ID: {run_id}")
            st.session_state["submitted_tasks"][run_id] = "pending"
        else:
            st.error("Failed to submit the training task.")

    if st.button("Submit Training Task using Pueue"):
        # Send POST request to submit the training task
        response = requests.post(
            f"{API_URL}/train", json=payload, params={"pueue": True}
        )
        if response.status_code == 200:
            response_data = response.json()
            task_id = response_data["task_id"]
            st.success(f"Training task has been submitted to Pueue. Task ID: {task_id}")
            st.session_state["submitted_pueue_tasks"][task_id] = requests.get(
                f"{API_URL}/pueue/running_status/{task_id}"
            ).json()
        else:
            st.error("Failed to submit the training task to Pueue.")

with resume_tab:
    # Resuming parameters input form
    st.header("Submit a Resuming Task")
    run_id = st.text_input("Run ID")

    # Prepare the payload
    payload = {
        "run_id": run_id,
    }

    if st.button("Submit Resuming Task"):

        # Send POST request to submit the resuming task
        response = requests.post(f"{API_URL}/resume", params=payload)
        if response.status_code == 200:
            response_data = response.json()
            run_id = response_data["run_id"]
            st.success(f"Training task has been resumed. Run ID: {run_id}")
            st.session_state["submitted_tasks"][run_id] = "pending"
        else:
            st.error(f"Failed to resume the training task. {response.json()}")

    if st.button("Submit Resuming Task using Pueue"):
        payload["pueue"] = True
        # Send POST request to submit the resuming task
        response = requests.post(f"{API_URL}/resume", params=payload)
        if response.status_code == 200:
            response_data = response.json()
            task_id = response_data["task_id"]
            st.success(
                f"Resuming task for run {run_id} has been submitted to Pueue. Task ID: {task_id}"
            )
            st.session_state["submitted_pueue_tasks"][task_id] = requests.get(
                f"{API_URL}/pueue/running_status/{task_id}"
            ).json()
        else:
            st.error("Failed to submit the resuming task to Pueue.")


# Display submitted tasks and their statuses
# Please replace st.experimental_fragment with st.fragment.
# st.experimental_fragment will be removed after 2025-01-01.
@st.fragment(run_every="30s")
def display_status():
    for run_id, status in st.session_state["submitted_tasks"].items():
        st.write(f"Run ID: {run_id}, Status: {status}")

        # Check the status of the task
        status_response = requests.get(f"{API_URL}/status/{run_id}")
        if status_response.status_code == 200:
            status_data = status_response.json()
            st.session_state["submitted_tasks"][run_id] = status_data["status"]
            st.write(status_data)
        else:
            st.error(f"Failed to retrieve the status of task {run_id}")


@st.fragment(run_every="30s")
def display_pueue_status():
    for task_id in st.session_state["submitted_pueue_tasks"].keys():
        running_status = requests.get(
            f"{API_URL}/pueue/running_status/{task_id}"
        ).json()
        st.session_state["submitted_pueue_tasks"][task_id] = running_status
        st.write(f"Task ID: {task_id}, Status: {running_status}")
        # Check the status of the task
        output = requests.get(
            f"{API_URL}/pueue/output/{task_id}",
        ).json()["output"]
        st.markdown(f"```\n{output}\n```", unsafe_allow_html=True)


st.header("Submitted Tasks")
fastapi_tab, pueue_tab = st.tabs(["FastAPI", "Pueue"])
with fastapi_tab:
    display_status()
with pueue_tab:
    display_pueue_status()

# Try to embed MLFlow
try:
    requests.head(MLFLOW_URL)
    mlflow_ui_running = True
except:
    mlflow_ui_running = False
if mlflow_ui_running:
    iframe(MLFLOW_URL, height=800, scrolling=True)
