import streamlit as st
import requests

# FastAPI server URL
API_URL = "http://localhost:8000"

# A dictionary to keep track of submitted tasks and their statuses
if "submitted_tasks" not in st.session_state:
    st.session_state["submitted_tasks"] = {}

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

    if st.button("Submit Training Task"):
        # Prepare the payload
        payload = {
            "learning_rate": learning_rate,
            "epochs": epochs,
            "gpu_id": gpu_id,
            "run_name": run_name if run_name else None,
            "exp_name": exp_name if exp_name else None,
        }

        # Send POST request to submit the training task
        response = requests.post(f"{API_URL}/train", json=payload)
        if response.status_code == 200:
            response_data = response.json()
            run_id = response_data["run_id"]
            st.success(f"Training task has been submitted. Run ID: {run_id}")
            st.session_state["submitted_tasks"][run_id] = "pending"
        else:
            st.error("Failed to submit the training task.")

with resume_tab:
    # Resuming parameters input form
    st.header("Submit a Resuming Task")
    run_id = st.text_input("Run ID")
    if st.button("Submit Resuming Task"):
        # Prepare the payload
        payload = {
            "run_id": run_id,
        }

        # Send POST request to submit the training task
        response = requests.post(f"{API_URL}/train", params=payload)
        if response.status_code == 200:
            response_data = response.json()
            run_id = response_data["run_id"]
            st.success(f"Training task has been resumed. Run ID: {run_id}")
            st.session_state["submitted_tasks"][run_id] = "pending"
        else:
            st.error("Failed to resume the training task.")


# Display submitted tasks and their statuses
@st.experimental_fragment(run_every="30s")
def display_status():
    st.header("Submitted Tasks")
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


display_status()
