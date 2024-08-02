import streamlit as st
import requests
from streamlit.components.v1 import iframe
import utils
from train import TrainArgs
from cli import ResumeArgs
from pueue import pueue_submit, get_pueue_task_status


# Streamlit UI
st.set_page_config("MLFlow Training Task Manager (Local Pueue)", layout="wide")
st.title("MLFlow Training Task Manager (Local Pueue)")

MLFLOW_URL = st.text_input(
    "MLFlow URL",
    "http://localhost:8080",
    help="If you are not accessing MLFlow from local and using localhost, since Pueue runs on same machine so it will be fine, but the iframe at the bottom might be wrong.",
)
mlflow_url_is_tracking_server = st.checkbox(
    "MLFlow URL is tracking server",
    False,
    help="If not, will use default one. i.e. `MLFLOW_TRACKING_URI` if not set will be `./mlruns`",
)
pueue_group = st.text_input(
    "Pueue Experiment Group",
    "",
    help="Pueue group that will share job queue and parallelism config",
)
pueue_parallel_num = st.number_input(
    "Pueue Parallel Number",
    utils.TorchDeviceManager.get_gpu_number(default=1),
    help="Default is the GPU number (if no GPU found then default to 1).",
)
pueue_dry_run = st.checkbox(
    "Pueue Dry Run", help="If enable, we won't actually send task to Pueue."
)


# A dictionary to keep track of submitted tasks and their statuses
if "submitted_pueue_tasks" not in st.session_state:
    st.session_state["submitted_pueue_tasks"] = {}


train_tab, resume_tab = st.tabs(["Train", "Resume"])
with train_tab:
    # Training parameters input form
    st.header("Submit a Training Task")

    inputs, empty_args = utils.create_streamlit_ui(TrainArgs)

    train_args = TrainArgs().from_dict(inputs)
    with st.expander("Parsed Training Argument"):
        st.write(train_args)

    if st.button("Submit Training Task using Local Pueue"):
        task_id = pueue_submit(
            train_args,
            pueue_group,
            pueue_parallel_num,
            pueue_return_task_id_only=True,
            dry_run=pueue_dry_run,
            extra_submit_env=(
                dict(MLFLOW_TRACKING_URI=MLFLOW_URL)
                if mlflow_url_is_tracking_server
                else None
            ),
        )

        if pueue_dry_run:
            st.text("Command:")
            st.markdown(f"```bash\n{task_id}\n```")
        else:
            st.session_state["submitted_pueue_tasks"][task_id] = get_pueue_task_status(
                "running_status", task_id
            )


with resume_tab:
    # Resuming parameters input form
    st.header("Submit a Resuming Task")

    inputs, empty_args = utils.create_streamlit_ui(ResumeArgs)

    resume_args = ResumeArgs().from_dict(inputs)
    with st.expander("Parsed Resume Argument"):
        st.write(resume_args)

    if st.button("Submit Resume Task using Local Pueue"):
        task_id = pueue_submit(
            resume_args,
            pueue_group,
            pueue_parallel_num,
            pueue_return_task_id_only=True,
            dry_run=pueue_dry_run,
            extra_submit_env=(
                dict(MLFLOW_TRACKING_URI=MLFLOW_URL)
                if mlflow_url_is_tracking_server
                else None
            ),
        )

        if pueue_dry_run:
            st.text("Command:")
            st.markdown(f"```bash\n{task_id}\n```")
        else:
            st.session_state["submitted_pueue_tasks"][task_id] = get_pueue_task_status(
                "running_status", task_id
            )


@st.fragment(run_every="30s")
def display_pueue_status():
    to_remove = []
    for task_id in st.session_state["submitted_pueue_tasks"].keys():
        try:
            running_status = get_pueue_task_status("running_status", task_id)
            st.session_state["submitted_pueue_tasks"][task_id] = running_status
            st.write(f"Task ID: {task_id}, Status: {running_status}")
            # Check the status of the task
            # NOTE: Won't have output when the task has not run (e.g. Queued)
            output = get_pueue_task_status("output", task_id)["output"]
            st.markdown(f"```\n{output}\n```", unsafe_allow_html=True)
        except:
            to_remove.append(task_id)
    for task_id in to_remove:
        del st.session_state["submitted_pueue_tasks"][task_id]
        st.toast(f"Remove invalid task_id {task_id}")


st.header("Submitted Tasks")
display_pueue_status()


@st.fragment()
def display_mlflow_ui():
    # Try to embed MLFlow
    try:
        requests.head(MLFLOW_URL)
        mlflow_ui_running = True
    except:
        mlflow_ui_running = False
    if mlflow_ui_running:
        iframe(MLFLOW_URL, height=800, scrolling=True)


display_mlflow_ui()
