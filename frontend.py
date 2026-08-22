"""Streamlit frontend for the Gradient-Rush multimodal RAG API."""

from pathlib import Path
from typing import Any

import requests
import streamlit as st


BACKEND_URL = "http://localhost:8000"
UPLOAD_ENDPOINTS = {
    ".mp4": "/upload/video",
    ".pdf": "/upload/pdf",
    ".png": "/upload/image",
    ".jpg": "/upload/image",
}


st.set_page_config(
    page_title="Gradient-Rush: Multimodal RAG Pipeline",
    page_icon="🧠",
    layout="wide",
)

st.title("Gradient-Rush: Multimodal RAG Pipeline")
st.caption("Search across transcripts, visual evidence, and document context.")


def _error_detail(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text or f"HTTP {response.status_code}"
    return str(payload.get("detail", payload)) if isinstance(payload, dict) else str(payload)


def _upload_source(uploaded_file: Any) -> None:
    extension = Path(uploaded_file.name).suffix.lower()
    endpoint = UPLOAD_ENDPOINTS.get(extension)
    if endpoint is None:
        st.error("Unsupported file type.")
        return

    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "application/octet-stream",
        )
    }
    try:
        with st.spinner("Processing and indexing your knowledge source..."):
            response = requests.post(
                f"{BACKEND_URL}{endpoint}",
                files=files,
                timeout=600,
            )
        if not response.ok:
            st.error(f"Upload failed: {_error_detail(response)}")
            return
        result = response.json()
        indexed_count = result.get("indexed_count", result.get("processed_nodes", 0))
        st.success(f"Indexed {indexed_count} KnowledgeNode(s).")
        st.toast("Knowledge source indexed successfully.")
    except requests.RequestException as exc:
        st.error(f"Could not reach the FastAPI backend: {exc}")
    except ValueError:
        st.error("The backend returned an invalid upload response.")


def _display_result(result: dict[str, Any] | None, *, multimodal: bool) -> None:
    if not result:
        st.info("No result found.")
        return

    if multimodal:
        st.markdown("**Spoken Transcript**")
        st.write(
            result.get("transcript")
            or result.get("content")
            or "No transcript available."
        )
        st.markdown("**Visual Summary**")
        st.write(result.get("visual_summary") or "No visual summary available.")
        st.markdown("**Timestamp / Location**")
        st.write(result.get("timestamp") or "Not available.")

        frame_path = result.get("frame_path")
        frame_file = _frame_file_path(frame_path)
        if frame_file is not None and frame_file.is_file():
            frame_url = _frame_url(frame_path, frame_file)
            try:
                st.image(frame_url, caption="Extracted visual evidence", width="stretch")
            except Exception:
                st.caption(f"Frame preview unavailable: {frame_path}")
    else:
        st.markdown("**Raw Spoken Transcript**")
        st.write(
            result.get("transcript")
            or result.get("content")
            or "No transcript available."
        )
        if not result.get("visual_summary"):
            st.warning("Visual evidence missed by text-only search!")


def _frame_file_path(frame_path: Any) -> Path | None:
    """Resolve a backend frame reference to a local file before rendering it."""
    if frame_path is None or str(frame_path).strip() in {"", "0"}:
        return None
    reference = str(frame_path).strip()
    if reference.startswith("/frames/"):
        return Path("data") / reference.lstrip("/")
    candidate = Path(reference)
    return candidate if candidate.is_file() else None


def _frame_url(frame_path: Any, frame_file: Path) -> str:
    """Build the browser URL for a verified local frame artifact."""
    reference = str(frame_path).strip()
    if reference.startswith("/frames/"):
        return f"{BACKEND_URL}{reference}"
    return f"{BACKEND_URL}/frames/{frame_file.name}"


with st.sidebar:
    st.header("1. Upload Knowledge Source")
    uploaded_file = st.file_uploader(
        "Choose a source file",
        type=["mp4", "pdf", "png", "jpg"],
        help="Upload video, PDF, PNG, or JPG knowledge sources.",
    )
    if uploaded_file is not None:
        upload_key = f"{uploaded_file.name}:{uploaded_file.size}"
        if st.session_state.get("uploaded_key") != upload_key:
            st.session_state["uploaded_key"] = upload_key
            _upload_source(uploaded_file)

st.header("2. Query & Baseline Comparison")
query = st.text_input(
    "Search question",
    placeholder="What database architecture was discussed?",
)
if st.button("Search & Compare", type="primary", use_container_width=True):
    if not query.strip():
        st.warning("Enter a search question first.")
    else:
        try:
            with st.spinner("Searching multimodal and text-only indexes..."):
                response = requests.post(
                    f"{BACKEND_URL}/query/compare",
                    json={"query": query, "limit": 5},
                    timeout=120,
                )
            if not response.ok:
                st.error(f"Search failed: {_error_detail(response)}")
            else:
                comparison = response.json()
                left_column, right_column = st.columns(2)
                with left_column:
                    st.subheader("✨ Multimodal RAG Result")
                    _display_result(comparison.get("multimodal_result"), multimodal=True)
                with right_column:
                    st.subheader("⚠️ Text-Only Baseline Result")
                    _display_result(
                        comparison.get("text_only_baseline_result"), multimodal=False
                    )
        except requests.RequestException as exc:
            st.error(f"Could not reach the FastAPI backend: {exc}")
        except ValueError:
            st.error("The backend returned an invalid comparison response.")
