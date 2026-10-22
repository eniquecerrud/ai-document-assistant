"""Web interface: upload documents, ask questions, read the citations.

The page calls :class:`folio.service.FolioService` in-process, so the UI and
the API share an index only when they point at the same ``FOLIO_VECTORSTORE_PATH``.
Each question is answered on its own; the transcript is display history.
"""

from __future__ import annotations

import hashlib
from typing import Any

import streamlit as st

from folio.config import load_settings
from folio.errors import FolioError
from folio.service import build_service

_SUGGESTIONS = (
    "¿Cuántos días de vacaciones tiene una persona empleada y con qué antelación se piden?",
    "How many days a week can someone work remotely, and what are the core hours?",
    "¿En cuánto tiempo hay que informar un incidente de seguridad?",
)


def main() -> None:
    """Render the Folio page. Streamlit reruns this function after each interaction."""
    st.set_page_config(page_title="Folio", page_icon="▤", layout="wide")
    _install_styles()
    service = _get_service()
    _sidebar(service)
    st.title("Folio")
    st.caption(
        "Pregunta sobre los documentos que has cargado. "
        "Cada respuesta incluye el archivo y la página de los pasajes usados."
    )

    messages: list[dict[str, Any]] = st.session_state.setdefault("messages", [])
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("citations"):
                _render_citations(message["citations"])

    pending = st.session_state.pop("pending_question", None)
    if not messages:
        columns = st.columns(len(_SUGGESTIONS))
        for column, suggestion in zip(columns, _SUGGESTIONS, strict=True):
            with column:
                if st.button(suggestion, use_container_width=True):
                    st.session_state.pending_question = suggestion
                    st.rerun()

    typed = st.chat_input("Escribe una pregunta sobre tus documentos")
    question = pending or typed
    if not question:
        return

    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Buscando en los documentos…"):
                answer = service.query(question)
        except FolioError as exc:
            st.error(str(exc))
            messages.append({"role": "assistant", "content": str(exc), "citations": []})
            return
        st.markdown(answer.text)
        citations = [
            {
                "source": item.source,
                "page": item.page,
                "score": item.score,
                "excerpt": item.excerpt,
            }
            for item in answer.citations
        ]
        _render_citations(citations)
    messages.append({"role": "assistant", "content": answer.text, "citations": citations})


def _sidebar(service: Any) -> None:
    settings = service.settings
    with st.sidebar:
        st.header("Documentos")
        uploads = st.file_uploader(
            "PDF, TXT o Markdown",
            type=["pdf", "txt", "md", "markdown"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        seen: set[str] = st.session_state.setdefault("ingested", set())
        if uploads:
            for upload in uploads:
                data = upload.getvalue()
                token = f"{upload.name}:{hashlib.sha256(data).hexdigest()}"
                if token in seen:
                    continue
                try:
                    result = service.ingest_bytes(upload.name, data)
                except FolioError as exc:
                    st.error(f"{upload.name}: {exc}")
                    continue
                seen.add(token)
                st.success(f"{result.source}: {result.chunks} fragmentos")

        documents = service.list_documents()
        if not documents:
            st.info("Todavía no hay documentos. Sube un archivo para empezar.")
        for document in documents:
            st.markdown(f"**{document.source}**")
            st.caption(f"{document.chunk_count} fragmentos · `{document.id}`")
            if st.button("Quitar", key=f"delete-{document.id}"):
                service.delete_document(document.id)
                st.session_state.ingested = set()
                st.rerun()

        st.divider()
        st.caption(f"Embeddings: `{settings.embedding.provider}` · `{settings.embedding.model}`")
        st.caption(f"Modelo: `{settings.llm.provider}` · `{settings.llm.model}`")
        st.caption(f"Índice: `{settings.vectorstore.provider}`")


@st.cache_resource
def _get_service() -> Any:
    return build_service(load_settings())


def _render_citations(citations: list[dict[str, Any]]) -> None:
    if not citations:
        return
    st.markdown("**Fuentes**")
    for number, item in enumerate(citations, start=1):
        st.markdown(
            f"`[{number}]` **{item['source']}** · página {item['page']} · "
            f"similitud {item['score']:.2f}"
        )
        st.caption(item["excerpt"])


def _install_styles() -> None:
    st.markdown(
        """
        <style>
          .stApp { background: #f6f3ee; }
          [data-testid="stSidebar"] { background: #e7f0f0; }
          h1 { letter-spacing: -0.03em; font-weight: 600; }
          [data-testid="stCaptionContainer"] { color: #44403c; }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
