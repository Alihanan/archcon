"""Local web application for displaying uploaded text files."""

from __future__ import annotations

import gradio as gr

from .io import MAX_FILE_SIZE_BYTES, TEXT_FILE_TYPES, read_text_file


def read_uploaded_text(file_path: str | None) -> tuple[str, str]:
    """Read a Gradio-uploaded file.

    This compatibility wrapper keeps the web callback name explicit while the
    actual file-reading logic remains reusable from :mod:`archcon.io`.
    """
    return read_text_file(file_path)


def _clear_output() -> tuple[str, str]:
    return "No file selected.", ""


def build_app() -> gr.Blocks:
    """Construct and return the Gradio application without starting it."""
    with gr.Blocks(
        title="ArchCon",
        analytics_enabled=False,
        delete_cache=(3600, 3600),
    ) as app:
        gr.Markdown(
            """
# ArchCon

Local web-interface backbone. Choose or drag a text file below; its contents
will be displayed locally in your browser.
"""
        )

        file_input = gr.File(
            label="Choose a text file",
            file_count="single",
            file_types=TEXT_FILE_TYPES,
            type="filepath",
        )
        status = gr.Markdown("No file selected.")
        content = gr.Textbox(
            label="File contents",
            lines=28,
            max_lines=50,
            interactive=False,
        )

        file_input.upload(
            fn=read_uploaded_text,
            inputs=file_input,
            outputs=[status, content],
            api_visibility="private",
        )
        file_input.clear(
            fn=_clear_output,
            inputs=None,
            outputs=[status, content],
            api_visibility="private",
        )

    return app


def start(
    host: str = "127.0.0.1",
    port: int = 7860,
    *,
    open_browser: bool = True,
) -> None:
    """Start the local web UI and block until it is stopped."""
    app = build_app()
    app.launch(
        server_name=host,
        server_port=port,
        inbrowser=open_browser,
        share=False,
        show_error=True,
        max_file_size=MAX_FILE_SIZE_BYTES,
        enable_monitoring=False,
        footer_links=[],
    )
