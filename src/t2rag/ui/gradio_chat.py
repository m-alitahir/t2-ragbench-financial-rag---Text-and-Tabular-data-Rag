"""Local Gradio chat page for manual RAG testing.

The UI is intentionally a thin layer over the existing retrieval and answer
classes. It does not build indexes, modify benchmark data, or expose the API
key. Each message is treated as an independent financial question so visible
chat history cannot silently change benchmark-style retrieval behavior.
"""

from __future__ import annotations

import atexit
import html
import logging
import time
from collections.abc import Callable, Sequence
from typing import Any

from t2rag.config import AppConfig
from t2rag.errors import T2RagError
from t2rag.generation.answerer import AnswerParseError, EvidenceAnswerer
from t2rag.generation.presentation import (
    ComparisonPlan,
    ComparisonPresenter,
    PresentationError,
    asks_for_average,
    compute_comparison,
    format_decimal_value,
    format_fact_value,
    is_comparison_question,
)
from t2rag.models import AnswerResult, SearchHit
from t2rag.retrieval.service import HybridSearchService
from t2rag.safety import question_block_reason


LOGGER = logging.getLogger(__name__)
RETRIEVAL_MODES = ("bm25", "dense", "hybrid", "reranked", "routed")
DATASET_FILTERS = ("FinQA", "ConvFinQA", "TAT-DQA", "All datasets")
SPLIT_FILTERS = ("dev", "train", "test", "All splits")


def _markdown_text(value: object) -> str:
    """Escape untrusted model or source text before Markdown display."""

    text = html.escape(str(value), quote=False)
    for character in (
        "\\",
        "`",
        "*",
        "_",
        "{",
        "}",
        "[",
        "]",
        "<",
        ">",
        "#",
        "|",
    ):
        text = text.replace(character, f"\\{character}")
    return text


def _compact_preview(text: str, *, limit: int = 700) -> str:
    """Return a readable evidence preview without flooding the chat window."""

    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit].rstrip()}…"


def _with_query_correction(answer: str, hits: Sequence[SearchHit]) -> str:
    """Show a spelling correction without replacing the user's question."""

    for hit in hits:
        corrected = hit.metadata.get("query_correction_corrected")
        if isinstance(corrected, str) and corrected.strip():
            return (
                f"> **Possible spelling match:** Also searched for "
                f"“{_markdown_text(corrected)}”.\n\n{answer}"
            )
    return answer


def _polished_comparison(
    plan: ComparisonPlan,
    *,
    include_average: bool = False,
) -> str:
    """Render a comparison with code-owned numbers and direction."""

    computed = compute_comparison(plan)
    baseline = computed.baseline
    compared = computed.comparison
    label = _markdown_text(plan.label)
    baseline_period = _markdown_text(baseline.period)
    compared_period = _markdown_text(compared.period)
    change_value = format_fact_value(compared, abs(computed.change))

    if computed.percentage_point_change is not None and computed.change != 0:
        points = format_decimal_value(abs(computed.percentage_point_change), places=1)
        verb = "increased" if computed.direction == "increase" else "decreased"
        direction_text = f"{verb} by {points} percentage points"
    elif computed.direction == "increase":
        direction_text = f"increased by {change_value}"
        if computed.relative_change_percent is not None:
            percent = format_decimal_value(computed.relative_change_percent, places=1)
            direction_text += f", or {percent}%"
    elif computed.direction == "decrease":
        direction_text = f"decreased by {change_value}"
        if computed.relative_change_percent is not None:
            percent = format_decimal_value(abs(computed.relative_change_percent), places=1)
            direction_text += f", or {percent}%"
    else:
        direction_text = "did not change"

    direct_parts: list[str] = []
    if include_average:
        average = format_fact_value(compared, computed.average)
        direct_parts.append(
            f"{label} averaged {average} across {baseline_period} and {compared_period}."
        )
    direct_parts.append(
        f"{label} {direction_text} from {baseline_period} to {compared_period}."
    )

    lines = [
        f"## {_markdown_text(plan.title)}",
        "",
        f"**{' '.join(direct_parts)}**",
        "",
        f"| Period or item | {label} |",
        "|---|---:|",
        f"| {baseline_period} | {format_fact_value(baseline)} |",
        f"| {compared_period} | {format_fact_value(compared)} |",
    ]
    if include_average:
        lines.append(
            f"| Two-period average | {format_fact_value(compared, computed.average)} |"
        )
    lines.append("")

    if computed.higher_fact_id is not None:
        higher = plan.fact(computed.higher_fact_id)
        lines.append(
            f"The higher value was reported for **{_markdown_text(higher.period)}**."
        )
    else:
        lines.append("The two reported values are equal.")

    if (
        computed.relative_change_percent is None
        and computed.percentage_point_change is None
        and computed.change != 0
    ):
        lines.extend(
            [
                "",
                "A relative percentage change is not shown because the baseline "
                "is zero or negative.",
            ]
        )
    return "\n".join(lines)


def format_final_response(
    result: AnswerResult | None,
    *,
    presentation: ComparisonPlan | None = None,
    include_average: bool = False,
) -> str:
    """Return only the polished answer that belongs in the chat."""

    if presentation is not None:
        return _polished_comparison(
            presentation,
            include_average=include_average,
        )
    if result is None:
        return (
            "## I could not verify an answer\n\n"
            "The requested comparison could not be supported by the retrieved documents."
        )
    if result.abstained:
        return (
            "## I could not verify an answer\n\n"
            "I could not find enough support in the selected financial documents. "
            "Try naming the company, period, and metric you want to examine."
        )
    clean_answer = " ".join(result.answer.split())
    return f"## Answer\n\n**{_markdown_text(clean_answer)}**"


def _selected_hits(
    result: AnswerResult | None,
    hits: Sequence[SearchHit],
    presentation: ComparisonPlan | None,
) -> list[SearchHit]:
    """Return only evidence used by the answer or verified comparison facts."""

    chunk_ids = {
        citation.chunk_id for citation in result.citations
    } if result is not None else set()
    if presentation is not None:
        for fact in presentation.facts:
            if fact.evidence_id.startswith("E") and fact.evidence_id[1:].isdigit():
                index = int(fact.evidence_id[1:]) - 1
                if 0 <= index < len(hits):
                    chunk_ids.add(hits[index].chunk_id)
    return [hit for hit in hits if hit.chunk_id in chunk_ids]


def format_source_details(
    result: AnswerResult | None,
    hits: Sequence[SearchHit],
    *,
    presentation: ComparisonPlan | None = None,
    include_average: bool = False,
) -> str:
    """Render code-owned calculations and sources for the closed UI panel."""

    lines = ["### Calculation details", ""]
    if presentation is not None:
        computed = compute_comparison(presentation)
        lines.extend(
            [
                f"- Baseline: **{_markdown_text(computed.baseline.period)}** — "
                f"{format_fact_value(computed.baseline)}",
                f"- Compared value: **{_markdown_text(computed.comparison.period)}** — "
                f"{format_fact_value(computed.comparison)}",
                f"- Absolute change: **{format_fact_value(computed.comparison, computed.change)}**",
            ]
        )
        if include_average:
            lines.append(
                f"- Two-period average: **"
                f"{format_fact_value(computed.comparison, computed.average)}**"
            )
        if computed.relative_change_percent is not None:
            relative = format_decimal_value(computed.relative_change_percent, places=1)
            lines.append(f"- Relative change from the baseline: **{relative}%**")
        if computed.percentage_point_change is not None:
            points = format_decimal_value(computed.percentage_point_change, places=1)
            lines.append(f"- Percentage-point change: **{points} points**")
    else:
        if result is None:
            return "No validated source details are available."
        lines.extend(
            [
                f"- Normalized answer: **{_markdown_text(result.normalized_answer or 'none')}**",
                f"- Result unit: **{_markdown_text(result.result_unit or 'not recorded')}**",
                f"- Calculation: `{_markdown_text(result.calculation or 'none')}`",
            ]
        )

    lines.extend(["", "### Sources", ""])
    selected = _selected_hits(result, hits, presentation)
    if not selected:
        lines.append("No source was used because the model abstained.")
    for index, hit in enumerate(selected, start=1):
        page = hit.page_number if hit.page_number is not None else "unknown"
        lines.extend(
            [
                f"{index}. `{_markdown_text(hit.alias_path)}` — page `{page}`",
                "",
                f"> {_markdown_text(_compact_preview(hit.source_text))}",
                "",
            ]
        )
    return "\n".join(lines).strip()


def format_chat_response(
    result: AnswerResult,
    hits: Sequence[SearchHit],
    *,
    retrieval_mode: str,
    elapsed_seconds: float,
    dataset_filter: str = "All datasets",
    split_filter: str = "All splits",
    presentation: ComparisonPlan | None = None,
    include_average: bool = False,
) -> str:
    """Compatibility wrapper that now returns only the polished main answer."""

    del hits, retrieval_mode, elapsed_seconds, dataset_filter, split_filter
    return format_final_response(
        result,
        presentation=presentation,
        include_average=include_average,
    )


class RagChatController:
    """Keep retrieval services warm and answer independent UI questions."""

    def __init__(
        self,
        config: AppConfig,
        *,
        service_factory: Callable[[str], HybridSearchService] | None = None,
        answerer: EvidenceAnswerer | None = None,
        presenter: ComparisonPresenter | None = None,
        presenter_factory: Callable[[], ComparisonPresenter] | None = None,
    ) -> None:
        self._config = config
        self._service_factory = service_factory or self._create_service
        self._answerer = answerer or EvidenceAnswerer(config)
        self._presenter = presenter
        self._presenter_factory = presenter_factory or (lambda: ComparisonPresenter(config))
        self._services: dict[str, HybridSearchService] = {}
        self._closed = False

    def _create_service(self, mode: str) -> HybridSearchService:
        return HybridSearchService.from_config(
            self._config,
            enable_dense=mode in {"dense", "hybrid", "reranked", "routed"},
            enable_reranker=mode in {"reranked", "routed"},
        )

    def _service(self, mode: str) -> HybridSearchService:
        if self._closed:
            raise RuntimeError("The chat controller is closed.")
        if mode not in RETRIEVAL_MODES:
            raise ValueError(f"Unsupported retrieval mode: {mode}")
        if mode not in self._services:
            self._services[mode] = self._service_factory(mode)
        return self._services[mode]

    @staticmethod
    def _retrieve(
        service: HybridSearchService,
        question: str,
        mode: str,
        dataset: str | None,
        split: str | None,
    ) -> list[SearchHit]:
        if mode == "bm25":
            return service.search_lexical(
                question,
                dataset=dataset,
                split=split,
                typo_tolerant=True,
            )
        if mode == "dense":
            return service.search_dense(question, dataset=dataset, split=split)
        if mode == "hybrid":
            return service.search(
                question,
                use_dense=True,
                use_reranker=False,
                dataset=dataset,
                split=split,
                typo_tolerant=True,
            )
        if mode == "reranked":
            return service.search(
                question,
                use_dense=True,
                use_reranker=True,
                dataset=dataset,
                split=split,
                typo_tolerant=True,
            )
        if mode == "routed":
            return service.search_routed(
                question,
                use_reranker=True,
                dataset=dataset,
                split=split,
                typo_tolerant=True,
            )[1]
        raise ValueError(f"Unsupported retrieval mode: {mode}")

    def _comparison_plan(
        self,
        question: str,
        result: AnswerResult | None,
        hits: Sequence[SearchHit],
    ) -> ComparisonPlan | None:
        """Build one optional plan without making it part of benchmark scoring."""

        if (result is not None and result.abstained) or not is_comparison_question(question):
            return None
        if self._presenter is None:
            self._presenter = self._presenter_factory()
        return self._presenter.present(question, result, hits)

    def answer_question_parts(
        self,
        question: str,
        retrieval_mode: str,
        dataset_filter: str = "FinQA",
        split_filter: str = "dev",
    ) -> tuple[str, str]:
        """Return a polished answer and separate source details."""

        clean_question = question.strip()
        if not clean_question:
            return "Enter a financial question.", "No document search was run."

        block_reason = question_block_reason(clean_question)
        if block_reason == "unsafe_instruction":
            return (
                "## I cannot help with that request\n\n"
                "I cannot reveal private instructions, credentials, or secrets. "
                "You can ask me a question about the indexed financial reports.",
                "No document search or paid model request was run.",
            )
        if block_reason == "unrelated_creative_request":
            return (
                "## That request is outside this financial collection\n\n"
                "I can answer questions supported by the indexed financial reports. "
                "Ask about a company, period, metric, table, or calculation.",
                "No document search or paid model request was run.",
            )

        started = time.perf_counter()
        try:
            service = self._service(retrieval_mode)
            dataset = None if dataset_filter == "All datasets" else dataset_filter
            split = None if split_filter == "All splits" else split_filter
            hits = self._retrieve(
                service,
                clean_question,
                retrieval_mode,
                dataset,
                split,
            )
        except (T2RagError, FileNotFoundError, ValueError, RuntimeError) as exc:
            LOGGER.warning("RAG chat request failed: %s", exc)
            return (
                "## The request could not be completed\n\n"
                f"{_markdown_text(exc)}",
                "No validated source details are available.",
            )
        except Exception:
            LOGGER.exception("Unexpected RAG chat failure.")
            return (
                "## The request could not be completed\n\n"
                "An unexpected local application error occurred.",
                "No validated source details are available.",
            )

        presentation: ComparisonPlan | None = None
        include_average = asks_for_average(clean_question)
        if is_comparison_question(clean_question):
            try:
                presentation = self._comparison_plan(clean_question, None, hits)
            except (PresentationError, T2RagError, ValueError, RuntimeError) as exc:
                LOGGER.warning("Comparison extraction fell back to the scalar answer: %s", exc)
            except Exception:
                LOGGER.exception("Unexpected comparison-extraction failure.")

        if presentation is not None:
            elapsed_seconds = time.perf_counter() - started
            LOGGER.info(
                "Completed comparison answer in %.2f seconds with retrieval mode %s.",
                elapsed_seconds,
                retrieval_mode,
            )
            return (
                _with_query_correction(
                    format_final_response(
                        None,
                        presentation=presentation,
                        include_average=include_average,
                    ),
                    hits,
                ),
                format_source_details(
                    None,
                    hits,
                    presentation=presentation,
                    include_average=include_average,
                ),
            )

        try:
            result = self._answerer.answer(clean_question, hits)
        except AnswerParseError as exc:
            LOGGER.warning("The answer model returned invalid output: %s", exc)
            return (
                "## I could not produce a reliable answer\n\n"
                "The generated response failed validation, so I did not show or guess it.",
                "The answer failed the structured-output validation step.",
            )
        except (T2RagError, FileNotFoundError, ValueError, RuntimeError) as exc:
            LOGGER.warning("RAG chat request failed: %s", exc)
            return (
                "## The request could not be completed\n\n"
                f"{_markdown_text(exc)}",
                "No validated source details are available.",
            )
        except Exception:
            LOGGER.exception("Unexpected RAG chat failure.")
            return (
                "## The request could not be completed\n\n"
                "An unexpected local application error occurred.",
                "No validated source details are available.",
            )

        elapsed_seconds = time.perf_counter() - started
        LOGGER.info(
            "Completed chat answer in %.2f seconds with retrieval mode %s.",
            elapsed_seconds,
            retrieval_mode,
        )
        return (
            _with_query_correction(format_final_response(result), hits),
            format_source_details(result, hits),
        )

    def answer_question(
        self,
        question: str,
        retrieval_mode: str,
        dataset_filter: str = "FinQA",
        split_filter: str = "dev",
    ) -> str:
        """Return only the polished main answer for non-Gradio callers."""

        answer, _details = self.answer_question_parts(
            question,
            retrieval_mode,
            dataset_filter,
            split_filter,
        )
        return answer

    def respond(
        self,
        message: str,
        history: list[dict[str, Any]],
        retrieval_mode: str,
        dataset_filter: str = "FinQA",
        split_filter: str = "dev",
    ) -> tuple[str, str]:
        """Gradio callback. History stays visible but does not alter retrieval."""

        del history
        return self.answer_question_parts(
            message,
            retrieval_mode,
            dataset_filter,
            split_filter,
        )

    def close(self) -> None:
        """Release Ollama and reranker HTTP clients created by the UI."""

        if self._closed:
            return
        self._closed = True
        for service in self._services.values():
            service.close()
        self._services.clear()


def create_gradio_app(
    controller: RagChatController,
    *,
    default_retrieval_mode: str = "reranked",
) -> Any:
    """Create the app without launching a server or making model requests."""

    if default_retrieval_mode not in RETRIEVAL_MODES:
        raise ValueError(
            f"Unsupported default retrieval mode: {default_retrieval_mode}"
        )
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "Gradio is not installed. Install the project dependencies again."
        ) from exc

    with gr.Blocks(title="T2-RAGBench Financial RAG") as demo:
        gr.Markdown(
            "# T2-RAGBench Financial RAG\n"
            "Ask one financial question at a time. Each question is independent. "
            "Dense modes use local Ollama embeddings. Reranked modes and every "
            "generated answer use paid DeepInfra requests. Comparison questions "
            "use a specialized verified extraction request. Clear spelling mistakes "
            "are expanded locally and shown before the answer."
        )
        retrieval_mode = gr.Dropdown(
            choices=list(RETRIEVAL_MODES),
            value=default_retrieval_mode,
            label="Retrieval mode",
            info="Use reranked for best quality. Use BM25 for cheaper retrieval testing.",
        )
        dataset_filter = gr.Dropdown(
            choices=list(DATASET_FILTERS),
            value="FinQA",
            label="Dataset filter",
            info="Use one dataset for benchmark-style testing.",
        )
        split_filter = gr.Dropdown(
            choices=list(SPLIT_FILTERS),
            value="dev",
            label="Split filter",
            info="Use dev while tuning. Keep test sealed until final evaluation.",
        )
        chatbot = gr.Chatbot(
            height=650,
            placeholder="Ask a question about the indexed financial reports.",
            sanitize_html=True,
        )
        question_box = gr.Textbox(
            label="Ask a financial question",
            placeholder="Type your question here, then select Get final answer.",
            lines=2,
            submit_btn="Get final answer",
            stop_btn="Stop",
        )
        with gr.Accordion("Sources and calculation", open=False):
            source_details = gr.Markdown(
                "Source details for the latest answer will appear here."
            )
        gr.ChatInterface(
            fn=controller.respond,
            chatbot=chatbot,
            textbox=question_box,
            additional_inputs=[retrieval_mode, dataset_filter, split_filter],
            additional_outputs=[source_details],
            concurrency_limit=1,
            save_history=True,
            title=None,
            description=None,
        )
    return demo


def launch_gradio_chat(
    config: AppConfig,
    *,
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = False,
    inbrowser: bool = True,
    retrieval_mode: str = "reranked",
) -> dict[str, Any]:
    """Launch a blocking local server and return settings after it stops."""

    if not host.strip():
        raise ValueError("The chat host cannot be empty.")
    if not 1 <= port <= 65535:
        raise ValueError("The chat port must be between 1 and 65535.")

    controller = RagChatController(config)
    atexit.register(controller.close)
    demo = create_gradio_app(
        controller,
        default_retrieval_mode=retrieval_mode,
    )
    demo.queue(default_concurrency_limit=1)
    try:
        demo.launch(
            server_name=host,
            server_port=port,
            share=share,
            inbrowser=inbrowser,
        )
    finally:
        controller.close()
    return {
        "host": host,
        "port": port,
        "share": share,
        "retrieval_mode": retrieval_mode,
    }
