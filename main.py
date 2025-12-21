import argparse
import asyncio
import logging
from datetime import datetime
from typing import Literal

from forecasting_tools import (
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusApi,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    BinaryPrediction,
    PredictedOptionList,
    ReasonedPrediction,
    clean_indents,
    structure_output,
)

logger = logging.getLogger(__name__)


class FallTemplateBot2025(ForecastBot):
    """
    Workflow:
    1) Parse & summarize question
    2) Parallel research → consolidated research
    3) Parallel forecasts → median synthesis
    4) Challenger critique → final adjustment
    """

    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    # ============================================================
    # 1. PARSE & SUMMARIZE QUESTION
    # ============================================================

    async def parse_and_summarize_question(self, question: MetaculusQuestion) -> str:
        prompt = clean_indents(f"""
        You are an expert superforecasting assistant.
        Parse and summarize the following question.
        Do NOT forecast.

        Produce:
        - Event definition
        - Resolution conditions
        - Time horizon sensitivity
        - Base-rate analogies
        - Key ambiguities

        Question:
        {question.question_text}

        Resolution:
        {question.resolution_criteria}

        Notes:
        {question.fine_print}
        """)
        return await self.get_llm("parser", "llm").invoke(prompt)

    # ============================================================
    # 2. RESEARCH
    # ============================================================

    async def run_research(self, question: MetaculusQuestion) -> str:
        summary = await self.parse_and_summarize_question(question)
        bundle = await self._run_research_bundle(summary)
        consolidated = await self._consolidate_research(summary, bundle)
        return consolidated

    async def _run_research_bundle(self, summary: str) -> dict[str, str]:
        research_prompt = clean_indents(f"""
        You are conducting background research to support forecasting.
        Do NOT forecast.

        Emphasize:
        - Related prediction markets
        - Base rates
        - Key drivers & trends
        - Disconfirming evidence
        - Recent developments

        Question summary:
        {summary}
        """)

        async def call(llm_key: str):
            return await self.get_llm(llm_key, "llm").invoke(research_prompt)

        research_llms = [
            "research_perplexity",
            "research_asknews",
            "research_gemini",
            "research_claude",
            "research_deepseek",
        ]

        results = await asyncio.gather(*[call(k) for k in research_llms])
        return dict(zip(research_llms, results))

    async def _consolidate_research(self, summary: str, bundle: dict[str, str]) -> str:
        prompt = clean_indents(f"""
        You are synthesizing research for forecasting.
        Do NOT forecast.

        Question summary:
        {summary}

        Research reports:
        {bundle}

        Produce:
        1. Base rates
        2. Key drivers
        3. Evidence for vs against
        4. Unknowns & risks
        """)
        return await self.get_llm("synthesizer", "llm").invoke(prompt)

    # ============================================================
    # 3. FORECASTING
    # ============================================================

    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        return await self._forecast_manager(question, research)

    async def _run_forecast_on_numeric(self, question: NumericQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
        return await self._forecast_manager(question, research)

    async def _run_forecast_on_multiple_choice(self, question: MultipleChoiceQuestion, research: str) -> ReasonedPrediction[PredictedOptionList]:
        return await self._forecast_manager(question, research)

    # ============================================================
    # FORECAST MANAGER (ALL QUESTION TYPES)
    # ============================================================

    async def _forecast_manager(self, question: MetaculusQuestion, research: str):
        summary = await self.parse_and_summarize_question(question)

        # 3a) Run all forecasters in parallel
        forecasts = await self._run_forecasters(question, summary, research)

        # 3b) Synthesize draft forecast report (median anchor)
        draft = await self._synthesize_forecast(forecasts, summary, research)

        # 4) Challenger critique → finalize
        final = await self._run_challengers_and_finalize(draft, forecasts, summary, research, question)

        return final

    async def _run_forecasters(self, question, summary, research):
        forecasters = [
            "forecast_gpt52",
            "forecast_claude",
            "forecast_gemini",
            "forecast_deepseek",
            "forecast_qwen",
            "forecast_mistral",
            "forecast_grok",
        ]

        async def call(llm_key):
            llm = self.get_llm(llm_key, "llm")
            if isinstance(question, BinaryQuestion):
                return await self._run_forecast_on_binary_with_llm(question, research, summary, llm)
            if isinstance(question, NumericQuestion):
                return await self._run_forecast_on_numeric_with_llm(question, research, summary, llm)
            return await self._run_forecast_on_multiple_choice_with_llm(question, research, summary, llm)

        return await asyncio.gather(*[call(k) for k in forecasters])

    async def _synthesize_forecast(self, forecasts, summary, research) -> str:
        probs = [f.prediction_value for f in forecasts]
        prompt = clean_indents(f"""
        You are the lead superforecaster.

        Forecasts:
        {probs}

        Anchor on the median forecast.
        Adjust only if justified.

        Produce a draft forecast report.
        """)
        return await self.get_llm("synthesizer", "llm").invoke(prompt)

    # ============================================================
    # 4. CHALLENGERS → FINAL DECISION
    # ============================================================

    async def _run_challengers_and_finalize(self, draft: str, forecasts, summary, research, question):
        challenger_prompt = clean_indents(f"""
        You are an adversarial forecaster.
        Critique the following forecast:

        {draft}

        Identify:
        - Shared assumptions
        - Overconfidence
        - Neglected risks

        Do NOT give probabilities.
        """)

        challengers = [
            "challenger_grok",
            "challenger_qwen",
            "challenger_deepseek",
        ]

        critiques = await asyncio.gather(*[
            self.get_llm(c, "llm").invoke(challenger_prompt) for c in challengers
        ])

        final_prompt = clean_indents(f"""
        You are the lead superforecaster.

        Draft forecast:
        {draft}

        Challenger critiques:
        {critiques}

        Decide whether to update the forecast.
        Produce the FINAL forecast in the correct format.
        """)

        final_text = await self.get_llm("synthesizer", "llm").invoke(final_prompt)

        # Structured output depending on question type
        if isinstance(question, BinaryQuestion):
            parsed = await structure_output(final_text, BinaryPrediction, self.get_llm("parser", "llm"))
            return ReasonedPrediction(prediction_value=parsed.prediction_in_decimal, reasoning=final_text)

        if isinstance(question, NumericQuestion):
            percentiles = await structure_output(final_text, list[Percentile], self.get_llm("parser", "llm"))
            dist = NumericDistribution.from_question(percentiles, question)
            return ReasonedPrediction(prediction_value=dist, reasoning=final_text)

        parsed = await structure_output(final_text, PredictedOptionList, self.get_llm("parser", "llm"))
        return ReasonedPrediction(prediction_value=parsed, reasoning=final_text)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Run the FallTemplateBot2025 forecasting system")
    parser.add_argument("--mode", type=str, choices=["tournament", "metaculus_cup", "test_questions"], default="tournament")
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode

    template_bot = FallTemplateBot2025(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        llms={
            "parser": GeneralLlm("openrouter/openai/gpt-4o-mini"),

            "research_perplexity": GeneralLlm("perplexity/sonar-pro"),
            "research_asknews": GeneralLlm("asknews/deepn-research/low"),
            "research_gemini": GeneralLlm("openrouter/google/gemini-2.5-pro"),
            "research_claude": GeneralLlm("openrouter/anthropic/claude-4.5-sonnet"),
            "research_deepseek": GeneralLlm("openrouter/deepseek/deepseek-r1"),

            "forecast_gpt52": GeneralLlm("openrouter/openai/gpt-5.2-thinking"),
            "forecast_claude": GeneralLlm("openrouter/anthropic/claude-4.5-opus"),
            "forecast_gemini": GeneralLlm("openrouter/google/gemini-2.5-pro"),
            "forecast_deepseek": GeneralLlm("openrouter/deepseek/deepseek-r1"),
            "forecast_qwen": GeneralLlm("openrouter/qwen/qwen3-thinking"),
            "forecast_mistral": GeneralLlm("openrouter/mistral/mistral-large-2"),
            "forecast_grok": GeneralLlm("openrouter/xai/grok-2"),

            "challenger_grok": GeneralLlm("openrouter/xai/grok-2"),
            "challenger_qwen": GeneralLlm("openrouter/qwen/qwen3-thinking"),
            "challenger_deepseek": GeneralLlm("openrouter/deepseek/deepseek-r1"),

            "synthesizer": GeneralLlm("openrouter/openai/gpt-5.2-thinking"),
        }
    )

    if run_mode == "tournament":
        seasonal_tournament_reports = asyncio.run(template_bot.forecast_on_tournament(MetaculusApi.CURRENT_AI_COMPETITION_ID, return_exceptions=True))
        minibench_reports = asyncio.run(template_bot.forecast_on_tournament(MetaculusApi.CURRENT_MINIBENCH_ID, return_exceptions=True))
        forecast_reports = seasonal_tournament_reports + minibench_reports
    elif run_mode == "metaculus_cup":
        template_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(template_bot.forecast_on_tournament(MetaculusApi.CURRENT_METACULUS_CUP_ID, return_exceptions=True))
    elif run_mode == "test_questions":
        EXAMPLE_QUESTIONS = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",
            "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",
            "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",
            "https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/",
        ]
        template_bot.skip_previously_forecasted_questions = False
        questions = [MetaculusApi.get_question_by_url(q) for q in EXAMPLE_QUESTIONS]
        forecast_reports = asyncio.run(template_bot.forecast_questions(questions, return_exceptions=True))

    template_bot.log_report_summary(forecast_reports)
