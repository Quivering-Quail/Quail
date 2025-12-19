import argparse
import asyncio
import logging
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Literal
from asknews_sdk import AsyncAskNewsSDK

from forecasting_tools import (
    AskNewsSearcher,
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
    SmartSearcher,
    clean_indents,
    structure_output,
)

logger = logging.getLogger(__name__)


class FallTemplateBot2025(ForecastBot):

    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    # -----------------------------
    # 1. Parse & Summarize Question
    # -----------------------------
    async def parse_and_summarize_question(self, question: MetaculusQuestion) -> str:
        prompt = clean_indents(f"""
        You are an expert superforecasting assistant.

        Parse and summarize the following forecasting question.
        Do NOT make a forecast.

        Produce:
        - Event definition
        - What counts as NO / status quo
        - Time horizon sensitivity
        - Relevant base-rate analogies
        - Key ambiguities in resolution

        Question:
        {question.question_text}

        Resolution:
        {question.resolution_criteria}

        Notes:
        {question.fine_print}
        """)
        return await self.get_llm("parser", "llm").invoke(prompt)

    # -----------------------------
    # 2. Commission Multi-Source Research
    # -----------------------------
    async def run_research_bundle(self, question, summary) -> dict[str, str]:
        async def run(model_key, prompt):
            return await self.get_llm(model_key, "llm").invoke(prompt)

        research_prompt = clean_indents(f"""
        You are conducting research to support a superforecaster answer the question below.
        Do NOT forecast.

        Emphasize:
        - Explicitly search for similar or related questions on:
          • Metaculus
          • Polymarket
          • Good Judgement Open
          • Manifold Markets
          • Kalshi
          • RAND Forecasting Initiative
        - If found, report:
          • Current implied probabilities
          • Trading volume / liquidity (if available)
          • Directional trend (rising / falling / stable)
        - If not found, explicitly state "No close market analogue found"
        
        - Statistical base rates (where appropriate)
        - Diagnostic evidence, trends and drivers
        - Source currency, relevance, authority, accuracy and purpose (CRAAP test)
        - Recent developments
        - The pathway to positive resolution (where relevant)
        - Evidence for and against other alternative outcomes (where relevant)

        Question summary:
        {summary}
        """)

        tasks = {
            "perplexity": run("research_perplexity", research_prompt),
            "asknews": run("research_asknews", research_prompt),
            "gemini": run("research_gemini", research_prompt),
            "claude": run("research_claude", research_prompt),
            "deepseek": run("research_deepseek", research_prompt),
        }
        return {k: await v for k, v in tasks.items()}

    # -----------------------------
    # 3. Consolidate Research
    # -----------------------------
    async def consolidate_research(self, summary, bundle) -> str:
        prompt = clean_indents(f"""
        You are synthesizing research for forecasting.
        Do NOT forecast.

        Question summary:
        {summary}

        Research inputs:
        {bundle}

        Produce sections:
        1. Base rates & historical analogues
        2. Evidence for change vs inertia
        3. Key drivers & indicators
        4. Disconfirming evidence
        5. Unknowns & resolution risks
        """)
        return await self.get_llm("summarizer", "llm").invoke(prompt)

    # -----------------------------
    # 4. Multi-Model Forecasting
    # -----------------------------
async def _run_forecast_with_llm(
    self,
    question,
    research,
    summary,
    llm_key: str,
):
    llm = self.get_llm(llm_key, "llm")

    if isinstance(question, BinaryQuestion):
        return await self._run_forecast_on_binary_with_llm(
            question, research, summary, llm
        )
    elif isinstance(question, NumericQuestion):
        return await self._run_forecast_on_numeric_with_llm(
            question, research, summary, llm
        )
    elif isinstance(question, MultipleChoiceQuestion):
        return await self._run_forecast_on_multiple_choice_with_llm(
            question, research, summary, llm
        )
    else:
        raise ValueError("Unsupported question type")

    # -----------------------------
    # 5. Median + Challenger Synthesis
    # -----------------------------
    async def synthesize_forecast(self, forecasts, summary, research):
        probs = [f.prediction_value for f in forecasts]
        median_prob = sorted(probs)[len(probs)//2]
        forecast_summaries = "\n".join(
            f"- Model {i+1}: {f.prediction_value}"
            for i, f in enumerate(forecasts)
        )

        challenger_prompt = clean_indents(f"""
        You are an adversarial forecaster.
        
        Here are multiple independent forecasts:
        {forecast_summaries}
        
        Identify:
        - Shared assumptions
        - Neglected tail risks
        - Status quo bias
        - Overconfident base-rate shifts
        
        Do NOT give a probability.

        """)
        
        challengers = [
            "challenger_grok",
            "forecast_deepseek",
            "forecast_qwen",
        ]

        challenger_outputs = []

        for challenger in challengers:
            critique = await self.get_llm(challenger, "llm").invoke(challenger_prompt)
            challenger_outputs.append(f"[{challenger}]\n{critique}")
        
        challenge = "\n\n".join(challenger_outputs)
        
        synthesis_prompt = clean_indents(f"""
        You are the lead superforecaster.

        Forecasts:
        {probs}

        Median anchor:
        {median_prob:.2%}

        Challenger critique:
        {challenge}

        You MUST anchor on the median.
        Adjust only if justified.

        Produce final probability and reasoning.
        """)
        return await self.get_llm("synthesizer", "llm").invoke(synthesis_prompt)
   
    async def _run_forecast_on_binary_with_llm(
        self,
        question: BinaryQuestion,
        research: str,
        summary: str,
        llm,
    ) -> ReasonedPrediction[float]:

        candidates = await self.generate_candidate_dates(datetime.now())
        candidate_table = "\n".join(
        f"- {label}: {date.strftime('%Y-%m-%d')}" for label, date in candidates
        )
        
        prompt = clean_indents(
            f"""
            You are asked to generate a probabilistic forecast for the following question:
            {question.question_text}

            Further information on the question:
            {summary}
            
            Relevant research:
            {research}
            
            Today's date:
            {datetime.now().strftime("%Y-%m-%d")}.

            While reasoning, the following questions may be helpful:
            1. Is there a relevant base rate I should anchor on?
            2. How long is there until the question timeframe expires?
            3. What trends and drivers are informative?
            4. What other outcomes are possible?
            5. How does the evidence weigh against the possible outcomes?
            6. How much should I deviate from the baseline, considering the diagnosticity of the evidence?
            7. What key uncertainties and assumptions impact my forecast?

            **Output (max 500 words):**
            1. Summarise the scope-sensitive extrapolation of the base rate (where relevant).
            2. Summarise the argument and evidence supporting each question outcome.
            3. The last thing you write is your final answer in this exact format:
            "
            Probability: ZZ%", [This must be a value between 0.1% and 99.9% - adjust 0% or 100% to 0.1% or 99.9%]
            "
            """
        )
        reasoning = await llm.invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        binary_prediction: BinaryPrediction = await structure_output(
            reasoning, BinaryPrediction, model=self.get_llm("parser", "llm")
        )
        decimal_pred = max(0.001, min(0.999, binary_prediction.prediction_in_decimal))

        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {decimal_pred}"
        )
        return ReasonedPrediction(prediction_value=decimal_pred, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice_with_llm(
        self,
        question: MultipleChoiceQuestion,
        research: str,
        summary: str,
        llm,
    ) -> ReasonedPrediction[PredictedOptionList]:
        
        prompt = clean_indents(
            f"""
            You are asked to produce a categoric probabilistic forecast for the following question:
            {question.question_text}

            Further information on the question:
            {summary}

            Relevant research:
            {research}

            Today's date:
            {datetime.now().strftime("%Y-%m-%d")}.

            While reasoning, the following questions may be helpful:
            1. Is there a relevant base rate I should anchor on?
            2. How long is there until the question timeframe expires?
            3. What trends and drivers are informative?
            4. What other outcomes are possible?
            5. How does the evidence weigh against the possible outcomes?
            6. How much should I deviate from the baseline, considering the diagnosticity of the evidence?
            7. What key uncertainties and assumptions impact my forecast?

            **Output (max 500 words):**
            1. Summarise the scope-sensitive extrapolation of the base rate (where relevant).
            2. Summarise the argument and evidence supporting each question outcome.
            3. The last thing you write is your final forecast for the N options {question.options} in this exact format and order:
            "
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            "
            """
        )
        parsing_instructions = clean_indents(
            f"""
            Make sure that all option names are one of the following:
            {question.options}
            The text you are parsing may prepend these options with some variation of "Option" which you should remove if not part of the option names I just gave you.
            """
        )
        reasoning = await llm.invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        predicted_option_list: PredictedOptionList = await structure_output(
            text_to_structure=reasoning,
            output_type=PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
        )
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {predicted_option_list}"
        )
        return ReasonedPrediction(
            prediction_value=predicted_option_list, reasoning=reasoning
        )

    async def _run_forecast_on_numeric_with_llm(
        self,
        question: NumericQuestion,
        research: str,
        summary: str,
        llm,
    ) -> ReasonedPrediction[NumericDistribution]:

        prompt = clean_indents(
            f"""
            You are asked to produce a probabilistic forecast for the following question:
            {question.question_text}
            
            Further information on the question:
            {summary}
            
            Units: 
            {question.unit_of_measure if question.unit_of_measure else "Not stated — please infer from context"}
            
            Relevant recent research:
            {research}
            
            Today’s date: 
            {datetime.now().strftime("%Y-%m-%d")}
            
            While reasoning, the following questions may be helpful:
            1. Is there a relevant base rate I should anchor on?
            2. How long is there until the question timeframe expires?
            3. What trends and drivers are informative?
            4. What other outcomes are possible?
            5. How does the evidence weigh against the possible outcomes?
            6. How much should I deviate from the baseline, considering the diagnosticity of the evidence?
            7. What key uncertainties and assumptions impact my forecast?

            **Output (max 500 words):**
            1. Summarise the scope-sensitive extrapolation of the base rate (where relevant).
            2. Summarise the argument and evidence supporting each question outcome.
            3. The last thing you write is your final answer in this exact format:
                        "
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            "
            """
        )
        reasoning = await llm.invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        percentile_list: list[Percentile] = await structure_output(
            reasoning, list[Percentile], model=self.get_llm("parser", "llm")
        )
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}"
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def generate_forecasts(self, question, summary, research):
        model_keys = [
            "forecast_gpt52",
            "forecast_claude",
            "forecast_gemini",
            "forecast_deepseek",
            "forecast_qwen",
            "forecast_mistral",
            "forecast_grok",
        ]
    
        async def run(model_key):
            return await self._run_forecast_with_llm(
                question=question,
                research=research,
                summary=summary,
                llm_key=model_key,
            )
    
        return await asyncio.gather(*[run(k) for k in model_keys])

    async def run_research(self, question: MetaculusQuestion, summary: str) -> str:
        research_bundle = await self.run_research_bundle(question, summary)
        return await self.consolidate_research(summary, research_bundle)

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        summary = await self.parse_and_summarize_question(question)
        return await self._run_forecast_on_binary_with_llm(
            question=question,
            research=research,
            summary=summary,
            llm=self.get_llm("forecast_gpt52", "llm"),
        )

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        summary = await self.parse_and_summarize_question(question)
        return await self._run_forecast_on_multiple_choice_with_llm(
            question=question,
            research=research,
            summary=summary,
            llm=self.get_llm("forecast_gpt52", "llm"),
        )

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        summary = await self.parse_and_summarize_question(question)
        return await self._run_forecast_on_numeric_with_llm(
            question=question,
            research=research,
            summary=summary,
            llm=self.get_llm("forecast_gpt52", "llm"),
        )
    
    async def _forecast_single_question(
        self, question: MetaculusQuestion
    ) -> ReasonedPrediction:
        async with self._concurrency_limiter:
            summary = await self.parse_and_summarize_question(question)
    
            research_bundle = await self.run_research_bundle(question, summary)
    
            consolidated_research = await self.consolidate_research(
                summary, research_bundle
            )
    
            forecasts = await self.generate_forecasts(
                question, summary, consolidated_research
            )
    
            final_prediction = await self.synthesize_forecast(
                forecasts, summary, consolidated_research
            )
    
            return final_prediction

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Suppress LiteLLM logging
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    parser = argparse.ArgumentParser(
        description="Run the Q1TemplateBot forecasting system"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "metaculus_cup", "test_questions"],
        default="tournament",
        help="Specify the run mode (default: tournament)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode
    assert run_mode in [
        "tournament",
        "metaculus_cup",
        "test_questions",
    ], "Invalid run mode"

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
        seasonal_tournament_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_AI_COMPETITION_ID, return_exceptions=True
            )
        )
        minibench_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_MINIBENCH_ID, return_exceptions=True
            )
        )
        forecast_reports = seasonal_tournament_reports + minibench_reports
    elif run_mode == "metaculus_cup":
        # The Metaculus cup is a good way to test the bot's performance on regularly open questions. You can also use AXC_2025_TOURNAMENT_ID = 32564 or AI_2027_TOURNAMENT_ID = "ai-2027"
        # The Metaculus cup may not be initialized near the beginning of a season (i.e. January, May, September)
        template_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_METACULUS_CUP_ID, return_exceptions=True
            )
        )
    elif run_mode == "test_questions":
        # Example questions are a good way to test the bot's performance on a single question
        EXAMPLE_QUESTIONS = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",  # Human Extinction - Binary
            "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",  # Age of Oldest Human - Numeric
            "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",  # Number of New Leading AI Labs - Multiple Choice
            "https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/",  # Number of US Labor Strikes Due to AI in 2029 - Discrete
        ]
        template_bot.skip_previously_forecasted_questions = False
        questions = [
            MetaculusApi.get_question_by_url(question_url)
            for question_url in EXAMPLE_QUESTIONS
        ]
        forecast_reports = asyncio.run(
            template_bot.forecast_questions(questions, return_exceptions=True)
        )
    template_bot.log_report_summary(forecast_reports)
