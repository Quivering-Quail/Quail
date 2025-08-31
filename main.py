import argparse
import asyncio
import logging
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Literal

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

    _max_concurrent_questions = (
        1
    )
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def generate_candidate_dates(today: datetime) -> list[tuple[str, datetime]]:
        offsets = [
            ("tomorrow", timedelta(days=1)),
            ("three days from now", timedelta(days=3)),
            ("one week from now", timedelta(weeks=1)),
            ("two weeks from now", timedelta(weeks=2)),
            ("one month from now", relativedelta(months=1)),
            ("two months from now", relativedelta(months=2)),
            ("three months from now", relativedelta(months=3)),
            ("six months from now", relativedelta(months=6)),
            ("one year from now", relativedelta(years=1)),
            ("two years from now", relativedelta(years=2)),
            ("three years from now", relativedelta(years=3)),
            ("five years from now", relativedelta(years=5)),
            ("10 years from now", relativedelta(years=10)),
            ("15 years from now", relativedelta(years=15)),
            ("20 years from now", relativedelta(years=20)),
            ("30 years from now", relativedelta(years=30)),
            ("50 years from now", relativedelta(years=50)),
            ("70 years from now", relativedelta(years=70)),
            ("90 years from now", relativedelta(years=90)),
            ("100 years from now", relativedelta(years=100)),
        ]
        return [(label, today + offset) for label, offset in offsets]

    
    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            asknews_research = ""
            default_research = ""
            research = ""
            default_researcher = self.get_llm("default")
            summarizer = self.get_llm("summarizer")

            research_prompt = clean_indents(
                f"""
                You are a research assistant supporting a superforecaster.  
                Your task is to gather relevant information to inform a forecast on the question below.  
                Do **not** make predictions or express opinions.            
                
                ---
                
                **Question:**  
                {question.question_text}
                
                **Resolution Criteria:**  
                {question.resolution_criteria}
                
                **Additional Notes:**  
                {question.fine_print}
                
                ---
                
                Approach this research question from **five distinct angles**, with minimal overlap:
                
                1. **Focused:** Specific events, entities or people directly related to the question  
                2. **Broad:** General trends, patterns and context in the relevant domain  
                3. **Forward-looking:** Projections, anticipated developments, or upcoming events  
                4. **Drivers & Indicators:** Policies, funding shifts, macro factors, early signals or underlying dynamics  
                5. **Methodology:** How resolution-related data is defined, tracked, or reported for resolution
                
                Prioritize coverage from the **past 18 months**, especially the **most recent** developments.         
                """
            )
            
            summarizer_prompt = clean_indents(
                f"""
                You are an expert research assistant supporting a superforecaster. 
                Your role is to process a large body of research and produce a summary that best informs forecasting on the following question. 
                You must never make forecasts yourself — only extract and organize relevant information.
                
                **Forecasting Question:**  
                {question.question_text}
                
                **Resolution Criteria:**  
                {question.resolution_criteria}
                
                **Additional Notes:**  
                {question.fine_print}
                
                ### Instructions:
                - Extract only the findings, arguments, and evidence that directly inform the forecasting question.  
                - Omit extraneous detail, tangential discussion, and repetition.  
                - Identify areas of consensus, points of disagreement, and key uncertainties.  
                - Flag notable gaps, limitations, or potential biases in the research base.  
                - Present the output in structured sections (e.g., *Key Findings*, *Supporting Evidence*, *Counterarguments*, *Research Gaps*).  
                - Your synthesis should be approximately 300-500 words and distil the most decision-relevant insights for a forecaster.  

                The News research is as follows:
                {asknews_research}

                The general research is as follows:
                {default_research}
            """
            )
                                  
            asknews_research = await AskNewsSearcher().get_formatted_deep_research(
                research_prompt,
                sources=["asknews", "google"],
                search_depth=2,
                max_depth=4,
            )
            logger.info(f"Found AskNews research for {question.page_url}:\n{asknews_research}")            
                                 
            default_research = await default_researcher.invoke(research_prompt)
            logger.info(f"Found default research for {question.page_url}:\n{default_research}")         
            
            research = await summarizer.invoke(summarizer_prompt)
            
            return research


    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:

        candidates = self.generate_candidate_dates(datetime.now())
        candidate_table = "\n".join(
        f"- {label}: {date.strftime('%Y-%m-%d')}" for label, date in candidates
        )
        
        prompt = clean_indents(
            f"""
            You are asked to generate a probabilistic forecast for the following question involving an [EVENT] and a threshold date of [DATE]:
            {question.question_text}

            Background:
            {question.background_info}

            Resolution criteria (not yet satisfied):
            {question.resolution_criteria}

            Additional notes:
            {question.fine_print}
            
            Relevant research:
            {research}
            
            Today's date:
            {datetime.now().strftime("%Y-%m-%d")}.

            **Instructions (follow steps in order)**
            Base your reasoning on current data, trends, and the resolution criteria. Be precise and thoughtful.
            
            **Step 1: Most Likely Date**
            Based on current knowledge and trends, estimate the **most likely date** the [EVENT] will occur. Refer to the resolution criteria.
            
            **Step 2: 99th Percentile Estimation Protocol**
            Estimate the **99th percentile** date by evaluating a set of candidate time points.
            For each date, independently estimate the probability (0–100%) that the [EVENT] will have occurred by that date, based on the resolution criteria.

            Candidate dates:
            {candidate_table}
            
            After assigning probabilities, interpolate as needed to estimate the 99th percentile date.
            
            **Step 3: Probability by [DATE]**
            Construct a PERT/Beta probability distribution using the following parameters:  
            - Minimum: today  
            - Mode: Step 1 date  
            - Maximum: Step 2 (99th percentile) date  
            - Shape: 4  
            Use this to estimate the probability that the event occurs by [DATE]. Apply numerical integration or CDF methods as needed to maximise accuracy. Do not mention PERT in your output.

            **Output (max 350 words):**
            1. Brief summary of key evidence  
            2. Most likely date (from Step 1) with justification  
            3. 99th percentile date (from Step 2) with justification  
            4. Brief mention that a statistical method was used to produce a probability density function  
            5. The last thing you write is your final answer (from Step 3) in this exact format:
            "
            Probability: ZZ%", 0-100
            "
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        binary_prediction: BinaryPrediction = await structure_output(
            reasoning, BinaryPrediction, model=self.get_llm("parser", "llm")
        )
        decimal_pred = max(0.00, min(1.00, binary_prediction.prediction_in_decimal))

        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {decimal_pred}"
        )
        return ReasonedPrediction(prediction_value=decimal_pred, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are asked to produce a categoric probabilistic forecast for the following question:
            {question.question_text}

            Forecast categories: 
            {question.options}

            Background:
            {question.background_info}

            Resolution criteria (not yet satisfied):
            {question.resolution_criteria}

            Additional notes:
            {question.fine_print}

            Relevant research:
            {research}

            Today's date:
            {datetime.now().strftime("%Y-%m-%d")}.

            **Instructions (follow steps in order)**
            Base your reasoning on current data, trends, and the resolution criteria. Be precise and thoughtful.
            
            **Step 1: Decompose categories (if needed)**
            Review the forecast categories. If any are compound or ambiguous, break them into simpler, mutually exclusive, and collectively exhaustive analytical categories. Reasons to decompose may include:
            - A category groups multiple outcomes (e.g. “X or Y”, “1–10”)  
            - A category involves joint or conditional structure (e.g. “X and Y”, “not X”) — in this case, list relevant conditionals (e.g. P(X|Y), P(X’|Y), etc.)

            **Step 2: Estimate probabilities**
            Independently assign a probability to each analytical category from Step 1, based on the resolution criteria and evidence.

            **Step 3: Aggregate**
            Recombine analytical probabilities using standard probability rules to compute the probability of each original category.

            **Step 4: Normalize (if needed)**
            If necessary, scale the probabilities so the total sums to 1.

            **Output (max 350 words):**
            1. Summary of key evidence  
            2. Brief explanation of your methodology  
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
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
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

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:

        prompt = clean_indents(
            f"""
            You are asked to produce a probabilistic forecast for the following question, which concerns an [EVENT] and a threshold date of [DATE]:
            {question.question_text}
            
            Background:
            {question.background_info}
            
            Resolution criteria (not yet satisfied):
            {question.resolution_criteria}
            
            Additional notes:
            {question.fine_print}
            
            Units: 
            {question.unit_of_measure if question.unit_of_measure else "Not stated — please infer from context"}
            
            Relevant recent research:
            {research}
            
            Today’s date: 
            {datetime.now().strftime("%Y-%m-%d")}
            
            **Formatting instructions:**
            - Use the correct units (e.g. 1,000,000 vs 1 million) based on context.
            - Do not use scientific notation.
            - Always present numeric ranges in ascending order (e.g. from smaller to larger).
            
            **Instructions (follow steps in order)**
            Base your reasoning on current data, trends, and the resolution criteria. Be precise and thoughtful.
            
            **Step 1: Estimate the Most Likely Event Count**
            Estimate the most likely number of [EVENT]s to occur by [DATE]. Consider current trends, historical patterns, and (if relevant) the number of events already observed. This is your mode estimate.
            
            **Step 2: Estimate the 99th Percentile**
            Estimate the 99th percentile of [EVENT] counts by [DATE], using the following structured protocol to reduce bias:
            - 2.1: Identify a conservative upper bound — a number you are certain the count will not exceed in any realistic scenario.
            - 2.2: Generate a logarithmic scale of 20 values between the current count and the upper bound. Reorder these values in this sequence: largest, smallest, second largest, second smallest, etc.
            - 2.3: For each value, in this exact order, independently estimate the probability (0–100%) that the [EVENT] count will reach or exceed that number by [DATE].
            After completing these evaluations, interpolate to estimate the 99th percentile value.
            
            **Step 3: Estimate Key Percentiles Using a Statistical Distribution**
            Construct a PERT/Beta probability distribution using the following parameters:
            - Minimum: current event count
            - Mode: Step 1 estimate
            - Maximum: Step 2 (99th percentile) estimate
            - Shape: 4
            Use this to compute the 10th, 20th, 40th, 60th, 80th, and 90th percentiles. Use numerical integration or CDF lookup tables as needed to maximise accuracy.

            **Output (max 350 words):**
            1. Summary of key evidence
            2. Most likely value (from Step 1) with brief justification
            3. 99th percentile value (from Step 2) with brief justification
            4. Brief mention that a statistical method was used to produce a probability density function
            5. Final answer in this exact format:
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
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        percentile_list: list[Percentile] = await structure_output(
            reasoning, list[Percentile], model=self.get_llm("parser", "llm")
        )
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}"
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

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
            # Default LLM for all roles
             "default": GeneralLlm(model="openrouter/openai/gpt-5", 
                timeout=40,
                allowed_tries=2),
        #         temperature=0.3,
                
             "summarizer": "openrouter/openai/gpt-4o-mini",
             "researcher": "asknews/meta-llama/Llama-4-Maverick-17B-128E-Instruct",
             "parser": "openrouter/openai/gpt-4o-mini",
        },
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
