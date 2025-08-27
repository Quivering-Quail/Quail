import argparse
import asyncio
import logging
from datetime import datetime
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
        1  # Set this to whatever works for your search-provider/ai-model rate limits
    )
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            research = ""
            researcher = self.get_llm("researcher")

            prompt = clean_indents(
                f"""
                You are an assistant to a superforecaster.
                The superforecaster will give you a question they intend to forecast on.
                To be a great assistant, you generate a concise but detailed summary of the most relevant news from the last year, with a particular focus on more recent news.
                You do not produce forecasts yourself.

                Question:
                {question.question_text}

                This question's outcome will be determined by the specific criteria below:
                {question.resolution_criteria}

                {question.fine_print}
                """
            )

            if isinstance(researcher, GeneralLlm):
                research = await researcher.invoke(prompt)
            elif researcher == "asknews/news-summaries":
                research = await AskNewsSearcher().get_formatted_news_async(
                    question.question_text
                )
            elif researcher == "asknews/deep-research/medium-depth":
                research = await AskNewsSearcher().get_formatted_deep_research(
                    question.question_text,
                    sources=["asknews", "google"],
                    search_depth=2,
                    max_depth=4,
                )
            elif researcher == "asknews/deep-research/high-depth":
                research = await AskNewsSearcher().get_formatted_deep_research(
                    question.question_text,
                    sources=["asknews", "google"],
                    search_depth=4,
                    max_depth=6,
                )
            elif researcher.startswith("smart-searcher"):
                model_name = researcher.removeprefix("smart-searcher/")
                searcher = SmartSearcher(
                    model=model_name,
                    temperature=0,
                    num_searches_to_run=2,
                    num_sites_per_search=10,
                    use_advanced_filters=False,
                )
                research = await searcher.invoke(prompt)
            elif not researcher or researcher == "None":
                research = ""
            else:
                research = await self.get_llm("researcher", "llm").invoke(prompt)
            logger.info(f"Found Research for URL {question.page_url}:\n{research}")
            return research


    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            I am going to ask you to complete a sequence of steps in order to generate a probabilistic forecast on the following question. The question describes an [EVENT] and a threshold [DATE]:
            {question.question_text}

            Here is some background information about this question:
            {question.background_info}

            This question's outcome will be determined by the specific resolution criteria below. The probabilistic forecast should refer to these specific resolution criteria. These resolution criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}
            
            Here is a summary of recent research relating to this question:
            {research}
            
            Today is {datetime.now().strftime("%Y-%m-%d")}.


            Treat the following steps as independent. Do not let your response to one influence the other. Complete each step separately and in the sequence specified.
            
            Step 1: Based on current knowledge and trends, what is the most likely date by which [EVENT] will occur? Consider the resolution criteria, and think deeply.
            
            Step 2: Estimate the 99th percentile of when a positive outcome will occur. Consider the resolution criteria, and think deeply. To avoid bias from anchoring or order effects, please follow this specific sampling protocol:
            2.1 - You will be given a list of time intervals relative to today. 
            2.2 - First, compute the exact dates these refer to (format: DD/MM/YYYY), keeping them in the specified order. 
            2.3 - Second, you must now evaluate each date independently, in the order specified, without being influenced by previous or subsequent dates. For each date, estimate the probability (0–100%) that the resolution criteria for the [EVENT] will have occurred *by* that date.
            
            Here are the candidate dates (in DD/MM/YYYY format):
            - tomorrow
            - 100 years from now
            - three days from now
            - 90 years from now
            - one week from now
            - 70 years from now
            - two weeks from now
            - 50 years from now
            - one month from now
            - 30 years from now
            - two monts from now
            - 20 years from now
            - three months from now
            - 15 years from now
            - six months from now
            - 10 years from now
            - one year from now
            - five years from now
            - two years from now
            - three years from now
            
            Please proceed with the evaluation using this protocol. 
            
            Estimate the 99th percentile, interpolating if necessary.
            
            Step 3: Use the PERT distribution to calculate the probability of [EVENT] (as described in the resolution criteria) by [DATE]. 
            Use the mode "most likely" date from Step 1, the minimum of today's date, the maximum of the 99th centile estimated in Step 2, and a shape parameter of 4.
            Be as rigorous as possible in this estimate, using Beta CDF tables and/or numerical integration, as appropriate.


            Output: Your output should contain only the following information (no more than 350 words total):

            1 - A very brief summary of the most relevant evidence relating to the question.
            2 - State the most likely date (Step 1), and provide a sentence or two of justification.
            3 - State the 99th percentile date, and provide a sentence or two of justification.
            4 - Very briefly describe that you used a statistical method to calculate a probability density function from which you computed the probability. Don't specifically mention PERT.
            5 - The last thing you write is your final answer as: "Probability: ZZ%", 0-100
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
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            The options are: {question.options}


            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}


            Here is a summary of recent research relating to this question:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A description of an scenario that results in an unexpected outcome.

            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
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
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            
            I am going to ask you to complete a sequence of steps in order to generate a probabilistic forecast on the following question. The question describes an [EVENT] and a threshold [DATE]:
            {question.question_text}

            Here is some background information about this question:
            {question.background_info}

            This question's outcome will be determined by the specific resolution criteria below. The probabilistic forecast should refer to these specific resolution criteria. These resolution criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}
            
            Here is a summary of recent research relating to this question:
            {research}
            
            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Formatting Instructions:
            - Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there.

            
            Treat the following steps as independent. Do not let your response to one influence the other. Complete each step separately and in the sequence specified.
            
            Step 1: Based on current knowledge, trends (and, if relevant, the  event count so far within the timeframe), estimate the most likely number of events that will occur by [DATE]. Consider the resolution criteria, and think deeply.
            
            Step 2: Estimate the 99th percentile of the event counts by [DATE]. Consider the resolution criteria, and think deeply. To avoid bias from anchoring or order effects, please follow this specific sampling protocol:
            2.1 - Produce a single numeric value that represents an absolute upper bound for the [EVENT] count by [DATE] — a value you are certain will not be exceeded under any plausible scenario.
            2.2 - Generate a logarithmic scale of 20 values between the current [EVENT] count and this upper bound. Reorder this list in the following alternating pattern, starting from the extremes and moving inward: largest, smallest, second largest, second smallest...
            2.3 - You must now evaluate each value independently, in the order specified, without being influenced by previous or subsequent values. For each value, estimate the probability (0–100%) that the [EVENT] count will meet this threshold *by* [DATE].
            
            Please proceed with the evaluation using this protocol. 
            
            Estimate the 99th percentile, interpolating if necessary.
            
            Step 3: You will now use the PERT distribution to estimate the 10th, 20th, 40th, 60th, 80th and 90th centile counts for the [EVENT] (as defined in the resolution criteria) by [DATE].
            3.1 - Construct a PERT distribution CDF to describe the probability distrubution across different [EVENT] counts by [DATE]. Use the mode "most likely" value from Step 1, the minimum of today's count, the maximum of the 99th centile estimated in Step 2, and a shape parameter of 4.
            3.2 - Use the PERT distribution CDF to calculate each of the centile values stated above. Be as rigorous as possible in these estimates, using Beta CDF tables and/or numerical integration, as appropriate.


            Output: Your output should contain only the following information (no more than 350 words total):

            1 - A brief summary of the most relevant evidence relating to the question.
            2 - State the most likely value (Step 1), and provide a sentence or two of justification.
            3 - State the 99th percentile value, and provide a sentence or two of justification.
            4 - Very briefly describe that you used a statistical method to calculate a probability density function from which you computed the probability. Don't specifically mention PERT.
            5 - The last thing you write is your final answer as:
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

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion
    ) -> tuple[str, str]:
        if question.nominal_upper_bound is not None:
            upper_bound_number = question.nominal_upper_bound
        else:
            upper_bound_number = question.upper_bound
        if question.nominal_lower_bound is not None:
            lower_bound_number = question.nominal_lower_bound
        else:
            lower_bound_number = question.lower_bound

        if question.open_upper_bound:
            upper_bound_message = f"The question creator thinks the number is likely not higher than {upper_bound_number}."
        else:
            upper_bound_message = (
                f"The outcome can not be higher than {upper_bound_number}."
            )

        if question.open_lower_bound:
            lower_bound_message = f"The question creator thinks the number is likely not lower than {lower_bound_number}."
        else:
            lower_bound_message = (
                f"The outcome can not be lower than {lower_bound_number}."
            )
        return upper_bound_message, lower_bound_message


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
             "default": GeneralLlm(model="openrouter/openai/gpt-5"),
        #         temperature=0.3,
        #         timeout=40,
        #         allowed_tries=2,
             
        #     "summarizer": "openai/gpt-4o-mini",
             "researcher": "asknews/deep-research/low",
        #     "parser": "openai/gpt-4o-mini",
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
