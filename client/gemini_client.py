import os

from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from threading import Lock


ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

MODEL_NAME = "gemini-3.5-flash-lite"

MAX_API_CALLS_PER_SESSION = 500
MAX_OUTPUT_TOKENS = 512

# Estimated pricing per million tokens.
INPUT_TOKEN_PRICE = 0.30
OUTPUT_TOKEN_PRICE = 2.50


class GeminiClient:

    # Shared across all instances of GeminiClient.
    request_count = 0
    input_tokens = 0
    output_tokens = 0
    thinking_tokens = 0

    request_lock = Lock()


    def __init__(self):
        load_dotenv(ENV_PATH)

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is missing.")

        self.client = genai.Client(api_key=api_key)
        self.model = MODEL_NAME


    def generate(self, prompt):

        # Only allow one outstanding API request.
        if not GeminiClient.request_lock.acquire(
            blocking=False
        ):
            return None

        try:

            # Enforce the session request limit.
            if (
                GeminiClient.request_count
                >= MAX_API_CALLS_PER_SESSION
            ):
                print("Gemini API session limit reached.")
                return None

            # Count attempted requests, including failures.
            GeminiClient.request_count += 1

            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                ),
            )

            # Record token consumption.
            self.record_usage(response)

            # Display cumulative usage.
            self.print_usage()

            return response.text

        except Exception as error:
            print(
                f"Gemini API error: "
                f"{type(error).__name__}"
            )

            return None

        finally:
            GeminiClient.request_lock.release()


    def record_usage(self, response):
        usage = response.usage_metadata

        if usage is None:
            return

        GeminiClient.input_tokens += (
            usage.prompt_token_count or 0
        )

        GeminiClient.output_tokens += (
            usage.candidates_token_count or 0
        )

        GeminiClient.thinking_tokens += (
            usage.thoughts_token_count or 0
        )


    def get_estimated_cost(self):

        input_cost = (
            GeminiClient.input_tokens
            / 1_000_000
            * INPUT_TOKEN_PRICE
        )

        output_cost = (
            (
                GeminiClient.output_tokens
                + GeminiClient.thinking_tokens
            )
            / 1_000_000
            * OUTPUT_TOKEN_PRICE
        )

        return input_cost + output_cost


    def get_usage(self):
        return {
            "requests": GeminiClient.request_count,

            "remaining_requests": max(
                0,
                MAX_API_CALLS_PER_SESSION
                - GeminiClient.request_count,
            ),

            "input_tokens": GeminiClient.input_tokens,
            "output_tokens": GeminiClient.output_tokens,
            "thinking_tokens": GeminiClient.thinking_tokens,

            "total_tokens": (
                GeminiClient.input_tokens
                + GeminiClient.output_tokens
                + GeminiClient.thinking_tokens
            ),

            "estimated_cost": self.get_estimated_cost(),
        }


    def print_usage(self):
        usage = self.get_usage()

        print(
            f"[Gemini] "
            f"Requests: {usage['requests']}/"
            f"{MAX_API_CALLS_PER_SESSION} | "
            f"Input: {usage['input_tokens']} | "
            f"Output: {usage['output_tokens']} | "
            f"Thinking: {usage['thinking_tokens']} | "
            f"Estimated Cost: "
            f"${usage['estimated_cost']:.6f}"
        )