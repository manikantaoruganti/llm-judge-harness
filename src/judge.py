import os
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

class JudgeResult(BaseModel):
    reason: str = Field(description="Detailed explanation for the faithfulness verdict.")
    faithful: bool = Field(description="True if the generated output preserves important facts and does not introduce unsupported information, False otherwise.")

def evaluate_faithfulness(
    source_text: str,
    generated_output: str
) -> JudgeResult:
    """
    Evaluates the faithfulness of a generated output against a source text using an LLM judge.
    """
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)

    structured_llm = llm.with_structured_output(JudgeResult)

    prompt_template = """
    You are an expert LLM judge tasked with evaluating the faithfulness of a GENERATED OUTPUT
    compared to a SOURCE TEXT. Your goal is to determine if the GENERATED OUTPUT
    preserves all important facts from the SOURCE TEXT and does not introduce any
    unsupported or hallucinated information.

    Strict Rubric:
    - Mark faithful=True ONLY if the GENERATED OUTPUT accurately reflects the SOURCE TEXT
      without any factual errors, omissions of important facts, or additions of
      unsupported claims.
    - Mark faithful=False if the GENERATED OUTPUT:
        - Invents facts not present in the SOURCE.
        - Changes names, numbers, or dates from the SOURCE.
        - Alters the sentiment or core meaning of the SOURCE.
        - Adds unsupported claims or promotional language not in the SOURCE.
        - Omits any important fact from the SOURCE.
    - Do not mark an output faithful merely because it sounds plausible.
    - Provide a concise but comprehensive reason for your verdict.

    SOURCE TEXT:
    {source_text}

    GENERATED OUTPUT:
    {generated_output}

    Your evaluation:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | structured_llm
    result = chain.invoke({"source_text": source_text, "generated_output": generated_output})
    return result

if __name__ == "__main__":
    # Example usage for local testing
    source = "Acme Corp announced a new product, the 'Quantum Widget', on May 15, 2024. It promises 20% efficiency gains."
    generated_faithful = "Acme Corp launched 'Quantum Widget' on May 15, 2024, boosting efficiency by 20%."
    generated_unfaithful_date = "Acme Corp launched 'Quantum Widget' on June 15, 2024, boosting efficiency by 20%."
    generated_unfaithful_promo = "Acme Corp launched 'Quantum Widget' on May 15, 2024, boosting efficiency by 20%! Buy now!"
    generated_unfaithful_fact = "Acme Corp launched 'Quantum Widget' on May 15, 2024, boosting efficiency by 50%."

    print("--- Faithful Example ---")
    # This will fail if OPENAI_API_KEY is not set
    # print(evaluate_faithfulness(source, generated_faithful))

    print("\n--- Unfaithful (Date Change) Example ---")
    # print(evaluate_faithfulness(source, generated_unfaithful_date))

    print("\n--- Unfaithful (Promo Added) Example ---")
    # print(evaluate_faithfulness(source, generated_unfaithful_promo))

    print("\n--- Unfaithful (Fact Changed) Example ---")
    # print(evaluate_faithfulness(source, generated_unfaithful_fact))

    print("\nNote: Live LLM calls are commented out for offline execution. Uncomment and set OPENAI_API_KEY to run.")
