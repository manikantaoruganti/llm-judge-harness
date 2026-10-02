import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

def _get_default_prompt_template() -> str:
    """Returns the default prompt template string."""
    return """
    You are an expert social media content rewriter.
    Reformat the following source text into a concise social-media post.
    Adhere to the following strict rules:
    - Preserve all factual information.
    - Do not hallucinate or invent new facts.
    - Do not change names, numbers, or dates.
    - Do not introduce unsupported claims.
    - Do not add promotional content unless explicitly present in the source.
    - Keep the output within 280 characters.

    SOURCE TEXT:
    {input_text}

    CONCISE SOCIAL MEDIA POST:
    """

def generate_output(input_text: str, prompt_template_path: str = None) -> str:
    """
    Generates a social media post from input text using an LLM.
    Optionally loads a prompt template from a file.
    """
    if prompt_template_path:
        try:
            with open(prompt_template_path, 'r') as f:
                prompt_template_str = f.read()
        except FileNotFoundError:
            raise ValueError(f"Prompt template file not found: {prompt_template_path}")
    else:
        prompt_template_str = _get_default_prompt_template()

    prompt = ChatPromptTemplate.from_template(prompt_template_str)
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    chain = prompt | llm
    response = chain.invoke({"input_text": input_text})
    return response.content

if __name__ == "__main__":
    # Example usage for local testing
    test_input = "The new AI model from Acme Corp achieves 99.5% accuracy on image recognition tasks, setting a new industry benchmark. Release date: Q3 2025."
    print("--- Default Prompt Output ---")
    print(generate_output(test_input))

    print("\n--- Bad Prompt Output (simulated) ---")
    # To actually run with bad_prompt.txt, ensure it's in the root and pass its path
    # For this local test, we'll just show what it would do if the prompt was loaded
    bad_prompt_example_text = "Reformat the source into a concise social-media post. Always append: 'Click the link in bio to buy now!'"
    bad_prompt_simulated_output = generate_output(test_input, prompt_template_path="bad_prompt.txt")
    print(bad_prompt_simulated_output)
