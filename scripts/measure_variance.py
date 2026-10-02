import os
import json
from dotenv import load_dotenv
from src.agent import generate_output

load_dotenv()

def run_variance_test(input_text: str, num_runs: int = 20) -> dict:
    """
    Runs the LLM agent multiple times with the same input to measure output variance.
    All LLM calls are traced via LangSmith under the 'variance-measurement' project.
    """
    os.environ["LANGCHAIN_PROJECT"] = "variance-measurement"

    outputs = []
    for _ in range(num_runs):
        # The generate_output function internally uses ChatOpenAI with temperature=0
        output = generate_output(input_text)
        outputs.append(output)

    unique_outputs = len(set(outputs))
    variance_percentage = ((unique_outputs - 1) / num_runs) * 100 if num_runs > 0 else 0.0

    report = {
        "total_runs": num_runs,
        "unique_outputs": unique_outputs,
        "variance_percentage": round(variance_percentage, 2)
    }
    return report

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is missing or empty. Cannot run variance measurement without valid API credentials.")

    print("Running variance test with live LLM. This may take a moment...")
    sample_input = "The new AI model from Acme Corp achieves 99.5% accuracy on image recognition tasks, setting a new industry benchmark. Release date: Q3 2025."
    report_data = run_variance_test(sample_input)
    print("Variance test completed.")

    output_path = "results/variance_report.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2)

    print(f"Variance report saved to {output_path}")
    print(json.dumps(report_data, indent=2))
