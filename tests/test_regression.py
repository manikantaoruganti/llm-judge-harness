import pytest
import csv
import os
from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage
from dotenv import load_dotenv

# Load environment variables from .env.example for local testing context
load_dotenv()

# Set LangSmith project for regression tests
os.environ["LANGCHAIN_PROJECT"] = "pytest-regression-suite"

# Import the agent function
from src.agent import generate_output

@pytest.fixture(scope="module")
def golden_dataset():
    """Loads the golden dataset from CSV."""
    dataset = []
    file_path = os.path.join(os.path.dirname(__file__), "fixtures", "golden_dataset.csv")
    with open(file_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset.append(row)
    return dataset

@pytest.fixture(scope="module")
def bad_prompt_content():
    """Loads the content of bad_prompt.txt."""
    file_path = os.path.join(os.path.dirname(__file__), "..", "bad_prompt.txt")
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

@patch('src.agent.ChatOpenAI')
def test_llm_reformatting_regression(mock_chat_openai, golden_dataset, bad_prompt_content):
    """
    Tests the LLM reformatting agent against the golden dataset.
    Mocks ChatOpenAI to allow offline testing and simulate bad prompt behavior.
    """
    mock_instance = MagicMock()
    mock_chat_openai.return_value = mock_instance

    is_bad_prompt_test = os.environ.get("TEST_BAD_PROMPT") == "1"
    prompt_path = os.path.join(os.path.dirname(__file__), "..", "bad_prompt.txt") if is_bad_prompt_test else None

    for i, row in enumerate(golden_dataset):
        input_text = row["input_text"]
        expected_output_contains = row["expected_output_contains"]

        if is_bad_prompt_test:
            # Simulate the LLM following the bad prompt by injecting promotional content
            mock_instance.invoke.return_value = AIMessage(
                content=f"Update: {input_text[:150]} Click the link in bio to buy now!"
            )
        else:
            # Simulate the LLM following the good prompt
            mock_instance.invoke.return_value = AIMessage(
                content=f"A concise social media update: {input_text}"
            )

        # Call the agent function, which will use the mocked ChatOpenAI
        output = generate_output(input_text, prompt_template_path=prompt_path)

        # Assertions for regression (active for all runs)
        assert len(output) <= 280, f"Output too long for source_id {row['source_id']}: {len(output)} chars"

        assert any(keyword.strip().lower() in output.lower() for keyword in expected_output_contains.split(',')), \
            f"Output for source_id {row['source_id']} does not contain expected content: '{expected_output_contains}'"

        assert "click the link in bio to buy now!" not in output.lower(), \
            f"Output for source_id {row['source_id']} contains unsupported promotional content injected by prompt."

@patch('src.agent.ChatOpenAI')
def test_golden_dataset_structure(mock_chat_openai, golden_dataset):
    """
    Tests the structure and minimum size of the golden dataset.
    """
    assert len(golden_dataset) >= 15, "Golden dataset must contain at least 15 rows."
    required_columns = ["source_id", "input_text", "expected_output_contains"]
    for i, row in enumerate(golden_dataset):
        assert all(col in row for col in required_columns), \
            f"Row {i+1} is missing required columns. Found: {list(row.keys())}, Expected: {required_columns}"
        assert row["source_id"], f"Row {i+1} has empty source_id."
        assert row["input_text"], f"Row {i+1} has empty input_text."
        assert row["expected_output_contains"], f"Row {i+1} has empty expected_output_contains."
