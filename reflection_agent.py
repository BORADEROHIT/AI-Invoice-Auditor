import json
from pathlib import Path

from litellm import completion

class ReflectionAgent:
    """
    RAG Reflection Agent
    """

    def __init__(
        self,
        persona_path: str = "configs/persona_invoice_agent.yaml"
    ):
        self.persona_path = Path(persona_path)
        self.model = self._load_model_from_persona()

    def _load_model_from_persona(self) -> str:
        default_model = "bedrock/cohere.command-r-plus-v1:0"

        if not self.persona_path.exists():
            return default_model

        inside_agent = False

        with open(self.persona_path, "r", encoding="utf-8") as file:
            for line in file:
                stripped = line.strip()

                if stripped == "rag_reflection_agent:":
                    inside_agent = True
                    continue

                if inside_agent and stripped.startswith("model:"):
                    _, value = stripped.split(":", 1)
                    return value.strip().strip('"').strip("'")

                if inside_agent and not line.startswith("  "):
                    break

        return default_model

    def evaluate(self, query: str, answer: str, context: str) -> dict:
        prompt = f"""
    Evaluate the RAG answer using RAG Triad.

    Return JSON only:
    {{
    "answer_relevance": 0.0,
    "groundedness": 0.0,
    "context_relevance": 0.0,
    "comments": "short explanation"
    }}

    Question:
    {query}

    Answer:
    {answer}

    Context:
    {context}
    """

        try:
            response = completion(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a RAG quality evaluator."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            raw_output = response["choices"][0]["message"]["content"]
            cleaned = raw_output.strip().replace("```json", "").replace("```", "").strip()

            return json.loads(cleaned)

        except Exception as error:
            return {
                "answer_relevance": 0.0,
                "groundedness": 0.0,
                "context_relevance": 0.0,
                "comments": f"Reflection failed: {str(error)}"
            }
