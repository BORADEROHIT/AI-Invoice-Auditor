from pathlib import Path

from litellm import completion

class GenerationAgent:
    """
    RAG Generation Agent
    Uses Cohere Command R+ through LiteLLM.
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

                if stripped == "rag_generation_agent:":
                    inside_agent = True
                    continue

                if inside_agent and stripped.startswith("model:"):
                    _, value = stripped.split(":", 1)
                    return value.strip().strip('"').strip("'")

                if inside_agent and not line.startswith("  "):
                    break

        return default_model

    def answer(self, query: str, documents: list) -> dict:
        context = "\n\n".join([doc.page_content for doc in documents])

        prompt = f"""
    Use only the context below to answer the user's invoice question.

    If the answer is not present in the context, say:
    "I do not know based on the available invoice records."

    Context:
    {context}

    Question:
    {query}
    """

        try:
            response = completion(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an invoice support assistant using RAG context."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            answer = response["choices"][0]["message"]["content"]

        except Exception as error:
            answer = f"RAG generation failed: {str(error)}"

        return {
            "answer": answer,
            "context": context
        }
