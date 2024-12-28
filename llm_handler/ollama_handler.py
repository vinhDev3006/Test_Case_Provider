import httpx
from langchain_ollama.llms import OllamaLLM


class OllamaModelHandler:
    """
    Manages interactions with the Ollama server and local model resources.

    Responsibilities:
    - Initialize and maintain an Ollama model instance (default: llama3.2:3b).
    - Verify and ensure a stable server connection.
    - Confirm model availability, pulling models if needed.
    - List and manage locally stored models.
    """

    def __init__(self, model_name="llama3.2:3b"):
        self.model_name = model_name
        self.ollama_model = OllamaLLM(model=self.model_name)
        self.local_models = [
            model["model"] for model in self.ollama_model._client.list().get("models")
        ]
        self._verify_server_connection()

    def _verify_server_connection(self):
        while True:
            try:
                self.ollama_model._client.ps()
                print("Connected to Ollama server")
                break
            except httpx.HTTPError as e:
                print(f"HTTP error for {e.request.url} - {e}")
                print("Retrying connection to Ollama server...")
                continue

    def _ensure_model_available(self):
        # Normalize the requested model name if it ends with ":latest"
        requested_model_name = (
            self.model_name[:-7]
            if self.model_name.endswith(":latest")
            else self.model_name
        )

        # Check if any local model matches the requested one when ignoring ":latest"
        found = any(
            (m[:-7] if m.endswith(":latest") else m) == requested_model_name
            for m in self.local_models
        )

        if not found:
            print(f"Model '{self.model_name}' not found locally, downloading...")
            self.ollama_model._client.pull(model=self.model_name)
            self.local_models.append(self.model_name)
        else:
            print(f"Model '{self.model_name}' is being activated.")

    def list_available_models(self):
        print("Available models:", self.local_models)

    def remove_model(self, model_name):
        if model_name not in self.local_models:
            print(f"Model '{model_name}' does not exist locally.")
            return

        try:
            self.ollama_model._client.delete(model=model_name)
            self.local_models.remove(model_name)
            print(f"Model '{model_name}' has been removed.")
        except Exception as e:
            print(f"Error removing model '{model_name}': {e}")

    def get_model_instance(self):
        self._ensure_model_available()
        return self.ollama_model
