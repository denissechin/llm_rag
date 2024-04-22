import os

from huggingface_hub import snapshot_download
from llama_cpp import Llama

DEFAULT_REPO = "IlyaGusev/saiga_llama3_8b_gguf"
DEFAULT_MODEL = "model-q4_K.gguf"


class LLMForRAG:
    def __init__(
        self,
        model_path: str | None = None,
        n_gpu_layers: int = 0,
        n_ctx: int = 2048,
        n_batch: int = 256,
        temperature: float = 0.5,
        max_tokens: int = 256,
        n_threads: int | None = None,
        verbose: bool = False,
    ) -> None:
        if model_path is None:
            model_path = os.path.join("models", DEFAULT_MODEL)
            if not os.path.exists(model_path):
                self._load_model(DEFAULT_REPO, DEFAULT_MODEL)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model found on path {model_path}")
        
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            n_batch=n_batch,
            temperature=temperature,
            max_tokens=max_tokens,
            n_threads=n_threads,
            top_p=0.95,
            verbose=verbose,
        )

    def _load_model(self, repo_id: str, file_name: str) -> None:
        snapshot_download(
            repo_id=repo_id,
            local_dir="./models/",
            allow_patterns=file_name,
            local_dir_use_symlinks=False,
        )

    def invoke(self, messages: list[dict[str, str]]) -> str:
        result = self.llm.create_chat_completion(messages=messages)
        return result["choices"][0]["message"]["content"]
