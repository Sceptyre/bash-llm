from dataclasses import dataclass

@dataclass
class BashLLMConfig:
    weak_model:     str = "Qwen/Qwen2.5-Coder-3B-Instruct"
    strong_model:   str = "Qwen/Qwen2.5-Coder-14B-Instruct"

    always_sudo:            bool = False
    prompt_before_execute:  bool = True

    @staticmethod
    def from_json_file(file_path: str) -> 'BashLLMConfig':
        from os import path
        if not path.exists(file_path): 
            raise FileNotFoundError(f"Config file `{file_path}` not found")

        import json
        with open(file_path, "r") as f:
            conf = json.load(f)

        return BashLLMConfig(**conf)