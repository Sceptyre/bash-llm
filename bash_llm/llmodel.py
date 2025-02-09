from transformers import AutoModelForCausalLM, AutoTokenizer

def LLModelSessionMessage(role: str, content: str) -> dict:
    return {
        "role": role,
        "content": content
    }

class LLModelSession:
    _system_prompt: str
    _history: list[dict]

    def __init__(self, model: 'LLModel', system_prompt: str):
        self.m = model
        self._history = [
            LLModelSessionMessage("system", system_prompt)
        ]

    def generate(self, message: str, **kvargs) -> str:
        kvargs = {
            "max_new_tokens": 512,
            **kvargs
        }

        self._history.append(LLModelSessionMessage("user", message))

        prompt = self.m._tokenizer.apply_chat_template(
            self._history,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.m._tokenizer([prompt], return_tensors="pt").to(self.m._model.device)

        generated_ids = self.m._model.generate(
            **inputs,
            **kvargs
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]

        resp = self.m._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        self._history.append(LLModelSessionMessage("assistant", resp))

        return resp
    
    def close(self):
        self = None


class LLModel:
    def __init__(self, path: str, **kvargs) -> None:
        self._model = AutoModelForCausalLM.from_pretrained(
            path,
            **kvargs
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            path,
            **kvargs
        )

    def new_session(self, system_prompt: str):
        return LLModelSession(self, system_prompt)