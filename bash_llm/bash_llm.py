from .prompts import CODER_SYSTEM_PROMPT, CODER_USER_TEMPLATE, PLANNER_SYSTEM_PROMPT, PLANNER_USER_TEMPLATE
from .config import BashLLMConfig
from .llmodel import LLModel

import subprocess

class BashLLM:
    def __init__(self, config: BashLLMConfig) -> None:
        self.config = config

        self._small_brain   = LLModel(self.config.weak_model)
        self._big_brain     = LLModel(self.config.strong_model)

        pass

    def handle_utility_menu(self, command: str) -> None:
        command_parts = command.split(" ")

        switch = {
            "/help": lambda x: print(switch.keys()),
            "/exit": lambda x: exit(),
        }

        default = lambda x: print(switch.keys())

        switch.get(command_parts[0], default)(command)

        return

    def execute_coder_response(self, response: str) -> None:
        bash_code = response

        leading_backticks   = response.startswith("`")

        if leading_backticks:
            lines = response.replace("\r", "").split("\n")

            bash_code = "\n".join(lines[1:-1])

        with open("model_response.sh", "w+", newline="\n") as f:
            f.write(bash_code)

        if self.config.always_sudo:
            cmd = ["bash", "-c", "sudo bash model_response.sh"]
        else: 
            cmd = ["bash", "model_response.sh"]
            
        print(f"Invoking: {' '.join(cmd)}")
        subprocess.call(cmd)

    def handle_planner(self, user_input: str) -> str:
        s = self._small_brain.new_session(PLANNER_SYSTEM_PROMPT)
        
        r = s.generate(PLANNER_USER_TEMPLATE.format(user_input))
        print(f"Planner Evaluation:\n{r}")

        return r

    def handle_coder(self, user_input: str, planner_plan: str) -> str:
        s = self._big_brain.new_session(CODER_SYSTEM_PROMPT)
        r = s.generate(CODER_USER_TEMPLATE.format(user_input, planner_plan), max_new_tokens=2048)
        print(f"Coder Response:\n{r}")

        return r

    def run(self) -> None:
        while True:
            user_input = input("Chat: ")

            if user_input.strip() == "": continue 

            if user_input.startswith("/"): 
                self.handle_utility_menu(user_input)
                continue

            planner_plan = self.handle_planner(user_input)
            coder_code = self.handle_coder(user_input, planner_plan)

            if not self.config.prompt_before_execute or (self.config.prompt_before_execute and input("Execute coder output? (y/n)").lower() == "y"):
                self.execute_coder_response(coder_code)

