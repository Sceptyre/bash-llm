if __name__ == "__main__": 
    from argparse import ArgumentParser
    parser = ArgumentParser(
        "bash_llm",
        description="What if you could interact with bash but in natural language? A very basic integration of a text generation model to operate a linux environment"
    )
    parser.add_argument("--config-file", "-f", default="config.json")

    args = parser.parse_args()

    from .bash_llm import BashLLM, BashLLMConfig
    cfg = BashLLMConfig.from_json_file(args.config_file)
    BashLLM(cfg).run()