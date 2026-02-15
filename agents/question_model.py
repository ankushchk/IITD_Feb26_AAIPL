import torch
from unsloth import FastLanguageModel


class QAgent:

    def __init__(self, model_path="hf_models/llama-3.1-8b-final"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            load_in_4bit=False,
            dtype=torch.bfloat16,
        )

        self.model.eval()

    def generate_response(self, prompt, sys_prompt=None, **kwargs):

        if sys_prompt:
            prompt = sys_prompt + "\n\n" + prompt

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=170,
                temperature=0.7,
                top_p=0.9,
            )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Trim after first Answer
        if "Answer:" in text:
            text = text.split("Answer:")[0] + "Answer:" + text.split("Answer:")[1].split("\n")[0]

        return text, None, None
