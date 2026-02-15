#!/usr/bin/python3

import re
import json
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Dict, Any

from .answer_model import AAgent


class AnsweringAgent(object):
    """
    Production-safe Answer Agent.
    Strict JSON output.
    Deterministic.
    No self-reflection loops.
    """

    def __init__(self, **kwargs):
        self.agent = AAgent(**kwargs)


    def build_prompt(self, question_data: Dict[str, Any]) -> Tuple[str, str]:

        system_prompt = (
            "You are a competitive reasoning expert. "
            "Return ONLY a valid JSON object with keys: answer and reasoning. "
            "Answer must be exactly one letter from A, B, C, or D. "
            "Reasoning must be clear and under 100 words. "
            "Do not include any extra text outside JSON."
        )

        user_prompt = (
            f"Question: {question_data['question']}\n"
            f"Choices: {self._format_choices(question_data['choices'])}\n\n"
            "Return strictly in this format:\n"
            "{\n"
            '  "answer": "A/B/C/D",\n'
            '  "reasoning": "Short explanation"\n'
            "}"
        )

        return user_prompt, system_prompt

    def answer_question(
        self, question_data: Dict | List[Dict], **kwargs
    ) -> Tuple[List[str], int | None, float | None]:

        if isinstance(question_data, list):
            prompts = []
            system_prompt = None
            for q in question_data:
                p, sp = self.build_prompt(q)
                prompts.append(p)
                system_prompt = sp
        else:
            prompts, system_prompt = self.build_prompt(question_data)

        # Force deterministic decoding
        kwargs.update({
            "temperature": 0.0,
            "do_sample": False,
            "max_new_tokens": 200,
        })

        resp, tl, gt = self.agent.generate_response(prompts, system_prompt, **kwargs)

        return resp, tl, gt

    def answer_batches(
        self, questions: List[Dict], batch_size: int = 5, **kwargs
    ) -> Tuple[List[str], List[int | None], List[float | None]]:

        answers = []
        tls, gts = [], []

        total_batches = (len(questions) + batch_size - 1) // batch_size
        pbar = tqdm(total=total_batches, desc="Answering", unit="batch")

        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i : i + batch_size]
            batch_answers, tl, gt = self.answer_question(batch_questions, **kwargs)

            answers.extend(batch_answers)
            tls.append(tl)
            gts.append(gt)

            pbar.update(1)

        pbar.close()
        return answers, tls, gts


    def filter_answers(self, ans: List[str]) -> List[Dict[str, str]]:

        filtered_answers = []

        for i, a in enumerate(ans):
            try:
                # Extract first JSON block only
                match = re.search(r"\{.*?\}", a, re.DOTALL)
                if not match:
                    filtered_answers.append(None)
                    continue

                parsed = json.loads(match.group(0))

                if (
                    isinstance(parsed.get("answer"), str)
                    and len(parsed["answer"]) == 1
                    and parsed["answer"].upper() in "ABCD"
                    and isinstance(parsed.get("reasoning"), str)
                ):
                    filtered_answers.append(parsed)
                else:
                    filtered_answers.append(None)

            except Exception:
                filtered_answers.append(None)

        return filtered_answers

    def save_answers(self, answers: List[str], file_path: str | Path) -> None:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(answers, f, indent=4)


    def _format_choices(self, choices: List[str]) -> str:
        formatted = []
        for idx, choice in enumerate(choices):
            letter = chr(65 + idx)  # A, B, C, D
            cleaned = re.sub(r"^[A-D][\.\)]\s*", "", choice.strip())
            formatted.append(f"{letter}) {cleaned}")
        return " ".join(formatted)

if __name__ == "__main__":

    import argparse

    argparser = argparse.ArgumentParser(description="Run the Answering Agent")
    argparser.add_argument("--input_file", type=str, required=True)
    argparser.add_argument("--output_file", type=str, required=True)
    argparser.add_argument("--batch_size", type=int, default=5)
    args = argparser.parse_args()

    with open(args.input_file, "r") as f:
        questions = json.load(f)

    agent = AnsweringAgent()

    answers, tls, gts = agent.answer_batches(
        questions=questions,
        batch_size=args.batch_size
    )

    filtered = agent.filter_answers(answers)

    agent.save_answers(filtered, args.output_file)

    print("Answering completed.")
