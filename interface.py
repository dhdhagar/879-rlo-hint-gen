from openai import OpenAI
import os
import json
import random, string


__version__ = '0.1'


PROMPT_SYS = "You're a math education expert."
PROMPT_INST_HINT = "Given a math question, the correct answer, the reasoning steps provided by the student, and the incorrect answer provided by the student, generate a hint meeting the requirements in the rubric. The hint should be up to two sentences long."
PROMPT_HINT_RUBRIC = "1. The hint should not make any incorrect statements and should be relevant to the current question and student answer.\n2. The hint should not directly reveal the correct answer to the student.\n3. The hint provides suggestions to the student that, when followed, will guide them towards the correct answer.\n4. The hint correctly points out the error the student made or the misconception underlying their answer.\n5. The hint is positive and has an encouraging tone."
PROMPT_CHECK_ANSWER = "Given a math question, the correct answer, and a student's answer, determine if the student's answer is correct. Answer only with a 'correct' or 'incorrect'."

default_hyperparameters = {
    "temperature": 1,
    "max_tokens": 400,
    "top_p": 0.95,
}

QUESTION_BANK = {
    # "1": {
    #     "question": "We deal from a well-shuffled 52-card deck. Calculate the probability that the 13th card is the first king to be dealt.",
    #     "answer": "TODO",
    # },
    # "2": {
    #     "question": "The king has only one sibling. What is the probability that the sibling is male? Assume that every birth results in a boy with probability 1/2, independent of other births.",
    #     "answer": "0.5",
    # },
    "1": {
        "question": "You meet a father and son, whose family was selected at random from among all two-child families with at least one boy. Let p be the probability that the man's other child is also a boy. What is 6p? Assume that it is equally probable for a child to be boy or a girl.",
        "answer": "2"
    },
    "2": {
        "question": "Eight rooks are placed in distinct squares of an 8 × 8 chessboard, with all possible placements being equally likely. In how many ways can you place them such that all the rooks are safe from one another, i.e., that there is no row or column with more than one rook.",
        "answer": "40320"
    },
    "3": {
        "question": "You toss independently a fair coin and you count the number of tosses until the first tail appears. If this number is n, you receive n dollars. What is the expected amount that you will receive?",
        "answer": "2"
    }
}

class OpenAIInterface:
    def __init__(self, question_bank=QUESTION_BANK):
        self.client = OpenAI()
        self.question_bank = question_bank
        self.id = None
        self.student_reasoning_1_pre_hint = None
        self.student_answer_1_pre_hint = None
        self.student_reasoning_1_post_hint = None
        self.student_answer_1_post_hint = None
        self.student_reasoning_2_pre_hint = None
        self.student_answer_2_pre_hint = None
        self.student_reasoning_2_post_hint = None
        self.student_answer_2_post_hint = None


    def get_responses(self, prompts, model="gpt-3.5-turbo", **kwargs):
        if not isinstance(prompts, list):
            prompts = [prompts]
        hyperparameters = {**default_hyperparameters, **kwargs}
        results = []
        for prompt in prompts:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": PROMPT_SYS
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                **hyperparameters
            )
            output = response.choices[0].message.content
            results.append(output)
        return results


    def get_data(self):
        data = {"student_id": self.get_student_id(),
                "reasoning_1_pre_hint": self.student_reasoning_1_pre_hint,
                "answer_1_pre_hint": self.student_answer_1_pre_hint,
                "reasoning_1_post_hint": self.student_reasoning_1_post_hint,
                "answer_1_post_hint": self.student_answer_1_post_hint,
                "reasoning_2_pre_hint": self.student_reasoning_2_pre_hint,
                "answer_2_pre_hint": self.student_answer_2_pre_hint,
                "reasoning_2_post_hint": self.student_reasoning_2_post_hint,
                "answer_2_post_hint": self.student_answer_2_post_hint,
                }
        
        return data
        

    def save_data(self):
        data = self.get_data()
        with open(f"data_{self.get_student_id()}.json", "w") as f:
            json.dump(data, f)


    def get_student_id(self):
        if( self.id == None):
            self.id = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
    
        return self.id
    

    def get_question(self, question_id):
        return self.question_bank[question_id]["question"]


    def get_answer(self, question_id):
        return self.question_bank[question_id]["answer"]


    def get_student_reasoning_steps(self):
        return input(f"Enter your reasoning steps (press enter when done):")
    

    def get_student_answer(self):
        return input(f"Enter your answer (press enter when done):")


    def create_hint_prompt(self, question, correct_answer, student_reasoning_steps, student_answer):
        return f"{PROMPT_INST_HINT}\n\n{PROMPT_HINT_RUBRIC}\n\nMath question: {question}\n\nCorrect answer: {correct_answer}\n\nStudent's reasoning steps: {student_reasoning_steps}\n\nStudent's answer: {student_answer}\n\nHint: "


    def get_hint(self, question_id, student_reasoning_steps, student_answer, model="gpt-4", **kwargs):
        prompt = self.create_hint_prompt(self.question_bank[question_id]["question"],
                                        self.question_bank[question_id]["answer"],
                                        student_reasoning_steps,
                                        student_answer)

        return self.get_responses(prompt, model=model, **kwargs)[0]


    def check_answer(self, question_id, student_answer):
        return self._string_clean(student_answer) == self._string_clean(self.question_bank[question_id]["answer"])


    def create_check_answer_prompt(self, question, correct_answer, student_answer):
        return f"{PROMPT_CHECK_ANSWER}\n\nMath question: {question}\n\nCorrect answer: {correct_answer}\n\nStudent's answer: {student_answer}\n\nIs the student's answer correct or incorrect? "


    def _string_clean(self, string):
        return string.replace("\n", "").replace("\t", "")



"""
    def check_answer_gpt(self, question_id, student_answer, model="gpt-3.5-turbo", **kwargs):
        prompt = self.create_check_answer_prompt(self.question_bank[question_id]["question"],
                                                 self.question_bank[question_id]["answer"],
                                                 student_answer)
        res = self.get_responses(prompt, model=model, **kwargs)[0]
        assert res in ["correct", "incorrect"]
        return self.get_responses(prompt, model=model, **kwargs)[0] == "correct"
"""