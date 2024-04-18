from openai import OpenAI

__version__ = '0.1'

PROMPT_SYS = "You're a math education expert."
PROMPT_INST_HINT = "Given a math question, the correct answer, and the incorrect answer provided by the student, generate a hint meeting the requirements in the rubric. The hint should be up to two sentences long."
PROMPT_HINT_RUBRIC = "1. The hint should not make any incorrect statements and should be relevant to the current question and student answer.\n2. The hint should not directly reveal the correct answer to the student.\n3. The hint provides suggestions to the student that, when followed, will guide them towards the correct answer.\n4. The hint correctly points out the error the student made or the misconception underlying their answer.\n5. The hint is positive and has an encouraging tone."
PROMPT_CHECK_ANSWER = "Given a math question, the correct answer, and a student's answer, determine if the student's answer is correct. Answer only with a 'correct' or 'incorrect'."

default_hyperparameters = {
    "temperature": 1,
    "max_tokens": 400,
    "top_p": 1,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "request_timeout": 45
}

question_bank = {
    "1": {
        "question": "What is 2 + 2?",
        "answer": "4",
    },
}


class OpenAIInterface:
    def __init__(self, api_key):
        self.client = OpenAI(api_key)

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

    def get_question(self, question_id):
        return question_bank[question_id]["question"]

    def get_answer(self, question_id):
        return question_bank[question_id]["answer"]

    def get_student_answer(self):
        return input(f"Enter your answer.")

    def create_hint_prompt(self, question, correct_answer, student_answer):
        return f"{PROMPT_INST_HINT}\n\n{PROMPT_HINT_RUBRIC}\n\nMath question: {question}\n\nCorrect answer: {correct_answer}\n\nStudent's answer {student_answer}\n\nHint: "

    def get_hint(self, question_id, student_answer, model="gpt-3.5-turbo", **kwargs):
        prompt = self.create_hint_prompt(question_bank[question_id]["question"],
                                         question_bank[question_id]["answer"],
                                         student_answer)
        return self.get_responses(prompt, model=model, **kwargs)[0]

    def check_answer(self, question_id, student_answer):
        return student_answer == question_bank[question_id]["answer"]

    def create_check_answer_prompt(self, question, correct_answer, student_answer):
        return f"{PROMPT_CHECK_ANSWER}\n\nMath question: {question}\n\nCorrect answer: {correct_answer}\n\nStudent's answer: {student_answer}\n\nIs the student's answer correct or incorrect? "

    def check_answer_gpt(self, question_id, student_answer, model="gpt-3.5-turbo", **kwargs):
        prompt = self.create_check_answer_prompt(question_bank[question_id]["question"],
                                                 question_bank[question_id]["answer"],
                                                 student_answer)
        res = self.get_responses(prompt, model=model, **kwargs)[0]
        assert res in ["correct", "incorrect"]
        return self.get_responses(prompt, model=model, **kwargs)[0] == "correct"
