"""
DSPy benchmark runner for SGLang's OpenAI-compatible API.

This keeps the original `bench_dspy_intro.py` intact. The legacy benchmark was
written against older DSPy client wrappers such as `HFClientSGLang`, which are
not available in current DSPy releases. This runner uses the current generic
`dspy.LM(...)` interface instead and takes the serving endpoint and model from
CLI arguments so notebook and Colab setups do not need to patch repository
files in place.
"""

import argparse

import dspy
from dspy.datasets import HotPotQA


class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""

    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()

        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)


def build_sglang_lm(args):
    model = args.model
    if not model.startswith("openai/"):
        model = f"openai/{model}"

    return dspy.LM(
        model,
        api_base=f"{args.url.rstrip('/')}:{args.port}/v1",
        api_key=args.api_key,
        model_type="chat",
    )


def main(args):
    lm = build_sglang_lm(args)

    colbertv2_wiki17_abstracts = dspy.ColBERTv2(
        url="http://20.102.90.50:2017/wiki17_abstracts"
    )
    dspy.settings.configure(lm=lm, rm=colbertv2_wiki17_abstracts)

    dataset = HotPotQA(
        train_seed=1, train_size=20, eval_seed=2023, dev_size=args.dev_size, test_size=0
    )

    trainset = [x.with_inputs("question") for x in dataset.train]
    devset = [x.with_inputs("question") for x in dataset.dev]

    print(len(trainset), len(devset))

    train_example = trainset[0]
    print(f"Question: {train_example.question}")
    print(f"Answer: {train_example.answer}")

    dev_example = devset[18]
    print(f"Question: {dev_example.question}")
    print(f"Answer: {dev_example.answer}")
    print(f"Relevant Wikipedia Titles: {dev_example.gold_titles}")

    print(
        f"For this dataset, training examples have input keys {train_example.inputs().keys()} and label keys {train_example.labels().keys()}"
    )
    print(
        f"For this dataset, dev examples have input keys {dev_example.inputs().keys()} and label keys {dev_example.labels().keys()}"
    )

    generate_answer = dspy.Predict(BasicQA)
    pred = generate_answer(question=dev_example.question)
    print(f"Question: {dev_example.question}")
    print(f"Predicted Answer: {pred.answer}")

    lm.inspect_history(n=1)

    generate_answer_with_chain_of_thought = dspy.ChainOfThought(BasicQA)
    pred = generate_answer_with_chain_of_thought(question=dev_example.question)
    print(f"Question: {dev_example.question}")
    rationale = getattr(pred, "rationale", None)
    if rationale is not None:
        thought = rationale.split(".", 1)[1].strip() if "." in rationale else rationale
        print(f"Thought: {thought}")
    else:
        print("Thought: <not exposed by this DSPy version>")
    print(f"Predicted Answer: {pred.answer}")

    retrieve = dspy.Retrieve(k=3)
    topK_passages = retrieve(dev_example.question).passages
    print(
        f"Top {retrieve.k} passages for question: {dev_example.question} \n",
        "-" * 30,
        "\n",
    )

    for idx, passage in enumerate(topK_passages):
        print(f"{idx+1}]", passage, "\n")

    retrieve("When was the first FIFA World Cup held?").passages[0]

    from dspy.teleprompt import BootstrapFewShot

    def validate_context_and_answer(example, pred, trace=None):
        answer_EM = dspy.evaluate.answer_exact_match(example, pred)
        answer_PM = dspy.evaluate.answer_passage_match(example, pred)
        return answer_EM and answer_PM

    teleprompter = BootstrapFewShot(metric=validate_context_and_answer)
    compiled_rag = teleprompter.compile(RAG(), trainset=trainset)

    my_question = "What castle did David Gregory inherit?"
    pred = compiled_rag(my_question)
    print(f"Question: {my_question}")
    print(f"Predicted Answer: {pred.answer}")
    print(f"Retrieved Contexts (truncated): {[c[:200] + '...' for c in pred.context]}")

    from dspy.evaluate.evaluate import Evaluate

    evaluate_on_hotpotqa = Evaluate(
        devset=devset,
        num_threads=args.num_threads,
        display_progress=True,
        display_table=5,
    )

    metric = dspy.evaluate.answer_exact_match
    evaluate_on_hotpotqa(compiled_rag, metric=metric)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://localhost")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--api-key", type=str, default="local")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--num-threads", type=int, default=32)
    parser.add_argument("--dev-size", type=int, default=150)
    args = parser.parse_args()
    main(args)
