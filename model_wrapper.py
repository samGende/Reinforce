from transformers import AutoTokenizer


class TokenWrapper:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, inputs):
        inputs = self.tokenizer(inputs, return_tensors='pt')
        outputs = self.model.generate(**inputs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
          