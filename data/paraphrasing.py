from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import random

print("Loading paraphrase Model...")

para_tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
para_model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")

print("Model Loaded!")

def paraphrase(text: str) -> str:
    """
    Reform the categorical messages from generate_inqueries using T5.
    """

    input_text = f"paraphrase: {text} </s>"
    input_ids = para_tokenizer(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding="max_length"
    ).input_ids

    outputs = para_model.generate(
        input_ids,
        max_length=512,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.9,
        num_return_sequences=1,
        early_stopping=True
    )

    result = para_tokenizer.decode(
        outputs[0],
        skip_special_token=True,
        clean_up_tokenization_spaces=True
    )

    return result

def validate_paraphrase(paraphrased: str, original: str) -> str:
    """
    Checks if the T5 output is usable.
    If it looks broken, returns the original message with a unique suffix.
    
    3 checks:
    1. Too short → probably lost all meaning
    2. Too long → probably repeated itself (T5 bug)
    3. Mostly non-French characters → translation broke
    """
    p = paraphrased.strip()
    o = original.strip()

    # Check 1: too short (less than 30% of original length)
    if len(p) < len(o) * 0.3:
        return o + f" [ref-{random.randint(10000, 99999)}]"

    # Check 2: too long (more than 3x original length — T5 looped)
    if len(p) > len(o) * 3:
        return o + f" [ref-{random.randint(10000, 99999)}]"

    # Check 3: less than 40% of words are real French-looking words
    # (at least 3 characters long — filters out single letters and garbage)
    words = p.split()
    real_words = [w for w in words if len(w) >= 3]
    if len(words) > 0 and len(real_words) / len(words) < 0.4:
        return o + f" [ref-{random.randint(10000, 99999)}]"

    return p