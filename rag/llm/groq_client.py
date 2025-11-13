import os
from groq import Groq

def groq_stream_completion(prompt: str,
                           model: str = "openai/gpt-oss-20b",
                           temperature: float = 0.2,
                           max_completion_tokens: int = 2048) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY in environment.")
    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        top_p=1,
        reasoning_effort="medium",
        stream=True,
        stop=None
    )
    out = ""
    for chunk in completion:
        piece = chunk.choices[0].delta.content or ""
        print(piece, end="", flush=True)
        out += piece
    print()
    return out
