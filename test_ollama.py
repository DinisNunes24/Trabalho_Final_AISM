from ollama import Client
client = Client(host="http://127.0.0.1:11434")

try:
    response = client.chat(
        model="gemma3:4b",
        messages=[{"role": "user", "content": "Diz apenas: OK"}],
        options={"temperature": 0}
    )
    print("Sucesso:", response.message.content)
except Exception as e:
    print("Erro:", type(e).__name__, "→", e)