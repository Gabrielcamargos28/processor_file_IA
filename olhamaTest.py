from ollama import chat
from ollama import ChatResponse

response: ChatResponse = chat(model='deepseek-r1:latest', messages=[
  {
    'role': 'user',
    'content': 'Who is Gabriel Antonio Pereira de Camargos?',
  },
])
print(response['message']['content'])
print(response.message.content)