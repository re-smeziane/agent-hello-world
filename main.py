from dotenv import load_dotenv
from groq import Groq
from tavily import TavilyClient
import os
import json

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# --- 1. Notre vrai outil de recherche ---
def search_web(query: str) -> str:
    results = tavily.search(query=query, max_results=3)
    # On formate les résultats en texte simple
    output = ""
    for r in results["results"]:
        output += f"Titre: {r['title']}\n"
        output += f"Contenu: {r['content']}\n\n"
    return output

# --- 2. Description de l'outil pour le LLM ---
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Recherche des informations récentes sur le web",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "La requête de recherche"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# --- 3. La boucle agent ---
def run_agent(user_message: str):
    print(f"\n👤 User: {user_message}")
    print("-" * 50)

    messages = [
        {
            "role": "system",
            "content": "Tu es un assistant de recherche. Utilise l'outil search_web pour trouver des informations récentes et fiables avant de répondre."
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    while True:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            parallel_tool_calls=False
        )

        message = response.choices[0].message

        if message.tool_calls:
            messages.append(message)

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                print(f"🔍 Recherche en cours : {tool_args['query']}")

                if tool_name == "search_web":
                    result = search_web(**tool_args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
        else:
            print(f"\n🤖 Agent:\n{message.content}")
            break

# --- 4. Test ---
run_agent("Quelles sont les dernières avancées en intelligence artificielle en 2025 ?")