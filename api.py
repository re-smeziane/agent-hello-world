from dotenv import load_dotenv
from groq import Groq
from tavily import TavilyClient
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import json

load_dotenv(override=False)


client = Groq(api_key=os.getenv("GROQ_API_KEY"))
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Outils Python ---
def search_web(query: str) -> str:
    results = tavily.search(query=query, max_results=5)
    output = ""
    for r in results["results"]:
        output += f"Titre: {r['title']}\n"
        output += f"URL: {r['url']}\n"
        output += f"Contenu: {r['content']}\n\n"
    return output

def generate_report(title: str, summary: str, key_points: list, sources: list) -> str:
    report = {
        "title": title,
        "summary": summary,
        "key_points": key_points,
        "sources": sources
    }
    return json.dumps(report, ensure_ascii=False)

# --- Description des outils pour le LLM ---
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
    },
    {
        "type": "function",
        "function": {
            "name": "generate_report",
            "description": "Génère un rapport structuré avec un titre, un résumé, des points clés et les sources. Toujours appeler cet outil en dernier pour formater la réponse finale.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Le titre du rapport"
                    },
                    "summary": {
                        "type": "string",
                        "description": "Un résumé de 2-3 phrases"
                    },
                    "key_points": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Liste de 3 à 5 points clés"
                    },
                    "sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Liste des URLs des sources utilisées"
                    }
                },
                "required": ["title", "summary", "key_points", "sources"]
            }
        }
    }
]


# --- Boucle agent ---
def run_agent(user_message: str) -> dict:
    messages = [
        {
            "role": "system",
            "content": """Tu es un assistant de recherche expert et rigoureux. Tu dois suivre ces étapes obligatoires :

            1. RECHERCHE : Fais exactement 3 recherches avec search_web en couvrant des angles différents :
            - Une recherche générale sur le sujet
            - Une recherche sur les dernières actualités ou développements récents
            - Une recherche sur les implications, impacts ou perspectives futures

            2. ANALYSE : Après tes recherches, synthétise les informations de façon approfondie.

            3. RAPPORT : Termine TOUJOURS en appelant generate_report avec :
            - Un titre précis et informatif
            - Un résumé dense de 4-5 phrases qui couvre l'essentiel
            - 5 points clés détaillés et concrets (pas vagues)
            - Toutes les URLs des sources trouvées

            Sois précis, factuel et exhaustif. Évite les généralités."""
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    searches_done = []
    final_report = None
    max_iterations = 5  # sécurité anti-boucle infinie
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"🔄 Itération {iteration}/{max_iterations}")

        try:
            response = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=messages,
                tools=tools,
                tool_choice="auto",
                parallel_tool_calls=False
            )
        except Exception as e:
            print(f"❌ Erreur LLM : {e}")
            raise Exception(f"Erreur lors de l'appel au LLM : {str(e)}")

        message = response.choices[0].message

        if message.tool_calls:
            messages.append(message)

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name

                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    print(f"❌ Arguments invalides pour {tool_name}")
                    continue

                if tool_name == "search_web":
                    print(f"🔍 Recherche : {tool_args['query']}")
                    searches_done.append(tool_args["query"])
                    try:
                        result = search_web(**tool_args)
                    except Exception as e:
                        print(f"❌ Erreur Tavily : {e}")
                        result = "Erreur lors de la recherche, continue avec les informations disponibles."

                elif tool_name == "generate_report":
                    print(f"📝 Génération du rapport...")
                    try:
                        result = generate_report(**tool_args)
                        final_report = tool_args
                    except Exception as e:
                        print(f"❌ Erreur generate_report : {e}")
                        result = "Erreur lors de la génération du rapport."

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })
                    break

                else:
                    print(f"⚠️ Outil inconnu : {tool_name}")
                    result = "Outil non disponible."

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })

            if final_report:
                break

        else:
            if not final_report:
                final_report = {
                    "title": "Résultat de recherche",
                    "summary": message.content,
                    "key_points": [],
                    "sources": []
                }
            break

    if not final_report:
        raise Exception("L'agent n'a pas pu générer de rapport après 5 itérations.")

    return {
        "searches": searches_done,
        "report": final_report
    }

# --- Modèles de données ---
class ResearchRequest(BaseModel):
    question: str


# --- Routes API ---
@app.get("/")
def root():
    return FileResponse("static/index.html")

@app.post("/research")
def research(request: ResearchRequest):
    try:
        result = run_agent(request.question)
        return result
    except Exception as e:
        print(f"❌ Erreur agent : {e}")
        return {
            "error": str(e),
            "searches": [],
            "report": {
                "title": "Erreur",
                "summary": "Une erreur s'est produite lors de la recherche. Réessaie dans quelques instants.",
                "key_points": [],
                "sources": []
            }
        }

