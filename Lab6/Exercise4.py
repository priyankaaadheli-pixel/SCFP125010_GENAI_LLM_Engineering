from langchain_groq import ChatGroq
from langchain_core.tools import tool
from dotenv import load_dotenv
import os

load_dotenv()

#llm model
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0
)

@tool
def get_weather(city: str) -> str:
    """Return sample current weather temperature and conditions for a city."""
    sample_weather = {
         "Solapur": {"temperature": 18, "conditions": "Cool"},
        "Mumbai": {"temperature": 30, "conditions": "humid and partly cloudy"},
        "Pune": {"temperature": 25, "conditions": "partly cloudy"},
        "Delhi": {"temperature": 22, "conditions": "cool and clear"}
    }
    data = sample_weather.get(city.title())
    if not data:
        return f"Weather data unavailable for {city}. Do not guess the weather."
    return str(data)

@tool
def recommend_outfit_based_on_weather(temperature: float, conditions: str) -> str:
    """Recommend an outfit based on temperature in Celsius and weather conditions."""
    if temperature >= 30:
        outfit = "Summer Tops"
    elif temperature <= 18:
        outfit = "Warm Clothes"
    else:
        outfit = "Any comfortable clothes"
    if "rain" in conditions.lower():
        outfit += " and bring an umbrella or raincoat"
    return f"Suggestion: wear {outfit}. Conditions: {conditions}."


#Binding tools with LLM
tools = [get_weather, recommend_outfit_based_on_weather]
llm_with_tools = llm.bind_tools(tools)
user_query = "What should I wear today in Solapur?"
messages = [{"role": "user", "content": user_query}]

msg_1 = llm_with_tools.invoke(messages)
print("Round 1")
print(msg_1.tool_calls)
messages.append(msg_1)

weather_calls = [c for c in msg_1.tool_calls if c["name"] == "get_weather"]
weather_result = None
for call in weather_calls:
    weather_result = get_weather.invoke(call["args"])
    print("\nFetched weather data:")
    print(weather_result)
    messages.append({"role": "tool", "tool_call_id": call["id"], "content": str(weather_result)})

msg_2 = llm_with_tools.invoke(messages)
print("\nRound 2")
print(msg_2.tool_calls)
messages.append(msg_2)

for call in msg_2.tool_calls:
    if call["name"] == "recommend_outfit":
        result = recommend_outfit_based_on_weather.invoke(call["args"])
        print("\nOutfit tool result:")
        print(result)
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": str(result)})

ans = llm_with_tools.invoke(messages)
print("\nFinal recommendation of the outfit:")
print(ans.content)