
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0
)
@tool
def find_restaurants(cuisine: str, city: str) -> str:
    """Find restaurants matching a cuisine in a city."""

    data = {
        ("Italian", "Pune"): ["Bella Italia", "Little Italy"],
        ("Indian", "Pune"): [
            "Shabree", "Durvankur", "Sukanta",
            "Vaishali", "SP's Biryani House"
        ],
        ("North Indian", "Pune"): [
            "Barbeque Nation", "Punjab Grill", "Shabree"
        ],
        ("South Indian", "Pune"): [
            "Vaishali", "Wadeshwar", "Ram Krishna"
        ]
    }

    return str(data.get((cuisine.title(), city.title()), []))


@tool
def get_restaurant_rating(restaurant_name: str) -> str:
    """Return restaurant rating out of 5."""

    ratings = {
        "Bella Italia": 4.3,
        "Little Italy": 4.1,
        "Shabree": 4.4,
        "Durvankur": 4.2,
        "Sukanta": 4.3,
        "Vaishali": 4.2,
        "SP's Biryani House": 4.1,
        "Barbeque Nation": 4.3,
        "Punjab Grill": 4.2,
        "Wadeshwar": 4.1,
        "Ram Krishna": 4.0,
        "Britannia & Co.": 4.4,
        "Gajalee": 4.3,
        "Trishna": 4.2,
        "The Great Punjab": 4.1,
        "Cafe Madras": 4.4,
        "Ram Ashraya": 4.3,
        "Example Italian Bistro": 4.0
    }

    rating = ratings.get(restaurant_name)

    return (
        f"Sample rating: {rating}/5"
        if rating is not None
        else "Rating unavailable."
    )

@tool
def check_restaurant_open(restaurant_name: str, day: str) -> str:
    """Check restaurant opening status on a particular day."""

    return (
        f"Sample data: {restaurant_name} "
        f"is listed as open on {day}."
    )

@tool
def estimate_cost_for_two(restaurant_name: str) -> str:
    """Return approximate restaurant cost for two in INR."""

    prices = {
        "Bella Italia": 1800,
        "Little Italy": 1600,
        "Shabree": 900,
        "Durvankur": 700,
        "Sukanta": 800,
        "Vaishali": 500,
        "SP's Biryani House": 600,
        "Barbeque Nation": 1800,
        "Punjab Grill": 2000,
        "Wadeshwar": 500,
        "Ram Krishna": 700,
        "Britannia & Co.": 1200,
        "Gajalee": 1500,
        "Trishna": 2000,
        "The Great Punjab": 1400,
        "Cafe Madras": 500,
        "Ram Ashraya": 500,
        "Example Italian Bistro": 2000
    }

    price = prices.get(restaurant_name)

    return (
        f"Sample cost for two: ₹{price}"
        if price is not None
        else "Cost estimate unavailable."
    )
@tool
def suggest_indian_dishes(cuisine: str) -> str:
    """Suggest popular Indian dishes based on cuisine."""

    dishes = {
        "Indian": [
            "Paneer Butter Masala",
            "Dal Makhani",
            "Butter Naan",
            "Veg Biryani",
            "Chole Bhature"
        ],
        "North Indian": [
            "Paneer Tikka",
            "Butter Chicken",
            "Dal Makhani",
            "Amritsari Kulcha",
            "Chole Bhature"
        ],
        "South Indian": [
            "Masala Dosa",
            "Idli Sambar",
            "Medu Vada",
            "Uttapam",
            "Mysore Dosa"
        ],
        "Maharashtrian": [
            "Misal Pav",
            "Puran Poli",
            "Vada Pav",
            "Pithla Bhakri",
            "Bharli Vangi"
        ]
    }

    return str(dishes.get(cuisine.title(), ["Cuisine not found."]))


#Bind all tools to LLM
tools = [
    find_restaurants,
    get_restaurant_rating,
    check_restaurant_open,
    estimate_cost_for_two,
    suggest_indian_dishes
]

llm_with_tools = llm.bind_tools(tools)


#User query
user_query = """
Suggest good Indian restaurants in Pune.
Include their ratings, opening status for today,
rough cost for two, and popular Indian dishes.
"""

messages = [{"role": "user", "content": user_query}]

#First LLM call
ai_message = llm_with_tools.invoke(messages)

print("Model suggested tool_calls:")
print(ai_message.tool_calls)

messages.append(ai_message)


#Execute selected tools
for call in ai_message.tool_calls:

    selected_tool = next(
        t for t in tools if t.name == call["name"]
    )

    result = selected_tool.invoke(call["args"])

    print(f"\nResult from {call['name']}:")
    print(result)

    messages.append({
        "role": "tool",
        "tool_call_id": call["id"],
        "content": str(result)
    })


# Final LLM response
ans = llm_with_tools.invoke(messages)
print("\nFinal Restaurant Recommendations:")
print(ans.content)
