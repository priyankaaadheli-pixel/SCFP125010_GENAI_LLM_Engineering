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

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class EMI(BaseModel):
    principal: float = Field(gt=0, description="Principal amount")
    annual_rate: float = Field(ge=0, le=30, description="Annual Interest Rate")
    tenure_months: int = Field(gt=0, description="Loan tenure in months")

def CalculateEMI(principal: float, annual_rate: float, tenure_months: int) -> str:
    monthly_rate = annual_rate / 100 / 12
    if monthly_rate == 0:
        emi = principal / tenure_months
    else:
        emi = principal * monthly_rate * (1 + monthly_rate) ** tenure_months / ((1 + monthly_rate) ** tenure_months - 1)
    return f"Monthly EMI:{emi:.2f}"

emi_tool = StructuredTool.from_function(
    func=CalculateEMI,
    name="emiFunction",
    description="Calculate monthly loan EMI. Inputs: principal, annual_rate, tenure_months.",
    args_schema=EMI
)

#Binding Tools with llm
tools = [emi_tool]
llm_with_tools = llm.bind_tools(tools)


user_query = "What's my EMI for a 10 lakh loan at 9% for 24 months?"
messages = [{"role": "user", "content": user_query}]
ai_message = llm_with_tools.invoke(messages)
print(ai_message.tool_calls)

tool_messages = []
for call in ai_message.tool_calls:
    selected_tool = next(t for t in tools if t.name == call["name"])
    result = selected_tool.invoke(call["args"])
    print("\nManually executed tool result:")
    print(result)
    tool_messages.append({"role": "tool", "tool_call_id": call["id"], "content": str(result)})

messages.append(ai_message)
messages.extend(tool_messages)
ans = llm_with_tools.invoke(messages)
print("\nFinal answer:")
print(ans.content)