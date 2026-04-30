from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from typing import TypedDict

load_dotenv()

# 1. Define the state
class AgentState(TypedDict):
    messages: list

# 2. System prompt - brief description/instructions for the agent
system_prompt = SystemMessage(content="""
You are a helpful assistant representing Matt's professional portfolio.
Answer questions about Matt based on the following information:

ABOUT:
Matt is a software engineer based in the Bay Area with experience in 
both general software engineering and games industry development.

PROJECTS:
- WWE 2K Games: Contributed to the development of multiple WWE 2K games, worked on various gameplay features, and
reworked the data pipeline sytem for content creation and deployment to live servers.
- FryRank: A Yelp-style web app focused on restaurant reviews with an 
  emphasis on fries. Built with AWS Lambda, DynamoDB, and API Gateway.

SKILLS:
- Backend: AWS Lambda, DynamoDB, API Gateway, Python, Java
- Frontend: React
- Other: System design, game design

Keep answers concise and professional. If asked something you don't 
know about Matt, say you don't have that information.
""")

# 3. Initialize the LLM
llm = ChatGroq(model="llama-3.3-70b-versatile")

# 4. Define a node
def chat_node(state: AgentState):
    messages = [system_prompt] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": state["messages"] + [response]}

# 5. Build the graph
graph = StateGraph(AgentState)
graph.add_node("chat", chat_node)
graph.add_edge(START, "chat")
graph.add_edge("chat", END)

# 6. Compile
agent = graph.compile()