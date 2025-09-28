
'''
1. get_agent_runnable()
'''

from retriever import build_pinecone_retriever, fetch_facts_for_question 
from config import llm_gen
from langchain_core.messages import SystemMessage, ToolMessage, BaseMessage, HumanMessage
from langgraph.graph import StateGraph, add_messages, END
from typing import TypedDict, Sequence
from typing_extensions import Annotated
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools import tool
#####################################################################################
# system_prompt = """
# # You are an intelligent AI assistant that answers questions STRICTLY based on retrieved sources.

# # 1.  **Analyze the user's query** to determine the best retrieval strategy.
# #     - You can use both tools to give the best results.
# #     - Use the `build_pinecone_retriever` for broad, semantic, or keyword-based questions.
# #     - Use the `fetch_facts_for_question` for specific questions about entities and their relationships.
    
# # 2.  **Call the necessary tools** with the appropriate arguments.

# # 3.  **Synthesize the answer** based *only* on the information returned by the tools.

# # 4.  **If the tools do not return relevant information**, state that you could not find an answer in the provided documents and do not add any information.
# # # """

# For Testing Tools 
system_prompt ='Strictly use both build_pinecone_retriever and fetch_facts_for_question tools to retrieve information then answer the question base on that retrieved information.'
# system_prompt = 'Strictly use the fetch_facts_for_question tools to answer the query'

#####################################################################################
# Tools setup

# TOOL TEST
# @tool
# def get_weather():
#     'get weather in Antipolo City'
#     return 'It is Raning in Antipolo'

tools = [
        #  get_weather, 
         build_pinecone_retriever, 
         fetch_facts_for_question
         ]
model = llm_gen.bind_tools(tools)

#####################################################################################
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

tools_dict = {our_tool.name: our_tool for our_tool in tools}

# --- Agent Nodes ---
def call_llm_with_tools(state:AgentState) -> AgentState:
    messages = list(state['messages'])
    messages = [SystemMessage(content=system_prompt)] + messages
    messages = model.invoke(messages)
    return {'messages': [messages]}


def router(state: AgentState):
    'check if the last message contains a tool calls'
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0


def take_action(state: AgentState) -> AgentState:
    '''Execute tool calls from LLM'sresponse'''

    tool_calls = state['messages'][-1].tool_calls
    results = []

    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")


        if not t['name'] in tools_dict:          # check if tool exists
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name. Please retry and select a tool from the list of available tools."
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ""))
            # args = t.get('args', {})
            # result = tools_dict[t['name']].invoke(args)
            print(f"Result length: {len(str(result))}")

        # append ToolMessage
        results.append(
            ToolMessage(
                tool_call_id=t['id'],
                name=t['name'],
                content=str(result)
            )
        )

    print("Tools execution complete. Back to the model!")
    # print(type(results), results)   # should be <class 'list'>
    return {'messages': results}


#####################################################################################
# --- Compile the Agent Graph ---
# import to fast api
def get_agent_runnable():
    """Compiles and returns the LangGraph agent."""
    graph = StateGraph(AgentState)
    graph.add_node('llm_with_tool', call_llm_with_tools)
    graph.add_node('take_action', take_action)

    graph.add_conditional_edges(
        'llm_with_tool',
        router, # hasstr tool_calls
        {True:'take_action', False: END}
    )

    graph.add_edge('take_action', 'llm_with_tool')
    graph.set_entry_point('llm_with_tool')

    memory = MemorySaver() 
    rag_agent =graph.compile(checkpointer=memory)

    return rag_agent

# ======================================================
# Quick smoke test from CLI (optional)
# ======================================================

# if __name__ == "__main__":
#     print('\n==== RAG AGENT ====')
              
#     # question = 'Do you know the weather in Antipolo?'
#     question = 'I want cokies how about you?'
#     # question = 'Tell me about brice hernandez?'

#     initial = {"messages": [HumanMessage(content=question)]}
#     config = {"configurable": {"thread_id": '6129'}}
#     response = get_agent_runnable().invoke(initial, config=config)
#     print(response['messages'][-1].content)


