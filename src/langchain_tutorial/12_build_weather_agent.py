import asyncio
import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain.tools import tool
from pyowm import OWM

load_dotenv(override=True)

# Initialize OpenWeatherMap client (has weather details for most major cities)
# NOTE: you will need an OpenWeatherMap API key 
OWM_API_KEY = os.getenv("OWM_API_KEY") # Ensure you set this environment variable
if not OWM_API_KEY:
    raise ValueError("OWM_API_KEY environment variable not set.")
owm = OWM(OWM_API_KEY)
mgr = owm.weather_manager()

@tool
def get_current_weather(city: str) -> str:
    """
    Fetches the current weather conditions for a specified city.
    Returns a string describing the weather.
    """
    try:
        observation = mgr.weather_at_place(city)
        weather = observation.weather
        status = weather.status
        temperature = weather.temperature('celsius')['temp']
        return f"Current weather in {city}: {status}, {temperature}°C."
    except Exception as e:
        return f"Could not retrieve weather for {city}. Error: {e}"

async def main():
    # Define the LLM
    llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0) # Or your preferred LLM

    # Get the agent prompt from LangChain Hub
    prompt = hub.pull("hwchase17/react")

    # Define the tools the agent can use
    tools = [get_current_weather]

    # Create the agent
    agent = create_react_agent(llm, tools, prompt)

    # Create the AgentExecutor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # Run the agent with an asynchronous invocation
    while True:
        print("Enter city name to fetch weather: ", end="")
        city = input()
        if len(city.strip()) == 0:
            print("Please enter a city name!")
            continue
        elif city.lower() in ["exit", "bye", "quit"]:
            break
        else:
            response = await agent_executor.ainvoke({"input": f"What's the weather like in {city}?"})
            print(response["output"])

    # response = await agent_executor.ainvoke({"input": "Tell me about the weather in Tokyo."})
    # print(response["output"])
    #
    # response = await agent_executor.ainvoke({"input": "Tell me about the weather in Mumbai."})
    # print(response["output"])

if __name__ == "__main__":
    asyncio.run(main())