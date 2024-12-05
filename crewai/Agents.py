from crewai import Agent,LLM
from Tools import search_tool
from dotenv import load_dotenv
import os
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI

os.environ["GEMINI_API_KEY"]=os.getenv("GEMINI_API_KEY")

llm=LLM(model="gemini/gemini-1.5-flash",
                           temperature=0.5,
                           verbose=True,
                           )


# os.environ["MISTRAL_API_KEY"]=os.getenv("MISTRAL_API_KEY")
# os.environ["HUGGINGFACE_API_KEY"]=os.getenv("HUGGINGFACE_API_KEY")

# llm = LLM(
#     model="mistral/mistral-large-latest",
#     temperature=0.7
# )

# llm = LLM(
#     model="huggingface/meta-llama/Meta-Llama-3.1-8B-Instruct",
#     base_url="https://huggingface.co/models?inference=warm&pipeline_tag=text-generation"
# )



researcher = Agent(
    role='Market Research Analyst',
    goal='Provide up-to-date market analysis of the {topic} industry',
    backstory='An expert analyst with a keen eye for market trends.',
    tools=[search_tool],
    llm=llm,
    verbose=True,
    memory=True,
    allow_delegation=True
)

writer = Agent(
    role='Content Writer',
    goal='Craft engaging blog posts about the {topic} industry',
    backstory='A skilled writer with a passion for technology.',
    tools=[search_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False
)
