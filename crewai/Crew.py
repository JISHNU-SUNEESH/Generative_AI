from crewai import Crew,Process
from Agents import researcher,writer
from Tools import search_tool
from Tasks import research,write

crew = Crew(
    agents=[researcher, writer],
    tasks=[research, write],
    process=Process.sequential,
    # verbose=True,
    # planning=True,  # Enable planning feature
    # planning_llm=llm
)

result=crew.kickoff(inputs={'topic':'AI'})
print(result)
