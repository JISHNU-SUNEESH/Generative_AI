from crewai import Task
from Agents import researcher,writer
from Tools import search_tool
research = Task(
    description='Research the latest trends in the AI industry and provide a summary.',
    expected_output='A summary of the top 3 trending developments in the AI industry with a unique perspective on their significance.',
    tools=[search_tool],
    agent=researcher
)

write = Task(
    description='Write an engaging blog post about the AI industry, based on the research analyst’s summary. Draw inspiration from the latest blog posts in the directory.',
    expected_output='A 4-paragraph blog post formatted in markdown with engaging, informative, and accessible content, avoiding complex jargon.',
    tools=[search_tool],
    agent=writer,
    async_execution=False,
    output_file='blog-posts/new_post.md',  # The final blog post will be saved here

)