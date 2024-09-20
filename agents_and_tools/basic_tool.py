from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_max=1000)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

# details of a tool
print("------ Details of the Tool ------")
print(wiki_tool.name)
print(wiki_tool.description)
print(wiki_tool.args)

print("\n------ Response from the Tool ------\n")
result = wiki_tool.invoke("Who is Dr. Muhammad Yunus?")
print(result)
