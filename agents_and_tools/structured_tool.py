from langchain.tools import StructuredTool


def search_function(query: str):
    return "LangChain"

search_tool = StructuredTool.from_function(func=search_function,
                                      name="search",
                                      description="Useful for when you need to answer question about current events")


print("------ Details of the Tool ------")
print(search_tool.name)
print(search_tool.description)
print(search_tool.args)

print("\n------ Tool Response ------")
print(search_tool.invoke("Search item for me."))
