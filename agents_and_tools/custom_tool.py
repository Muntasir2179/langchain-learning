from pydantic import BaseModel, Field
from langchain.tools import tool


class SearchInput(BaseModel):
    query: str = Field(description="should be a search query")

# in the tool decorator here we are defining details for the custom tool
@tool("search-tool", args_schema=SearchInput, return_direct=True)
def search(query: str) -> str:
    # provided docstring will be treated as the tool description
    """Look up things online.""" 
    return "LangChain"


print("------ Details of the Tool ------")
print(search.name)
print(search.description)
print(search.args)


print("\n------ Tool Response ------")
result = search.invoke("Search some item for me.")
print(result)
