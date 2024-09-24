import os
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

class StructuredQuery(BaseModel):
    query: str
    filter: str

class QueryConstructor:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('openai_api_key'))
        self.function_description = [
            {
                "name": "structured_query_extractor",
                "description": "Extract structured query and filter from given question",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The main query to search for",
                        },
                        "filter": {
                            "type": "string",
                            "description": "The filter to apply on metadata, in the format of a boolean expression",
                        },
                    },
                    "required": ["query", "filter"],
                },
            }
        ]

    def construct_query(self, question, metadata_field_info):
        try:
            prompt = f"Given the question '{question}', extract a structured query and filter based on the following metadata: {metadata_field_info}"
            response = self.client.beta.chat.completions.parse(
                model='gpt-4o-mini-2024-07-18',
                messages=[{'role':'user','content':prompt}],
                response_format=StructuredQuery
            )
            return response.choices[0].message.parsed
        except Exception as e:
            print(e)
            raise


if __name__ == "__main__":
    query_constructor = QueryConstructor()
    question = "What is the capital of France?"
    metadata_field_info = "country"
    structured_query = query_constructor.construct_query(question, metadata_field_info)
    print(structured_query)
