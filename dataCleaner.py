import ast
import pandas as pd
from typing import List
from langchain.schema import Document


def convert_df_to_documents(df: pd.DataFrame) -> List[Document]:
    documents = []
    
    for _, row in df.iterrows():
        page_content = ""
        
        for column in df.columns:
            if pd.notna(row[column]):
                if column in ['positive_refer_ratio', 'reviews_refer_ratio']:
                    try:
                        ratios = ast.literal_eval(row[column])
                        page_content += f"{column}:\n"
                        for key, value in ratios.items():
                            if pd.notna(value):
                                page_content += f"  {key}: {value}\n"
                    except:
                        print(f"Error parsing {column} for {row['game']}")
                else:
                    page_content += f"{column}: {row[column]}\n"
        
        document = Document(
            page_content=page_content.strip(),
            metadata={} 
        )
        
        documents.append(document)
    
    return documents


# Example usage:
if __name__ == "__main__":
    df = pd.read_csv('modified_cleaned_dataset.csv', encoding='latin1')
    docs = convert_df_to_documents(df)
    print(docs[2])
    print(len(docs))