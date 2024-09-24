## Run the main text summarization routine. Assume this is the main file to run.

import argparse

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader

def main():
    # Assume a single file argument is passed in.
    parser = argparse.ArgumentParser(description="Summarize a text file.")
    parser.add_argument("file", help="The text file to summarize.")
    args = parser.parse_args()

    doc = TextLoader(args.file).load()

    prompt = ChatPromptTemplate.from_messages(
        [("system", "Write a concise one-page summary of the following:\\n\\n{context}")]
    )

    llm = ChatOpenAI(model="gpt-4o-mini")

    chain = create_stuff_documents_chain(llm, prompt)

    result = chain.invoke({"context": doc})
    print(result)



if __name__ == "__main__":
    main()