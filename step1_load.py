from langchain_community.document_loaders import TextLoader


# Create the document loader
loader = TextLoader("knowledge.txt")


# Load the document
documents = loader.load()


# Print number of documents
print("Number of documents:", len(documents))


# Print the first document
print("\nDocument:")
print(documents[0])


# Print actual text
print("\nContent:")
print(documents[0].page_content)


# Print metadata
print("\nMetadata:")
print(documents[0].metadata)