import chromadb
# Point to your storage folder defined in .env
client = chromadb.PersistentClient(path="storage/chroma")
collection = client.get_collection("academiq_chunks")
print(f"Total chunks in database: {collection.count()}")