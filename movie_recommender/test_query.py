from endee import Endee
from sentence_transformers import SentenceTransformer

print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Connecting to Endee...")
try:
    endee_client = Endee()
    index = endee_client.get_index(name="movies")
    print("Got index.")
    
    query_vector = model.encode(["A futuristic movie where robots take over the world"])[0].tolist()
    
    print("Querying...")
    results = index.query(vector=query_vector, top_k=2)
    print(f"Type of results: {type(results)}")
    print(f"Results: {results}")
    
    if len(results) > 0:
        r = results[0]
        print(f"Type of first result: {type(r)}")
        print(f"Attributes: {dir(r)}")
except Exception as e:
    print(f"Error: {e}")
