from chromadb import PersistentClient

# ChromaDB'yi bağla
persist_path = "./chromadb"  # Path'inizi doğru ayarlayın
chroma_client = PersistentClient(persist_path)
collection = chroma_client.get_or_create_collection(name="user_preferences")

# Koleksiyondaki tüm verileri sıfırlamak
try:
    # Önce koleksiyondaki tüm kayıtları alın
    results = collection.get()
    all_ids = results.get("ids", [])

    # Eğer kayıtlar varsa sil
    if all_ids:
        collection.delete(ids=all_ids)  # Tüm verileri ID'lere göre sil
        print("Tüm veriler başarıyla silindi.")
    else:
        print("Koleksiyonda zaten veri yok.")
except Exception as e:
    print(f"Veriler silinirken bir hata oluştu: {e}")
