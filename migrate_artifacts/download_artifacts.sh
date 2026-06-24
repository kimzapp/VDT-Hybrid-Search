# download all files
gdown "https://drive.google.com/open?id=1cRext5L6n6wfzX2rnFctcvZac4Q-p_In" -O bge_small_en_v1.5_embedding_faiss.zip 
gdown "https://drive.google.com/open?id=1eglaqwloGB3m02QxHnMbqnLFzXHNJAqb" -O bm25_index.zip 
gdown "https://drive.google.com/open?id=1cEcwGtloy_mfo2ljlFpY140tsgM7Xfiw" -O bm25_index_vi.zip 
gdown "https://drive.google.com/open?id=1X0dtLmejxI3QVN8jx7fQwMDpE-6KEI7L" -O cache.zip 
gdown "https://drive.google.com/open?id=1IhYBymIpM9ZzGaWMffLC3douQcj-NCPT" -O dense_index_vi.zip 
gdown "https://drive.google.com/open?id=1qLwbCScylC9TUUbHT8oiTiVQPVdoXS6i" -O paraphrased_queries.zip 
gdown "https://drive.google.com/open?id=1fSOfiqSa6pmc0WARBj_ZfbxhnJSmdJWE" -O runs.zip

# unzip all files
unzip bge_small_en_v1.5_embedding_faiss.zip
unzip bm25_index.zip
unzip bm25_index_vi.zip
unzip cache.zip
unzip dense_index_vi.zip
unzip paraphrased_queries.zip
unzip runs.zip

# remove zip files
rm bge_small_en_v1.5_embedding_faiss.zip
rm bm25_index.zip
rm bm25_index_vi.zip
rm cache.zip
rm dense_index_vi.zip
rm paraphrased_queries.zip
rm runs.zip