# download all files
curl -L -o ./paraphrased-query.zip https://www.kaggle.com/api/v1/datasets/download/hongkimgip/paraphrased-query
curl -L -o ./bm25-index.zip https://www.kaggle.com/api/v1/datasets/download/hongkimgip/bm25-index
curl -L -o ./cache-artifacts.zip https://www.kaggle.com/api/v1/datasets/download/hongkimgip/cache-artifacts
curl -L -o ./runs-artifacts.zip https://www.kaggle.com/api/v1/datasets/download/hongkimgip/runs-artifacts
curl -L -o ./bm25-index-vi.zip https://www.kaggle.com/api/v1/datasets/download/hongkimgip/bm25-index-vi
curl -L -o ./bge-small-en-v15-embedding-faiss.zip https://www.kaggle.com/api/v1/datasets/download/hongkimgip/bge-small-en-v15-embedding-faiss
curl -L -o ./dense-index-vi.zip https://www.kaggle.com/api/v1/datasets/download/hongkimgip/dense-index-vi

# unzip all files
unzip bge-small-en-v15-embedding-faiss.zip -d bge_small_en_v1.5_embedding_faiss
rm bge-small-en-v15-embedding-faiss.zip

unzip bm25-index.zip -d bm25_index
rm bm25-index.zip

unzip bm25-index-vi.zip -d bm25_index_vi
rm bm25-index-vi.zip

unzip cache-artifacts.zip -d cache
rm cache-artifacts.zip

unzip dense-index-vi.zip -d dense_index_vi
rm dense-index-vi.zip

unzip paraphrased-query.zip -d paraphrased_query
rm paraphrased-query.zip

unzip runs-artifacts.zip -d runs
rm runs-artifacts.zip