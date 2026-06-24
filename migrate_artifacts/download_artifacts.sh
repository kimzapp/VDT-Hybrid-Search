# download all files
curl -L -o ./paraphrased-query.zip https://www.kaggle.com/api/v1/datasets/download/hongkimgip/paraphrased-query
curl -L -o ./bm25-index.zip https://www.kaggle.com/api/v1/datasets/download/hongkimgip/bm25-index
curl -L -o ./cache-artifacts.zip https://www.kaggle.com/api/v1/datasets/download/hongkimgip/cache-artifacts
curl -L -o ./runs-artifacts.zip https://www.kaggle.com/api/v1/datasets/download/hongkimgip/runs-artifacts
curl -L -o ./bm25-index-vi.zip https://www.kaggle.com/api/v1/datasets/download/hongkimgip/bm25-index-vi
curl -L -o ./bge-small-en-v15-embedding-faiss.zip https://www.kaggle.com/api/v1/datasets/download/hongkimgip/bge-small-en-v15-embedding-faiss
curl -L -o ./dense-index-vi.zip https://www.kaggle.com/api/v1/datasets/download/hongkimgip/dense-index-vi


# unzip all files
unzip bge-small-en-v15-embedding-faiss.zip
unzip bm25-index.zip
unzip bm25-index-vi.zip
unzip cache-artifacts.zip
unzip dense-index-vi.zip
unzip paraphrased-query.zip
unzip runs-artifacts.zip

# remove zip files
rm bge-small-en-v15-embedding-faiss.zip
rm bm25-index.zip
rm bm25-index-vi.zip
rm cache-artifacts.zip
rm dense-index-vi.zip
rm paraphrased-query.zip
rm runs-artifacts.zip