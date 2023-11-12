# DGOW

# Graph Neural Networks on Discriminative Graphs of Words

## Structural Similarity 





  - To generate the *spectral* emmbedding and compute the inter-similarity and the the intra-similarity :
    ```
    cd ./Similarity/FastGAE_embedding
    python generate_embedding_GAE.py --config "MG" --dataset "R8"
    python compute_similarity.py  --config "MG" --dataset "R8"
    ```
  - To generate the *FastGAE* emmbedding and compute the inter-similarity and the the intra-similarity :
    ```
    cd ./Similarity/Spectral_embedding
    python generate_embedding.py --config "MG" --dataset "R8"
    python compute_similarity.py  --config "MG" --dataset "R8"
    ```

## Using DGoW-GNN Model
    ```
    cd ./code
    python main.py --dataset "R8" 
    ```

