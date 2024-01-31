
from sentence_transformers import SentenceTransformer
import pickle
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import streamlit as st

class Embeddings:

  def __init__(self, embedding_dimension='50d', transformer_name="all-MiniLM-L6-v2"):
    """
    Initialize the class
    """
    self.embedding_dimension = embedding_dimension
    self.transformer_name = transformer_name


  def load_glove_embeddings(self, embedding_dimension):
    word_index_temp = "word_index_dict_" + str(embedding_dimension) + "_temp.pkl"
    embeddings_temp = "embeddings_" + str(embedding_dimension) + "_temp.npy"


    # Load word index dictionary
    word_index_dict = pickle.load(open(word_index_temp, "rb"), encoding="latin")

    # Load embeddings numpy
    embeddings = np.load(embeddings_temp)

    return word_index_dict, embeddings


  def get_glove_embedding(self, word,word_index_dict,embeddings, embedding_dimension):
    """
    Retrieve glove embedding of a specific dimension
    """
    if word.lower() in word_index_dict:
      return embeddings[word_index_dict[word.lower()]]
    else:
      return np.zeros(int(embedding_dimension.split("d")[0]))



  def get_sentence_transformer_embedding(self, sentence, transformer_name="all-MiniLM-L6-v2"):
    """
    Encode a sentence using sentence transformer and return embedding
    """

    sentenceTransformer = SentenceTransformer(transformer_name)

    return sentenceTransformer.encode(sentence)



  def get_averaged_glove_embeddings(self, sentence, word_index_dict, embeddings, embedding_dimension):

      embedding = np.zeros(int(embedding_dimension.split("d")[0]))
      
      # Split sentence into words
      words = sentence.split(" ")

      # Initialize a counter for the number of words found in the embeddings
      count = 0

      for word in words:
          # Get the embedding of the word using the get_glove_embedding function
          word_embedding = self.get_glove_embedding(word, word_index_dict, embeddings, embedding_dimension)

          # Add the word's embedding to the total and increment the counter
          embedding += word_embedding
          count += 1

      # If at least one word was found in the embeddings, divide the total embeddings by the number of words
      if count > 0:
          embedding /= count

      return embedding


class Search:

  def __init__(self,embeddings):
    self.embeddings = embeddings


  def cosine_similarity(self, x, y):

    return np.dot(x,y)/max(la.norm(x)*la.norm(y),1e-3)

  def get_topK_similar_categories(self, sentence, categories, K=10, embedding_dimension='50d', embedding_type='glove'):
    """
    Return the most similar categories to a given sentence -
    This is a baseline implementation of a semantic search engine
    """

    # Load word index dictionary and embeddings
    word_index_dict, embeddings = self.embeddings.load_glove_embeddings(embedding_dimension)

    if embedding_type == 'glove':
      # Get the averaged glove embeddings for the sentence
      sentence_embedding = self.embeddings.get_averaged_glove_embeddings(sentence, word_index_dict, embeddings, embedding_dimension)  
    else:
      # Get the sentence transformer embedding for the sentence
      sentence_embedding = self.embeddings.get_sentence_transformer_embedding(sentence)

    # Calculate the cosine similarity between the sentence and each category
    similarities = []
    for category in categories:
      if embedding_type == 'glove':
        # Get the averaged glove embeddings for the category
        category_embedding = self.embeddings.get_averaged_glove_embeddings(category, word_index_dict, embeddings, embedding_dimension)
      else:
        # Get the sentence transformer embedding for the category
        category_embedding = self.embeddings.get_sentence_transformer_embedding(category) 
      similarity = self.cosine_similarity(sentence_embedding, category_embedding)
      similarities.append((category, similarity))

    # Sort the categories by similarity and return the top K
    similarities.sort(key=lambda x: x[1], reverse=True)
    topK_categories = [category for category, similarity in similarities[:K]]
    similarities = [similarity for category, similarity in similarities[:K]]

    return topK_categories,similarities


def plot_pie_chart(categories,scores):
    # Extract categories and their corresponding scores
    categories = categories
    scores = scores

    # Create a new figure and axes
    fig, ax = plt.subplots()

    # Create pie chart on the given axes
    ax.pie(scores, labels=categories, autopct='%1.1f%%')

    # Add title
    ax.set_title("Category Similarity Scores")

    # Return the figure
    return fig

st.title("Search Most Similar Category")
st.subheader(
    "Pass in space separated categories you want this search demo to be about."
)

st.text_input(
    label="Categories", key="categories", value="Flowers Colors Cars Weather Food"
)
print(st.session_state["categories"])
print(type(st.session_state["categories"]))


st.subheader("Pass in an input word or even a sentence")
text_search = st.text_input(
    label="Input your sentence",
    key="text_search",
    value="Roses are red, trucks are blue, and Seattle is grey right now",
)

categories = st.session_state.categories.split(" ")

embeddings = Embeddings()
search = Search(embeddings)

# Find closest word to an input word
if st.session_state.text_search:
    # Glove embeddings
    print("Glove Embedding")
    
    with st.spinner("Obtaining Cosine similarity for Glove..."):
        sorted_categories_glove,scores_glove = search.get_topK_similar_categories(text_search, categories, 10, "50d", "glove")
        

    # Sentence transformer embeddings
    print("Sentence Transformer Embedding")

    with st.spinner("Obtaining Cosine similarity for 384d sentence transformer..."):
        sorted_categories_ST,scores_ST = search.get_topK_similar_categories(text_search, categories, 10, "50d", "Transformer")
    
    
    sorted_cosine_scores_models = {"glove": scores_glove, "transformer": scores_ST}
    sorted_categories_models = {"glove": sorted_categories_glove, "transformer": sorted_categories_ST}

    # Results and Plot Pie Chart for Glove
    print("Categories are: ", st.session_state.categories)
    st.subheader(
        "Closest word I have between: "
        + st.session_state.categories
        + " as per different Embeddings"
    )

    print(f'Sorted Glove = {sorted_categories_glove}')
    print(f'Sorted Transform = {sorted_categories_ST}')

    models = ["Glove", "Sentence Transformer"]
    tabs = st.tabs(models)
    figs = {}
    for model in models:
        #fig, ax = plt.subplots()
        if model == "Glove":
            figs[model] = plot_pie_chart(sorted_categories_models["glove"], sorted_cosine_scores_models["glove"])
            with tabs[0]: st.pyplot(figs[model])
        elif model == "Sentence Transformer":
            figs[model] = plot_pie_chart(sorted_categories_models["transformer"], sorted_cosine_scores_models["transformer"])
            with tabs[1]: st.pyplot(figs[model])

