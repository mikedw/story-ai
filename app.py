import streamlit as st
from openai import OpenAI
from openai.types import CreateEmbeddingResponse, Embedding
from openai.types import ChatModel
from pinecone import Pinecone
import os


# Check if environment variables are present. If not, throw an error
if os.getenv('PINECONE_API_KEY') is None:
    st.error("PINECONE_API_KEY not set. Please set this environment variable and restart the app.")
if os.getenv('PINECONE_INDEX') is None:
    st.error("PINECONE_INDEX not set. Please set this environment variable and restart the app.")
if os.getenv('PINECONE_HOST') is None:
    st.error("PINECONE_HOST not set. Please set this environment variable and restart the app.")
if os.getenv('EMBEDDINGS_MODEL') is None:
    st.error("EMBEDDINGS_MODEL not set. Please set this environment variable and restart the app.")
if os.getenv('GPT_MODEL_NAME') is None:
    st.error("GPT_MODEL_NAME not set. Please set this environment variable and restart the app.")
if os.getenv('OPENAI_API_KEY') is None:
    st.error("OPENAI_API_KEY not set. Please set this environment variable and restart the app.")

st.title("Story AI prototype")
query = st.text_area("Share a story to get feedback on how to improve it.")

if st.button("Get feedback"):
   
    # # get Pinecone API environment variables
    pinecone_api = os.getenv('PINECONE_API_KEY')
    pinecone_index = os.getenv('PINECONE_INDEX')
    embeddings_model_name = os.getenv('EMBEDDINGS_MODEL')
    gpt_model_name = os.getenv('GPT_MODEL_NAME')
    
    # # get OpenAI environment variables
    openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
 
    # Initialize Pinecone client and set index
    pinecone = Pinecone()
    index = pinecone.Index(host=os.getenv('PINECONE_HOST'))
 
    # Convert your query into a vector using Azure OpenAI
    try:
        query_vector = openai.embeddings.create(
            input=query,
            model=embeddings_model_name,
        ).data[0].embedding
    except Exception as e:
        st.error(f"Error calling OpenAI Embedding API: {e}")
        st.stop()
 
    # Search for the most similar vectors in Pinecone
    search_response = index.query(
        top_k=3,
        vector=query_vector,
        include_metadata=True)

    chunks = [item["metadata"]['text'] for item in search_response['matches']]
 
    # Combine texts into a single chunk to insert in the prompt
    joined_chunks = "\n".join(chunks)

    # Write the selected chunks into the UI
    # with st.expander("Relevant material"):
    #     for i, t in enumerate(chunks):
    #         t = t.replace("\n", " ")
    #         st.write("Chunk ", i, " - ", t)
    
    with st.spinner("Summarizing..."):
        try:
            # Build the prompt
            # prompt = f"""
            # Answer the following question based on the context below. Don't try to make up an answer. Do not answer beyond this context.
            # ---
            # QUESTION: {query}                                            
            # ---
            # CONTEXT:
            # {joined_chunks}
            # """
            prompt = f"""
            Provide feedback on the shared story, try to suggest improvements and learning points from context below. Keep response less than 200 words.
            ---
            STORY: {query}                                            
            ---
            CONTEXT:
            {joined_chunks}
            """
 
            # Run chat completion using GPT-4
            response = openai.chat.completions.create(
                model=gpt_model_name,
                messages=[
                    { "role": "system", "content":  "You are a Q&A assistant." },
                    { "role": "user", "content": prompt }
                ],
                temperature=0.7,
                max_tokens=1000
            )
 
            # Write query answer
            st.markdown("### Feedback:")
            st.write(response.choices[0].message.content)
   
   
        except Exception as e:
            st.error(f"Error with OpenAI Chat Completion: {e}")