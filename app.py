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

st.header("Customizing LLM prompt")
default_prompt_instruction = """You are a storytelling feedback  assistant. You will first evaluate the provided story without the context, and determine if it has or implies an arc with a beginning, middle, and end. If so, suggest improvements and learning points from the provided context.  Keep response less than 250 words.

-------
STORY: {story_from_user}                                            
-------
CONTEXT: {course_context}
"""

sidebar = st.sidebar
with sidebar:
    st.text("Customize prompt instrutions")
    current_prompt = st.empty()
    prompt = current_prompt.text_area(label="Add your custom prompt", value=default_prompt_instruction, height=300, key="1")
    if st.button("Reset", type="primary"):
         prompt = default_prompt_instruction
         promtpt = current_prompt.text_area(label="Add your custom prompt", value=default_prompt_instruction, height=300, key="2")

st.header("Story input from user")

query = st.text_area("Share a story to get feedback on how to improve it.", height=300)

if st.button("Get feedback") or query:
   
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
            prompt = prompt.replace("{story_from_user}", query)
            prompt = prompt.replace("{course_context}", joined_chunks)
 
            # Run chat completion using GPT-4
            response = openai.chat.completions.create(
                model=gpt_model_name,
                messages=[
                    { "role": "system", "content":  "You are an assistant providing feedback on stories." },
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