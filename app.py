import streamlit as st
import requests as rq
from bs4 import BeautifulSoup
import json
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
import xml.etree.ElementTree as ET
from stqdm import stqdm
# from google.generativeai import GenerativeModel
from langchain.embeddings import OpenAIEmbeddings
# from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv(), override=True)

prompt = '''
You are a highly intelligent and articulate Chat Bot with deep understanding of various subjects. 
Your task is to analyze the given YouTube transcript and answer questions based on it.

Instructions:
1. You will be provided with the transcript text of a YouTube video and its video ID.
2. A user will ask questions related to the content of the transcript.
3. Your answers must be based on the information provided in the transcript. Do not include any information not found in the transcript.
4. Present your answers in bullet points for clarity.
5. Ensure that your response is concise, accurate, and within 250 words.

Here is the transcript text and video ID and user_question:
'''

def get_vectorstore(chunks):
    ''' 
    to convert a list of text chunks into a vector store.
    Uses OpenAIEmbeddings to generate embeddings for each text chunk.
    The embeddings are stored in a FAISS index.
    Takes a list of text chunks as input and returns a FAISS vector store.
    '''
    # print(f'Chunks: {chunks}')
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    print(vectorstore)
    return vectorstore

def get_text_chunks(text):
    '''
    to split a given text into smaller chunks.
    Uses CharacterTextSplitter to divide the text based on the specified separator, chunk size, and overlap.
    Takes a string as input and returns a list of text chunks.
 '''
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=100, chunk_overlap=10, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_video_ids(video_list_id):
    URL_VIDEO_LIST_FORMAT = 'https://www.youtube.com/watch?v=&list={}'
    video_list_url = URL_VIDEO_LIST_FORMAT.format(video_list_id)
    response = rq.get(video_list_url)
    html_content = response.text
    soup = BeautifulSoup(html_content, 'html.parser')
    script_tags = soup.find_all('script')   
    yt_initial_data = None
    for script in script_tags:
        if 'ytInitialData' in script.text:  
            json_text = script.text.split('ytInitialData = ')[1].split('</script>')[0].strip()[:-1]
            yt_initial_data = json.loads(json_text)
            break
    if yt_initial_data:
        video_list = yt_initial_data['contents']['twoColumnWatchNextResults']['playlist']['playlist']['contents']
        return [video['playlistPanelVideoRenderer']['videoId'] for video in video_list if 'playlistPanelVideoRenderer' in video]
    else:
        print("ytInitialData not found in the HTML of the webpage.")
        return []

def get_transcript(video_id):
    URL_TRANSCRIPT_FORMAT = 'https://youtubetranscript.com/?server_vid2={}'
    transcript_url = URL_TRANSCRIPT_FORMAT.format(video_id)
    response = rq.get(transcript_url)
    xml_data = response.text
    root = ET.fromstring(xml_data)
    return ' '.join([text.text for text in root.findall('.//text')])

def process_video(video_id, transcript_file):
    try:
        video_transcript = get_transcript(video_id)
        transcript_file.write(f"video_id: {video_id}\n")
        transcript_file.write(f"transcript: {video_transcript}\n\n")
    except Exception as ex:
        stqdm.write(str(ex))
    
def get_conversation_chain(vectorstore):
    '''
    to create a conversational retrieval chain using a vector store.
    Initializes a ChatOpenAI model for generating responses.
    Uses ConversationBufferMemory to maintain the chat history.
    Constructs a ConversationalRetrievalChain that retrieves relevant information from the vector store.
    Takes a FAISS vector store as input and returns a conversational chain object.
    '''
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    '''
    to handle user input and manage the conversation flow.
    Checks if a conversation chain exists; if not, creates one using the vector store.
    Processes the user question and retrieves the response from the conversation chain.
    Displays the user's question and the bot's answer using custom templates.
    Updates the session state with the new chat history, ensuring no duplicate entries.
    Takes a user question as input.
    ''' 
    
    if st.session_state.conversation is None and st.session_state.vectorstore is not None:
        st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore)
    
    response = st.session_state.conversation({"question": user_question})

    return response

def main():

    if "transcript_content" not in st.session_state:
        st.session_state.transcript_content = ""

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.set_page_config(
        page_title='YT playlist or video url',
        page_icon='💻',
        layout='centered'
        )
    
    with st.sidebar:
        choice = st.radio(options=['YT_video','YT_playlist'],label='Choose one')
        # print(choice)
        enter_url = st.text_input('Enter the URL:  ')
        button = st.button('Process')
        if button:
            with st.spinner('Loading Transcripts......'):
                if enter_url:
                    if choice == 'YT_video':
                        video_id = enter_url.split('watch?v=')[1]
                        transcript_filename = f"{video_id}.txt"
                        with open(transcript_filename, "w") as transcript_file:
                            process_video(video_id, transcript_file)                     
                    elif choice == 'YT_playlist':
                        playlist_id = enter_url.split('list=')[1]                        
                        video_ids = get_video_ids(playlist_id)                                            
                        transcript_filename = f"{playlist_id}.txt"
                        with open(transcript_filename, "w") as transcript_file:
                            for video_id in stqdm(video_ids):
                                process_video(video_id, transcript_file)
                    if transcript_filename:
                        with open(transcript_filename, "r") as transcript_file:
                            st.session_state.transcript_content = transcript_file.read()
                           
                else:
                    st.error('Enter Valid URL')

    container = st.container() 
    with container:
        transcript_text = st.session_state.transcript_content.replace('[INAUDIBLE]', ' ').replace('\n',' ')
        if transcript_text:
            st.session_state.vectorstore = get_vectorstore(get_text_chunks(transcript_text))
            st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore)
        st.title("YouTube Transcript Query")
        st.subheader('Ask Question Related to the Video or Playlist')
        user_question = st.text_input('Chatting Area')
        if user_question:
            response_text = handle_userinput(user_question=user_question)
            print(response_text['chat_history'])
            for i, response in enumerate(response_text['chat_history']):
                if i == 0: # question
                    container.write("User Question")
                    container.write(response.content)
                else:
                    container.write("AI Answer")
                    container.write(response.content)
        else:
            container.warning('Provide a question')

        # else:
        #     container.write("Please provide a valid transcript and user question.")


if __name__ == '__main__':
    main()