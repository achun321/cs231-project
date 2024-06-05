import os 
import pdb
import pickle
import re
import faiss
import numpy as np
from langchain.llms import OpenAI
from langchain.vectorstores.faiss import FAISS
from langchain.chains import ChatVectorDBChain
from langchain.prompts.prompt import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings 
from langchain.schema import Document
from langchain_community.docstore.in_memory import InMemoryDocstore


_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
You can assume the discussion is about the video content.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

qa_template = """You are an AI assistant designed for answering questions about a video.
You are given a document and a question, the document records what people see and hear from this video.
Try to connect these information and provide a yes/no answer to the question only responding "yes" or "no".
Question: {question}
=========
{context}
=========
"""
QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["question", "context"])
    

class LlmReasoner():
    def __init__(self, args):
        self.history = []
        self.gpt_version = args.gpt_version
        self.data_dir = args.data_dir
        self.tmp_dir = args.tmp_dir
        self.qa_chain = None
        self.text_vectorstore = None
        self.clip_vectorstore = None
        self.top_k = 3
        self.llm = OpenAI(temperature=0,  model_name=self.gpt_version)
         
    def exist_vectorstore(self, video_id):
        text_pkl_path = os.path.join(self.tmp_dir, f"{video_id}.pkl")
        feature_index_path = os.path.join(self.tmp_dir, f"{video_id}_clip_features.faiss")

        text_exists = os.path.exists(text_pkl_path)
        feature_exists = os.path.exists(feature_index_path)

        if text_exists and feature_exists:
            self.text_vectorstore = FAISS.load_local(text_pkl_path, OpenAIEmbeddings())
            self.clip_vectorstore = faiss.read_index(feature_index_path)

            # Initialize the QA chain with both vector stores if both are available
            self.qa_chain = ChatVectorDBChain.from_llm(
                self.llm,
                [self.text_vectorstore, self.clip_vectorstore],  # Use both vector stores
                qa_prompt=QA_PROMPT,
                condense_question_prompt=CONDENSE_QUESTION_PROMPT,
            )
            self.qa_chain.top_k_docs_for_context = self.top_k
            return True
        else:
            print("Missing one or more vectorstore components:")
            if not text_exists:
                print(f"Text vectorstore not found at {text_pkl_path}")
            if not feature_exists:
                print(f"CLIP vectorstore not found at {feature_index_path}")
            return False

    
    def create_vectorstore(self, video_id):     
        pkl_path = os.path.join(self.tmp_dir, f"{video_id}.pkl")
        feature_path = os.path.join(self.tmp_dir, f"{video_id}_clip_features.npy")
        feature_index_path = os.path.join(self.tmp_dir, f"{video_id}_clip_features.faiss")
        
        if not os.path.exists(pkl_path):
            log_path = os.path.join(self.data_dir, f"{video_id}.log")
            with open(log_path, 'r') as f:
                raw_text = f.read()

            # Split text into smaller sections based on "When"
            sections = re.split(r'(?=When\s\d{1,2}:\d{2}:\d{2})', raw_text)
            documents = [
                Document(page_content=section.strip(), metadata={"source": log_path}, lookup_index=i)
                for i, section in enumerate(sections) if section.strip()
            ]

            # Split text
            # text_splitter = RecursiveCharacterTextSplitter()
            # documents = text_splitter.split_documents(raw_documents)


            # Load Data to vectorstore
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(documents, embeddings)
        
            # Save vectorstore
            vectorstore.save_local(pkl_path)
            # with open(pkl_path, "wb") as f:
            #     pickle.dump(vectorstore, f)
        
        self.text_vectorstore = FAISS.load_local(pkl_path, OpenAIEmbeddings())
        if os.path.exists(feature_path):
            clip_features = np.load(feature_path)
            if not os.path.exists(feature_index_path):
                dim = clip_features.shape[1]
                clip_index = faiss.IndexFlatL2(dim)
                clip_index.add(clip_features.astype('float32'))
                faiss.write_index(clip_index, feature_index_path)
            else:
                clip_index = faiss.read_index(feature_index_path)

            # If you still need to link features to metadata:
            metadata = [{"video_id": video_id, "frame_index": i} for i in range(len(clip_features))]
            self.clip_vectorstore = FAISS(clip_index, metadata=metadata)
        else:
            print(f"No CLIP feature file found for video_id {video_id}. Please check your paths and preprocessing.")


        # with open(pkl_path, 'rb') as file:
        #     self.vectorstore = pickle.load(file)
        
        if self.text_vectorstore and self.clip_vectorstore:
            self.qa_chain = ChatVectorDBChain.from_llm(
                self.llm,
                [self.text_vectorstore, self.clip_vectorstore],  # Use both vector stores
                qa_prompt=QA_PROMPT,
                condense_question_prompt=CONDENSE_QUESTION_PROMPT,
            )
            self.qa_chain.top_k_docs_for_context = self.top_k
        else:
            print("One or both vector stores are not initialized correctly.")
        return 

    def __call__(self, question):
        print(f"Question: {question}")
        # Retrieve the documents
        if self.clip_vectorstore:
            retrieved_clip_docs = self.clip_vectorstore.similarity_search(question, k=self.top_k)
        retrieved_docs = self.text_vectorstore.similarity_search(question, k=self.top_k)
        
        # Print the retrieved sections of the log
        for i, doc in enumerate(retrieved_docs):
            print(f"Document {i+1}:")
            print(doc.page_content)
            if self.clip_vectorstore:
                print("Retrieved Clip Documents (represented by indices or IDs):")
            for i, doc in enumerate(retrieved_clip_docs):
                print(f"Clip Document {i+1}: {doc}\n")
            print("\n")
        response = self.qa_chain({"question": question, "chat_history": self.history})["answer"]
        self.history.append((question, response))
        
        print(f"Assistant: {response}")
        print("\n")
        return response
    
    def clean_history(self):
        self.history = []
