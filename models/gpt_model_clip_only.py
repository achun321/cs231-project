import os 
import pdb
import pickle
import re
import faiss
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from langchain.llms import OpenAI
from langchain.vectorstores.faiss import FAISS
from langchain.chains import ChatVectorDBChain
from langchain.embeddings import FakeEmbeddings as Embeddings
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
    
class CLIPEmbeddings:
    def __init__(self, model_name, device):
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)


    def embed(self, text: str) -> np.ndarray:
        # Prepare the text input for CLIP
        inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        # Generate embeddings
        with torch.no_grad():
            embeddings = self.model.get_text_features(**inputs)
        # Move embeddings to CPU and convert to numpy for compatibility with FAISS
        return embeddings.cpu().numpy().astype('float32')

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
        self.clip_embeddings = CLIPEmbeddings(args.feature_extractor, args.feature_extractor_device)
    
    def similarity_search(self, query_embedding, k=4):
        """Search for the k most similar vectors in the FAISS index.

        Args:
            query_embedding (numpy.ndarray): The query embedding as a numpy array.
            k (int): The number of nearest neighbors to retrieve.

        Returns:
            list: A list of tuples (Document, score), where Document is the retrieved document
                  and score is the distance/similarity score.
        """
        # Ensure query_embedding is a 2D array [1, dimension]
        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)

        distances, indices = self.clip_vectorstore.index.search(query_embedding.astype(np.float32), k)
        
        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            # Use the index directly as the document ID
            doc = self.clip_vectorstore.docstore.search(str(idx))
            results.append((doc, score))
        
        return results
         
    def exist_vectorstore(self, video_id):
        feature_index_path = os.path.join(self.tmp_dir, f"{video_id}_clip_features.faiss")
        feature_exists = os.path.exists(feature_index_path)
        print("TEST")

        if feature_exists and self.clip_vectorstore:
            print("TEST2")
            self.qa_chain = ChatVectorDBChain.from_llm(
                self.llm,
                self.clip_vectorstore,  
                qa_prompt=QA_PROMPT,
                condense_question_prompt=CONDENSE_QUESTION_PROMPT,
            )
            self.qa_chain.top_k_docs_for_context = self.top_k
            return True
        else:
            if not feature_exists:
                print(f"CLIP vectorstore not found at {feature_index_path}")
            return False

    
    def create_vectorstore(self, video_id):     
        feature_path = os.path.join(self.tmp_dir, f"{video_id}_clip_features.npy")
        feature_index_path = os.path.join(self.tmp_dir, f"{video_id}_clip_features.faiss")
        
        if os.path.exists(feature_path):
            clip_features = np.load(feature_path)
            if not os.path.exists(feature_index_path):
                print("CLIP FEATURES: ", clip_features)
                dim = clip_features.shape[1]
                clip_index = faiss.IndexFlatL2(dim)
                clip_index.add(clip_features.astype('float32'))
                print("CLIP INDEX: ", clip_index)
                faiss.write_index(clip_index, feature_index_path)
            else:
                clip_index = faiss.read_index(feature_index_path)

            docstore = InMemoryDocstore()
            documents_to_add = {str(i): {"id": str(i), "content": f"Embedding at index {i}"} for i in range(len(clip_features))}
            docstore.add(documents_to_add)

            self.clip_vectorstore = FAISS(embedding_function=self.clip_embeddings.embed,index=clip_index, docstore=docstore, index_to_docstore_id={i: str(i) for i in range(len(clip_features))})
            for doc_id, doc in self.clip_vectorstore.docstore._dict.items():
                print(f"Doc ID: {doc_id}, Content: {doc}")
        else:
            print(f"No CLIP feature file found for video_id {video_id}. Please check your paths and preprocessing.")


        # with open(pkl_path, 'rb') as file:
        #     self.vectorstore = pickle.load(file)
        
        if self.clip_vectorstore:
            self.qa_chain = ChatVectorDBChain.from_llm(
                self.llm,
                self.clip_vectorstore, 
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
            question_embedding = self.clip_embeddings.embed(question)
            retrieved_clip_docs = self.similarity_search(question_embedding, k=self.top_k)
        
        print("Retrieved Documents: ", retrieved_clip_docs)
        response = self.qa_chain({"question": question, "chat_history": self.history})["answer"]
        self.history.append((question, response))
        print(f"Assistant: {response}")
        print(f"Assistant: {response}")
        print("\n")
        return response
    
    def clean_history(self):
        self.history = []
