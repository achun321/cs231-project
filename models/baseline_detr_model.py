import json
import os
import re
import string
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from langchain.llms import OpenAI
from langchain.vectorstores.faiss import FAISS
from langchain.chains import ChatVectorDBChain
from langchain.prompts.prompt import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document

# Define your prompt templates
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
    """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.
    You can assume the discussion is about the video content.
    Chat History:
    {chat_history}
    Follow-Up Input: {question}
    Standalone question:"""
)

QA_PROMPT = PromptTemplate(
    template="""You are an AI assistant designed for answering questions about a video.
    You are given a document and a question, the document records what people see and hear from this video.
    Try to connect this information and provide a yes/no answer to the question with the exact outputs of "yes" or "no" with no additional text or punctuation.
    Question: {question}
    =========
    {context}
    =========
    """,
    input_variables=["question", "context"]
)

class LlmReasoner:
    def __init__(self, args):
        self.history = []
        self.gpt_version = args['gpt_version']
        self.data_dir = args['data_dir']
        self.tmp_dir = args['tmp_dir']
        self.qa_chain = None
        self.vectorstore = None
        self.top_k = 3
        self.llm = OpenAI(temperature=0, model_name=self.gpt_version, openai_api_key=args['openai_api_key'])
        self.retrieved_docs_history = []  # Store retrieved docs for each question
        self.init_vectorstore()

    def init_vectorstore(self):
        pkl_path = os.path.join(self.tmp_dir, "detr_output_video.pkl")
        self.vectorstore = FAISS.load_local(pkl_path, OpenAIEmbeddings(openai_api_key='sk-proj-aj8krgV5TWH6Gmm3X9m2T3BlbkFJt1YsHQBdEjxOJ8DvzNTo'))
        self.qa_chain = ChatVectorDBChain.from_llm(
            self.llm,
            self.vectorstore,
            qa_prompt=QA_PROMPT,
            condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        )
        self.qa_chain.top_k_docs_for_context = self.top_k

    def __call__(self, question):
        retrieved_docs = self.vectorstore.similarity_search(question, k=self.top_k)
        self.retrieved_docs_history.append((question, retrieved_docs))
        
        retrieved_contexts = [doc.page_content for doc in retrieved_docs]
        response = self.qa_chain({"question": question, "chat_history": self.history, "context": " ".join(retrieved_contexts)})["answer"]
        self.history.append((question, response))
        
        return response

    def clean_history(self):
        self.history = []
        self.retrieved_docs_history = []

def parse_timestamps(retrieved_contexts):
    timestamps = []
    for context in retrieved_contexts:
        match = re.search(r'When (\d+:\d+:\d+) - (\d+:\d+:\d+)', context)
        if match:
            timestamps.append(f"{match.group(1)}-{match.group(2)}")
    return timestamps

def format_key_to_timestamp(key):
    # Assuming each key represents seconds
    hours, remainder = divmod(int(key), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}-{hours:02}:{minutes:02}:{seconds + 1:02}"

def calculate_accuracy(correct_answers, predicted_answers):
    correct_count = sum(1 for true, pred in zip(correct_answers, predicted_answers) if true == pred)
    return correct_count / len(correct_answers)

def calculate_precision_recall_f1(correct_answers, predicted_answers, positive_label='yes'):
    true_positives = sum(1 for true, pred in zip(correct_answers, predicted_answers) if true == pred == positive_label)
    total_predicted_positives = sum(1 for pred in predicted_answers if pred == positive_label)
    total_actual_positives = sum(1 for true in correct_answers if true == positive_label)
    
    precision = true_positives / total_predicted_positives if total_predicted_positives else 0
    recall = true_positives / total_actual_positives if total_actual_positives else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
    
    return precision, recall, f1

def evaluate_model(llm_reasoner, combined_mapping):
    correct_answers = []
    predicted_answers = []
    results = {}
    i = 0

    for key, entry in combined_mapping.items():
        print("Key: ", key, "out of ", len(combined_mapping))
        question = entry["question"]
        correct_answer = entry["correct_answer"]
        print(question, correct_answer)
        correct_timestamp = format_key_to_timestamp(key)
        try:
            predicted_answer = llm_reasoner(question).strip().lower().rstrip(string.punctuation + " ")
            print(predicted_answer)
            predicted_answers.append(predicted_answer)
            correct_answers.append(correct_answer)
            
            retrieved_contexts = [doc.page_content for doc in llm_reasoner.retrieved_docs_history[-1][1]]
            retrieved_timestamps = parse_timestamps(retrieved_contexts)

            results[key] = {
                "question": question,
                "predicted_answer": predicted_answer,
                "correct_answer": correct_answer,
                "correct_timestamp": correct_timestamp,
                "retrieved_timestamps": retrieved_timestamps
            }
        except Exception as e:
            print(f"Error processing question '{question}': {e}")
            raise Exception
        
    accuracy = calculate_accuracy(correct_answers, predicted_answers)
    precision, recall, f1 = calculate_precision_recall_f1(correct_answers, predicted_answers)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    return results

if __name__ == "__main__":
    args = {
        'gpt_version': 'gpt-3.5-turbo',
        'data_dir': '/home/adamchun/cs231n-project/data/VQAv2/',  # Directory where logs are stored
        'tmp_dir': '/home/adamchun/cs231n-project/tmp/',
        'openai_api_key': 'sk-proj-haPf5tVo5iWLoDMwa2qTT3BlbkFJnzC05nR0VusmR7N8EQlK'
    }

    llm_reasoner = LlmReasoner(args)
    results_output_path = '/home/adamchun/cs231n-project/data/VQAv2/detr_results.json'

    with open('/home/adamchun/cs231n-project/data/VQAv2/combined_mapping.json', 'r') as f:
        combined_mapping = json.load(f)
    # Evaluate the model
    results = evaluate_model(llm_reasoner, combined_mapping)
    with open(results_output_path, 'w') as f:
        json.dump(results, f, indent=4)