import json
import os
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
    Try to connect this information and provide a yes/no answer to the question only responding "yes" or "no".
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
        self.init_vectorstore()

    def init_vectorstore(self):
        pkl_path = os.path.join(self.tmp_dir, "output_video.pkl")
        self.vectorstore = FAISS.load_local(pkl_path, OpenAIEmbeddings(openai_api_key='sk-proj-haPf5tVo5iWLoDMwa2qTT3BlbkFJnzC05nR0VusmR7N8EQlK'))
        self.qa_chain = ChatVectorDBChain.from_llm(
            self.llm,
            self.vectorstore,
            qa_prompt=QA_PROMPT,
            condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        )
        self.qa_chain.top_k_docs_for_context = self.top_k

    def __call__(self, question):
        response = self.qa_chain({"question": question, "chat_history": self.history})["answer"]
        self.history.append((question, response))
        return response

    def clean_history(self):
        self.history = []

def evaluate_model(llm_reasoner, combined_mapping):
    correct_answers = []
    predicted_answers = []

    for entry in combined_mapping.values():
        question = entry["question"]
        correct_answer = entry["correct_answer"]
        print(question, correct_answer)
        try:
            predicted_answer = llm_reasoner(question).strip().lower()
            print(predicted_answer)
        except Exception as e:
            print(f"Error processing question '{question}': {e}")
            continue

        correct_answers.append(correct_answer)
        predicted_answers.append(predicted_answer)

    accuracy = accuracy_score(correct_answers, predicted_answers)
    precision = precision_score(correct_answers, predicted_answers, pos_label='yes')
    recall = recall_score(correct_answers, predicted_answers, pos_label='yes')
    f1 = f1_score(correct_answers, predicted_answers, pos_label='yes')

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

if __name__ == "__main__":
    args = {
        'gpt_version': 'gpt-3.5-turbo',
        'data_dir': '/home/adamchun/cs231n-project/data/VQAv2/',  # Directory where logs are stored
        'tmp_dir': '/home/adamchun/cs231n-project/tmp/',
        'openai_api_key': 'sk-proj-haPf5tVo5iWLoDMwa2qTT3BlbkFJnzC05nR0VusmR7N8EQlK'
    }

    llm_reasoner = LlmReasoner(args)
    
    with open('/home/adamchun/cs231n-project/data/VQAv2/combined_mapping.json', 'r') as f:
        combined_mapping = json.load(f)

    # Evaluate the model
    evaluate_model(llm_reasoner, combined_mapping)