from langchain.chains import ConversationalRetrievalChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import LLMChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders.merge import MergedDataLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv
import sys
import openai
import os
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader
from langchain.vectorstores import Chroma

sys.path.append('../..')

_ = load_dotenv(find_dotenv())  # read local .env file

openai.api_key = os.environ['OPENAI_API_KEY']
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
os.environ["LANGCHAIN_API_KEY"] = os.environ['LANG_SMITH_API_KEY']

persist_directory = 'embeddings/chroma/'
ALLOW_PERSIST_EMBEDDING = True


class Chat:
    def __init__(self, from_disk=False, load_pdf=True, load_youtube=True, load_url=True) -> None:
        self.vectordb = self.load_resources(
            from_disk, load_pdf, load_youtube, load_url)
        self.chat_history = []
        self.llm = ChatOpenAI(temperature=0)

    def load_resources(self, from_disk=False, load_pdf=True, load_youtube=True, load_url=True):

        print("Start load_resources...")
        print(locals())

        embedding = OpenAIEmbeddings()

        ##### Youtube video ######
        video_urls = ['https://www.youtube.com/watch?v=kuZNIvdwnMc',
                      'https://www.youtube.com/watch?v=hZE5fT7CVdo']
        save_dir = "data/youtube/"
        youtube_loader = GenericLoader(
            YoutubeAudioLoader(video_urls, save_dir),
            OpenAIWhisperParser()
        )

        ##### PDF ######
        pdf_loaders = [
            # Duplicate documents on purpose - messy data
            PyPDFLoader("data/pdf/2023Catalog_1-20.pdf"),
            PyPDFLoader("data/pdf/2023Catalog_21-40.pdf"),
            PyPDFLoader("data/pdf/2023Catalog_41-60.pdf"),
            PyPDFLoader("data/pdf/2023Catalog_61-80.pdf"),
            PyPDFLoader("data/pdf/2023Catalog_81-100.pdf"),
            PyPDFLoader("data/pdf/2023Catalog_101-120.pdf"),
            PyPDFLoader("data/pdf/2023Catalog_121-140.pdf"),
            PyPDFLoader("data/pdf/2023Catalog_141-159.pdf"),
            PyPDFLoader("data/pdf/2023Catalog_160-178.pdf"),
            PyPDFLoader("data/pdf/2023Catalog_179-end.pdf")
        ]

        ##### URLs #####
        web_loader = WebBaseLoader(["https://www.sfbu.edu/admissions/student-health-insurance",
                                    "https://www.sfbu.edu/admissions/entrance-health",
                                    "https://www.sfbu.edu/iep/fees"])

        all_loaders = []
        if load_pdf:
            all_loaders.extend(pdf_loaders)
        if load_youtube:
            all_loaders.append(youtube_loader)
        if load_url:
            all_loaders.append(web_loader)

        loader_all = MergedDataLoader(
            loaders=all_loaders)

        embedding = OpenAIEmbeddings()
        persist_directory = 'embeddings/chroma/'

        if from_disk == False:
            docs_all = loader_all.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=150
            )

            splits = text_splitter.split_documents(docs_all)

            vectordb = Chroma.from_documents(
                documents=splits,
                embedding=embedding,
                persist_directory=persist_directory
            )

            if ALLOW_PERSIST_EMBEDDING:
                vectordb.persist()
        else:
            vectordb = Chroma(persist_directory='embeddings/chroma/',
                              embedding_function=embedding)

        print("Finish load_resources...")

        return vectordb

    def get_prompt_template(self):
        template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.
        {context}
        Question: {question}
        """

        return template

    def clear_chat_history(self):
        self.chat_history = []

    def retrieval_answer(self, question):

        vectordb = self.vectordb
        chat_history = self.chat_history
        llm = self.llm

        # Build prompt

        memory = ConversationSummaryBufferMemory(
            llm=llm, input_key='question', output_key='answer')

        # vectordbkwargs = {"search_distance": 0.9}

        question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
        doc_chain = load_qa_chain(llm, chain_type="map_reduce")

        chain = ConversationalRetrievalChain(
            retriever=vectordb.as_retriever(),
            question_generator=question_generator,
            combine_docs_chain=doc_chain,
            return_source_documents=True,
            return_generated_question=True,
            response_if_no_docs_found="I don't know",
            memory=memory,
            verbose=True
        )

        result = chain({"question": question, "chat_history": chat_history})
        answer = result["answer"]
        chat_history.append((question, answer))

        return answer


def main():
    chat = Chat()
    question = "What is SFBU Health Insurance Coverage Period"
    a1 = chat.retrieval_answer(question)
    print(a1)


if __name__ == "__main__":
    main()
