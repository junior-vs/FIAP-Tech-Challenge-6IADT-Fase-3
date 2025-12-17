"""
Módulo: src/infrastructure/vector_store.py
Descrição: Gerencia ingestão de XML (MedQuAD), JSON (PubMedQA) e PDF.
"""

import os
import shutil
import json
import xml.etree.ElementTree as ET
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import settings
from src.infrastructure.llm_factory import LLMFactory

class VectorStoreRepository:
    def __init__(self):
        self.embeddings = LLMFactory.get_embeddings()
        self.vectorstore = self._initialize_db()

    def _load_medquad_xml(self, file_path: str) -> List[Document]:
        """
        Parser para XML do MedQuAD. Lida com casos de respostas vazias (MedlinePlus).
        """
        docs = []
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Tenta pegar metadados globais do arquivo
            focus_elem = root.find('Focus')
            focus = focus_elem.text if focus_elem is not None else "General Health"
            
            # Pega a URL original se disponível (útil quando não há resposta no XML)
            url = root.get('url', 'URL não informada')
            
            qa_pairs = root.findall('.//QAPair')
            
            for pair in qa_pairs:
                question_elem = pair.find('Question')
                answer_elem = pair.find('Answer')
                
                question = question_elem.text if question_elem is not None else ""
                answer = answer_elem.text if answer_elem is not None else ""
                
                # AJUSTE: Se não tiver resposta, criamos um aviso indicando a fonte
                if not answer.strip():
                    answer = f"Conteúdo protegido por copyright. Consulte a fonte oficial: {url}"
                
                # Monta o conteúdo para o RAG
                page_content = (
                    f"Topic: {focus}\n"
                    f"Question: {question}\n"
                    f"Answer: {answer}\n"
                    f"Source URL: {url}"
                )
                
                metadata = {
                    "source": os.path.basename(file_path),
                    "type": "medquad_xml",
                    "focus": focus,
                    "original_id": pair.get("pid", "unknown")
                }
                
                docs.append(Document(page_content=page_content, metadata=metadata))
                
        except Exception as e:
            print(f"⚠️ Erro no XML {os.path.basename(file_path)}: {e}")
            
        return docs

    def _load_pubmed_json(self, file_path: str) -> List[Document]:
        """
        Carregador para PubMedQA (JSON).
        """
        docs = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"   PubMedQA: processando {len(data)} registros de {os.path.basename(file_path)}...")
            
            for pubmed_id, content in data.items():
                question = content.get("QUESTION", "")
                contexts = content.get("CONTEXTS", [])
                long_answer = content.get("LONG_ANSWER", "")
                
                context_text = "\n".join(contexts) if isinstance(contexts, list) else str(contexts)
                
                page_content = (
                    f"Context: {context_text}\n"
                    f"Question: {question}\n"
                    f"Expert Answer: {long_answer}"
                )
                
                metadata = {
                    "source": os.path.basename(file_path),
                    "pubmed_id": pubmed_id,
                    "type": "pubmed_qa"
                }
                docs.append(Document(page_content=page_content, metadata=metadata))
        except Exception as e:
            print(f"❌ Erro JSON {file_path}: {e}")
        return docs

    def _load_documents(self):
        """
        Varre recursivamente a pasta configurada (os.walk) para encontrar arquivos.
        """
        if not os.path.exists(settings.docs_path):
            # Tenta criar, mas avisa se estiver vazio
            os.makedirs(settings.docs_path, exist_ok=True)
            print(f"⚠️ Pasta {settings.docs_path} criada e vazia.")
            return []

        print(f"📂 Varrendo base de conhecimento em: {settings.docs_path}")
        all_docs = []

        # os.walk garante que entramos em subpastas (7_SeniorHealth_QA, ori_pqal, etc)
        for root, dirs, files in os.walk(settings.docs_path):
            for file in files:
                file_path = os.path.join(root, file)
                
                # 1. XMLs
                if file.endswith(".xml"):
                    all_docs.extend(self._load_medquad_xml(file_path))
                
                # 2. JSONs (apenas os de dados, ignorando configs)
                elif file.endswith(".json"):
                    if "pqal" in file or "ground_truth" in file:
                         all_docs.extend(self._load_pubmed_json(file_path))
                
                # 3. PDFs
                elif file.endswith(".pdf"):
                    try:
                        loader = PyPDFLoader(file_path)
                        all_docs.extend(loader.load())
                    except Exception:
                        pass # Ignora erros de PDF corrompido

        print(f"📄 Total de documentos carregados: {len(all_docs)}")
        return all_docs

    def _initialize_db(self):
        # Se banco existe, carrega
        if os.path.exists(settings.vector_db_path) and os.listdir(settings.vector_db_path):
            print("💾 Carregando banco vetorial existente...")
            return Chroma(
                persist_directory=settings.vector_db_path, 
                embedding_function=self.embeddings
            )
        
        # Se não, cria
        print("⚙️ Criando novo índice...")
        raw_docs = self._load_documents()
        
        if not raw_docs:
            print("⚠️ AVISO: Nenhum documento encontrado. O assistente não terá conhecimento.")
            return Chroma(embedding_function=self.embeddings, persist_directory=settings.vector_db_path)

        splitter = RecursiveCharacterTextSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
        chunks = splitter.split_documents(raw_docs)
        
        vectorstore = Chroma.from_documents(documents=chunks, embedding=self.embeddings, persist_directory=settings.vector_db_path)
        print("✅ Banco Vetorial Pronto!")
        return vectorstore

    def get_retriever(self, k: int = 4):
        return self.vectorstore.as_retriever(search_kwargs={"k": k})
    
    def reset_database(self):
        if os.path.exists(settings.vector_db_path):
            shutil.rmtree(settings.vector_db_path)
        self.vectorstore = self._initialize_db()