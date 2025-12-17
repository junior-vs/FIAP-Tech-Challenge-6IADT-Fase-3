"""
Módulo: src/infrastructure/vector_store.py
Descrição: Gerencia a ingestão de múltiplos formatos de documentos médicos.
Suporta:
1. XML (Padrão MedQuAD)
2. JSON (Padrão PubMedQA)
3. PDFs (Documentos Gerais)
"""

import os
import shutil
import json
import xml.etree.ElementTree as ET
from typing import List

# Importações do LangChain
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
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
        Parser customizado para arquivos XML do MedQuAD.
        Extrai pares de <Question> e <Answer> mantendo o contexto do <Focus>.
        """
        docs = []
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # O "Foco" do documento (ex: nome do medicamento ou doença)
            focus_elem = root.find('Focus')
            focus = focus_elem.text if focus_elem is not None else "General Health"
            
            # Itera sobre cada par de pergunta/resposta
            qa_pairs = root.findall('.//QAPair')
            
            for pair in qa_pairs:
                question_elem = pair.find('Question')
                answer_elem = pair.find('Answer')
                
                question = question_elem.text if question_elem is not None else ""
                answer = answer_elem.text if answer_elem is not None else ""
                
                # Tratamento para casos onde a resposta pode estar vazia (comuns em alguns datasets públicos)
                if not answer.strip():
                    continue  # Pula registros sem resposta útil
                
                # Monta o conteúdo que será indexado
                page_content = (
                    f"Topic: {focus}\n"
                    f"Question: {question}\n"
                    f"Answer: {answer}"
                )
                
                metadata = {
                    "source": os.path.basename(file_path),
                    "type": "medquad_xml",
                    "focus": focus,
                    "original_id": pair.get("pid", "unknown")
                }
                
                docs.append(Document(page_content=page_content, metadata=metadata))
                
        except Exception as e:
            print(f"⚠️ Erro ao processar XML MedQuAD {os.path.basename(file_path)}: {e}")
            
        return docs

    def _load_pubmed_json(self, file_path: str) -> List[Document]:
        """
        Carregador para o formato PubMedQA (ori_pqal.json).
        """
        docs = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"   🔍 Processando dataset PubMedQA ({len(data)} registros)...")
            
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
            print(f"❌ Erro ao ler JSON {file_path}: {e}")
            
        return docs

    def _load_documents(self):
        """
        Estratégia híbrida de carregamento:
        - Varre a pasta recursivamente.
        - Identifica a extensão.
        - Aplica o loader específico (XML customizado, JSON customizado ou PDF padrão).
        """
        if not os.path.exists(settings.docs_path):
            os.makedirs(settings.docs_path, exist_ok=True)
            print(f"⚠️ Pasta {settings.docs_path} criada.")
            return []

        print(f"📂 Iniciando ingestão na pasta: {settings.docs_path}")
        all_docs = []

        # Varredura manual para ter controle total sobre cada extensão
        for root, dirs, files in os.walk(settings.docs_path):
            for file in files:
                file_path = os.path.join(root, file)
                
                # 1. Processar XMLs (MedQuAD)
                if file.endswith(".xml"):
                    xml_docs = self._load_medquad_xml(file_path)
                    all_docs.extend(xml_docs)
                
                # 2. Processar JSONs (PubMedQA)
                elif file.endswith(".json"):
                    # Verifica se é o arquivo de perguntas (evita ler arquivos de config)
                    if "pqal" in file or "ground_truth" in file:
                         json_docs = self._load_pubmed_json(file_path)
                         all_docs.extend(json_docs)
                
                # 3. Processar PDFs (Protocolos)
                elif file.endswith(".pdf"):
                    try:
                        loader = PyPDFLoader(file_path)
                        all_docs.extend(loader.load())
                    except Exception as e:
                        print(f"Erro no PDF {file}: {e}")

        print(f"📄 Ingestão concluída. Total de documentos fragmentados: {len(all_docs)}")
        return all_docs

    def _initialize_db(self):
        """
        Inicializa ou carrega o ChromaDB.
        """
        # Se o banco existe e tem arquivos, carrega direto (mais rápido)
        if os.path.exists(settings.vector_db_path) and os.listdir(settings.vector_db_path):
            print("💾 Banco vetorial encontrado. Carregando da memória...")
            return Chroma(
                persist_directory=settings.vector_db_path, 
                embedding_function=self.embeddings
            )
        
        print("⚙️ Criando novo índice vetorial unificado (PDF + XML + JSON)...")
        
        raw_docs = self._load_documents()
        if not raw_docs:
            print("⚠️ Nenhum documento válido encontrado. Verifique a pasta docs/knowledge_base.")
            return Chroma(
                embedding_function=self.embeddings,
                persist_directory=settings.vector_db_path
            )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size, 
            chunk_overlap=settings.chunk_overlap
        )
        chunks = splitter.split_documents(raw_docs)
        print(f"🧩 Vetorizando {len(chunks)} fragmentos de informação...")
        
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=settings.vector_db_path
        )
        print("✅ Banco de Conhecimento Médico pronto!")
        return vectorstore

    def get_retriever(self, k: int = 4):
        return self.vectorstore.as_retriever(search_kwargs={"k": k})

    def reset_database(self):
        if os.path.exists(settings.vector_db_path):
            shutil.rmtree(settings.vector_db_path)
        self.vectorstore = self._initialize_db()