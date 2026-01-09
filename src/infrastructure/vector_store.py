"""
Módulo: src/infrastructure/vector_store.py
Descrição: Gerencia ingestão de XML (MedQuAD), JSON (PubMedQA) e PDF.
"""

import json
import logging
import os
import xml.etree.ElementTree as ET
from typing import List

# Desabilitar symlinks warning no Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings

logger = logging.getLogger(__name__)

class VectorStoreRepository:
    """Gerencia operações com vector store (Chroma)."""
    
    def __init__(self):
        self.docs_path = settings.docs_full_path
        self.db_path = settings.vector_db_full_path
        self.embeddings = self._get_embeddings()
        self.vector_store = self._initialize_vectorstore()

    def _get_embeddings(self):
        """Inicializa embeddings com SentenceTransformer (sem limites de quota)."""
        logger.info("🔄 Inicializando embeddings...")
        try:
            # Usar SentenceTransformer diretamente (mais robusto no Windows)
            from sentence_transformers import SentenceTransformer
            from langchain_core.embeddings import Embeddings
            
            model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
            
            # Wrapper simples para LangChain
            class CustomEmbeddings(Embeddings):
                def embed_documents(self, texts):
                    return model.encode(texts, convert_to_numpy=True).tolist()
                def embed_query(self, text):
                    return model.encode(text, convert_to_numpy=True).tolist()
            
            embeddings = CustomEmbeddings()
            logger.info("✅ Embeddings inicializados (SentenceTransformer - sem quota limits)")
            return embeddings
        except Exception as e:
            logger.error(f"❌ Erro ao inicializar embeddings: {e}")
            logger.error(f"   Para corrigir, execute: python scripts/download_embeddings.py")
            raise

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
                if not answer.strip(): # type: ignore
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

    def _chunk_documents(self, documents):
        """Divide documentos em chunks."""
        logger.info("✂️ Dividindo documentos em chunks...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        logger.info(f"✅ {len(chunks)} chunks criados")
        return chunks
    
    def _initialize_vectorstore(self):
        """Inicializa ou carrega vector store existente."""
        logger.info(f"💾 Inicializando Chroma em {self.db_path}...")
        try:
            if (self.db_path / "chroma.sqlite3").exists():
                logger.info("📦 Carregando banco vetorial existente...")
                vector_store = Chroma(
                    embedding_function=self.embeddings,
                    persist_directory=str(self.db_path),
                    collection_name="medical_protocols"
                )
                logger.info("✅ Vectorstore carregado")

                # Se existir DB mas estiver vazio, popula automaticamente.
                try:
                    existing_count = vector_store._collection.count()  # type: ignore[attr-defined]
                except Exception:
                    existing_count = None

                if existing_count == 0:
                    logger.warning(
                        "⚠️ Vectorstore existente está vazio (0 chunks). Reindexando base de conhecimento..."
                    )
                    documents = self._load_documents()
                    if not documents:
                        logger.warning(
                            "⚠️ Nenhum documento encontrado na base de conhecimento; vectorstore permanecerá vazio."
                        )
                    else:
                        chunks = self._chunk_documents(documents)
                        vector_store.add_documents(chunks)
                        # Algumas versões expõem persist(), outras persistem automaticamente.
                        persist_fn = getattr(vector_store, "persist", None)
                        if callable(persist_fn):
                            persist_fn()

                        try:
                            new_count = vector_store._collection.count()  # type: ignore[attr-defined]
                        except Exception:
                            new_count = None
                        logger.info(f"✅ Vectorstore populado (chunks={new_count})")
            else:
                logger.info("🆕 Criando novo banco vetorial...")
                documents = self._load_documents()
                chunks = self._chunk_documents(documents)
                
                vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=self.embeddings,
                    persist_directory=str(self.db_path),
                    collection_name="medical_protocols"
                )
                logger.info("✅ Vectorstore criado")
            
            return vector_store
        except Exception as e:
            logger.error(f"❌ Erro ao inicializar vectorstore: {e}")
            raise

    def get_retriever(self):
        """Retorna retriever configurado."""
        return self.vector_store.as_retriever(
            search_kwargs={"k": 4}
        )
    
    def reset_vectorstore(self):
        """Reseta vector store completamente."""
        logger.warning("🔄 Resetando vectorstore...")
        import shutil
        if self.db_path.exists():
            shutil.rmtree(self.db_path)
        self.vector_store = self._initialize_vectorstore()
        logger.info("✅ Vectorstore resetado")