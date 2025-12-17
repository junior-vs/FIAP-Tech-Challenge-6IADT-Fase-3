"""
Módulo: src/infrastructure/vector_store.py
Descrição: Gerencia a ingestão de documentos médicos (XML) e a busca vetorial.
Motivo da alteração: Ajuste solicitado para leitura exclusiva de arquivos XML (protocolos estruturados).
"""

import os
import shutil
from typing import List

# Importações para carregamento
from langchain_community.document_loaders import DirectoryLoader, UnstructuredXMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma 

from src.config import settings
from src.infrastructure.llm_factory import LLMFactory

class VectorStoreRepository:
    def __init__(self):
        self.embeddings = LLMFactory.get_embeddings()
        self.vectorstore = self._initialize_db()

    def _load_documents(self):
        """
        Carrega documentos XML da pasta configurada em settings.docs_path.
        """
        if not os.path.exists(settings.docs_path):
            os.makedirs(settings.docs_path, exist_ok=True)
            print(f"⚠️ Pasta {settings.docs_path} criada. Adicione seus XMLs lá!")
            return []

        print(f"📂 Carregando protocolos XML da pasta: {settings.docs_path}...")
        
        # Carrega apenas arquivos .xml
        # O UnstructuredXMLLoader remove as tags (<tag>) e mantém o texto do conteúdo
        xml_loader = DirectoryLoader(
            settings.docs_path, 
            glob="**/*.xml", 
            loader_cls=UnstructuredXMLLoader
        )
        
        try:
            docs = xml_loader.load()
            print(f"📄 Total de documentos XML carregados: {len(docs)}")
            return docs
        except Exception as e:
            print(f"❌ Erro ao ler arquivos XML: {e}")
            print("Dica: Verifique se instalou 'pip install unstructured'")
            return []

    def _initialize_db(self):
        """
        Inicializa o ChromaDB. Carrega se existir, cria se não.
        """
        if os.path.exists(settings.vector_db_path) and os.listdir(settings.vector_db_path):
            print("💾 Carregando banco vetorial existente...")
            return Chroma(
                persist_directory=settings.vector_db_path, 
                embedding_function=self.embeddings
            )
        
        print("⚙️ Banco não encontrado. Criando novo índice de conhecimento médico (XML)...")
        
        raw_docs = self._load_documents()
        if not raw_docs:
            print("⚠️ Nenhum documento XML válido encontrado para indexar.")
            # Retorna banco vazio para não travar a aplicação
            return Chroma(
                embedding_function=self.embeddings,
                persist_directory=settings.vector_db_path
            )

        # Dividir em chunks (XMLs podem ser longos)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size, 
            chunk_overlap=settings.chunk_overlap
        )
        chunks = splitter.split_documents(raw_docs)
        print(f"🧩 Documentos divididos em {len(chunks)} fragmentos.")
        
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=settings.vector_db_path
        )
        print("✅ Indexação de XMLs concluída!")
        return vectorstore

    def get_retriever(self, k: int = 4):
        return self.vectorstore.as_retriever(search_kwargs={"k": k})

    def reset_database(self):
        if os.path.exists(settings.vector_db_path):
            shutil.rmtree(settings.vector_db_path)
        self.vectorstore = self._initialize_db()