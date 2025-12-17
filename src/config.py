from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    gemini_api_key: str
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.0
    chunk_size: int = 1000
    chunk_overlap: int = 200
    book_url: str = "https://www.gutenberg.org/files/55752/55752-0.txt"  # Dom Casmurro
    storage_path: str = "dom_casmurro.txt"
    vectorstore_path: str = Field(default="vectorstore", description="Path for Chroma vector store")
    chroma_collection_name: str = Field(default="dom_casmurro", description="Chroma collection name")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings() # pyright: ignore[reportCallIssue]