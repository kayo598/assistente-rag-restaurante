import pytest

from rag_restaurante.busca import carregar_indice


@pytest.fixture(autouse=True)
def sem_chaves_do_computador(monkeypatch):
    """Os testes não podem depender das chaves do computador de quem roda: com a GEMINI_API_KEY
    definida, um teste de "sem chave" escolheria o Gemini e poderia chamar a API de verdade."""
    for nome in ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_BASE_URL"):
        monkeypatch.delenv(nome, raising=False)


@pytest.fixture(scope="session")
def indice():
    return carregar_indice()


class ClienteFixo:
    """Faz o papel do LLM: devolve sempre o mesmo texto e guarda o que recebeu."""

    def __init__(self, texto):
        self.texto = texto
        self.recebido = []

    def completar(self, sistema, mensagem):
        self.recebido.append((sistema, mensagem))
        if isinstance(self.texto, BaseException):
            raise self.texto
        return self.texto
