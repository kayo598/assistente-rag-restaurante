"""Cliente da Claude API (Messages API), só com a biblioteca padrão.

O assistente fala sempre no formato da Claude API. Para usar o Google Gemini (que tem plano
grátis), veja gemini.py: ele traduz o pedido e a resposta, e o assistente não muda.

- Pede uma resposta em texto (sem ferramentas).
- Erros temporários (limite de uso, sobrecarga, rede, tempo esgotado) têm nova tentativa
  com espera crescente. Erros que não mudam tentando de novo (chave inválida) são "fatais".
"""

import http.client
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable

URL_BASE_PADRAO = "https://api.anthropic.com"
MODELO_PADRAO = "claude-haiku-5-5"
# Nos modelos 5.x o raciocínio (thinking) vem ligado por padrão e conta neste limite. Com 400
# (o valor antigo), a resposta podia parar depois do raciocínio e antes de qualquer texto.
MAX_TOKENS = 4096
# Pouco raciocínio basta para responder a partir de trechos, e sai mais rápido e mais barato.
# Desligar de vez não serve para todos: cada modelo 5.x aceita um valor diferente, e o Opus 5.5
# não aceita nenhum.
ESFORCO = "low"
# Parada antes do fim: a resposta está incompleta.
PARADAS_DE_CORTE = {"max_tokens", "model_context_window_exceeded"}
SEM_CHAVE = (
    "Defina a variável ANTHROPIC_API_KEY (Claude API, paga) ou GEMINI_API_KEY (Google Gemini, tem plano"
    " grátis), ou use --offline."
)
STATUS_TEMPORARIOS = {408, 409, 429, 500, 502, 503, 504, 529}
STATUS_FATAIS = {401, 403, 404}  # chave inválida, sem permissão, modelo inexistente


class ErroLLM(RuntimeError):
    def __init__(
        self, mensagem: str, fatal: bool = False, codigo: int | None = None, corpo: str = "", esgotou: bool = False
    ):
        super().__init__(mensagem)
        self.fatal = fatal  # True = não adianta tentar de novo (ex.: chave inválida)
        self.codigo = codigo  # status HTTP, quando a API respondeu com erro
        self.corpo = corpo  # corpo do erro, para decidir se vale tentar de outro jeito
        self.esgotou = esgotou  # True = falhou em todas as tentativas (sobrecarga, limite, sem resposta)


DICAS = {
    401: "verifique a ANTHROPIC_API_KEY (para usar o Gemini, que é grátis: --provedor gemini)",
    403: "a chave não tem permissão para este recurso",
    404: "verifique o nome do modelo (--modelo) e a ANTHROPIC_BASE_URL",
    413: "a pergunta ou os trechos ficaram grandes demais",
}


def aceita_esforco(modelo: str) -> bool:
    """O parâmetro de esforço existe nos modelos 5.x e depois ("claude-haiku-5-5")."""
    versao = re.match(r"claude-[a-z]+-(\d+)", modelo)
    return versao is not None and int(versao.group(1)) >= 5


def mensagem_de_erro(codigo: int, corpo: str, dicas: dict[int, str] = DICAS) -> str:
    """'API respondeu 401 (authentication_error: invalid x-api-key). Dica: verifique a ANTHROPIC_API_KEY'
    em vez do JSON cru da API. Entende também o formato do Gemini, que usa "status" no lugar de
    "type" e às vezes manda o erro dentro de uma lista."""
    try:
        dados = json.loads(corpo)
        if isinstance(dados, list) and dados:
            dados = dados[0]
        erro = dados["error"]
        tipo = erro.get("type") or erro.get("status") or "erro"
        texto = " ".join(str(erro.get("message", "")).split())
        if len(texto) > 300:  # o Gemini manda parágrafos com links no erro 429
            texto = texto[:297] + "..."
        detalhe = f"{tipo}: {texto}".strip(": ")
    except (ValueError, KeyError, TypeError, AttributeError):
        detalhe = " ".join(corpo.split())[:200] or "sem detalhes"
    dica = dicas.get(codigo)
    return f"API respondeu {codigo} ({detalhe})" + (f". Dica: {dica}" if dica else "")


def sem_chave() -> str:
    """A mensagem de chave ausente, com o comando certo para o sistema de quem roda."""
    if sys.platform == "win32":
        return SEM_CHAVE + ' No PowerShell: $env:GEMINI_API_KEY = "sua-chave"'
    return SEM_CHAVE + ' No terminal: export GEMINI_API_KEY="sua-chave"'


ASPAS = "\"'\u201c\u201d\u2018\u2019"  # retas e curvas (o Word e o WhatsApp trocam as retas pelas curvas)


def limpar_chave(chave: str, variavel: str) -> str:
    """Espaços, quebras de linha e aspas em volta vêm de copiar e colar, e são tirados. Um caractere
    estranho no meio (acento, aspas curvas) quebraria o cabeçalho HTTP com um erro sem explicação,
    então vira uma mensagem clara. A chave nunca aparece na mensagem."""
    limpa = chave.strip().strip(ASPAS).strip()
    if not limpa or any(not 33 <= ord(c) <= 126 for c in limpa):
        raise ErroLLM(
            f"A chave em {variavel} tem espaço, acento ou aspas no meio. Copie a chave de novo, sem nada em volta.",
            fatal=True,
        )
    return limpa


def _retry_after(cabecalhos, padrao: float) -> float:
    """Usa o tempo pedido pela API, se for um número válido entre 0 e 60 segundos."""
    try:
        valor = float(cabecalhos.get("retry-after"))
    except (TypeError, ValueError, AttributeError):
        return padrao
    if not math.isfinite(valor) or valor < 0:
        return padrao
    return min(valor, 60.0)


class ClienteClaude:
    NOME = "Claude API"
    VARIAVEIS_CHAVE: tuple[str, ...] = ("ANTHROPIC_API_KEY",)
    VARIAVEL_URL = "ANTHROPIC_BASE_URL"
    URL_BASE = URL_BASE_PADRAO
    CAMINHO = "/v1/messages"
    MODELO = MODELO_PADRAO
    DICAS = DICAS
    TIMEOUT = 60.0  # segundos esperando cada resposta

    def __init__(
        self,
        chave: str | None = None,
        *,
        modelo: str | None = None,
        timeout: float | None = None,
        max_retentativas: int = 3,
        espera: Callable[[float], None] = time.sleep,
        url_base: str | None = None,
        avisar: Callable[[str], None] | None = None,
    ):
        variavel = next((v for v in self.VARIAVEIS_CHAVE if os.environ.get(v, "").strip()), self.VARIAVEIS_CHAVE[0])
        chave = chave or os.environ.get(variavel, "")
        if not chave.strip():
            raise ErroLLM(sem_chave(), fatal=True)
        self.chave: str = limpar_chave(chave, variavel)
        self.modelo = modelo or self.MODELO
        self.timeout = timeout or self.TIMEOUT
        self.max_retentativas = max_retentativas
        self.espera = espera  # trocável nos testes, para não esperar de verdade
        # Avisos durante esperas longas (sem eles, o programa parece travado). Vão para o stderr.
        self.avisar = avisar or (lambda texto: print(texto, file=sys.stderr, flush=True))
        # Mesma variável do SDK oficial: permite usar um proxy corporativo ou um servidor de teste.
        base = url_base or os.environ.get(self.VARIAVEL_URL) or self.URL_BASE
        self.url = base.rstrip("/") + self.CAMINHO
        self.uso = {"input_tokens": 0, "output_tokens": 0}  # tokens gastos, somados ao longo da sessão
        # Motivo de parada da última resposta: "max_tokens" = cortada; "refusal" = o modelo recusou.
        self.ultima_parada: str | None = None

    def completar(self, sistema: str, mensagem: str) -> str:
        """Manda uma mensagem e devolve o texto da resposta."""
        self.ultima_parada = None
        resposta = self.enviar(sistema, [{"role": "user", "content": mensagem}])
        if isinstance(resposta, dict) and isinstance(resposta.get("stop_reason"), str):
            self.ultima_parada = resposta["stop_reason"]
        blocos = resposta.get("content") if isinstance(resposta, dict) else None
        if not isinstance(blocos, list):
            blocos = []
        # Só os blocos de texto: os de raciocínio ("thinking") não são resposta.
        texto = "".join(
            b["text"]
            for b in blocos
            if isinstance(b, dict) and b.get("type", "text") == "text" and isinstance(b.get("text"), str)
        )
        if not texto.strip() and self.ultima_parada in PARADAS_DE_CORTE:
            # Sem esta checagem, a resposta vazia virava "Não encontrei essa informação".
            raise ErroLLM(f"a resposta acabou antes de ter texto ({self.ultima_parada}); tente de novo")
        return texto

    def pedido(self, sistema: str, mensagens: list[dict]) -> dict:
        """O corpo da requisição, no formato da API deste cliente."""
        # Sem "temperature": os modelos 5.x recusam (Sonnet 5.5 responde 400) ou pedem para omitir.
        pedido: dict = {"model": self.modelo, "max_tokens": MAX_TOKENS, "system": sistema, "messages": mensagens}
        if aceita_esforco(self.modelo):
            pedido["output_config"] = {"effort": ESFORCO}
        return pedido

    def traduzir(self, resposta) -> dict:
        """A resposta no formato da Claude API (o Gemini precisa de tradução; a Claude, não)."""
        return resposta

    def enviar(self, sistema: str, mensagens: list[dict]) -> dict:
        corpo = json.dumps(self.pedido(sistema, mensagens)).encode("utf-8")

        motivo = ""
        ultimo: int | None = None  # último status HTTP, para a dica ao desistir
        for tentativa in range(self.max_retentativas + 1):
            pausa = float(2**tentativa)  # 1 s, 2 s, 4 s...
            try:
                resposta = self.traduzir(self._requisitar(corpo))
                self._somar_uso(resposta)
                return resposta
            except urllib.error.HTTPError as erro:
                try:
                    corpo_erro = erro.read().decode("utf-8", "replace")[:20000]
                except (OSError, http.client.HTTPException):  # a conexão pode cair lendo o corpo do erro
                    corpo_erro = ""
                if erro.code not in STATUS_TEMPORARIOS or self.desistir(erro.code, corpo_erro):
                    raise ErroLLM(
                        mensagem_de_erro(erro.code, corpo_erro, self.DICAS),
                        fatal=self.fatal(erro.code, corpo_erro),
                        codigo=erro.code,
                        corpo=corpo_erro,
                    ) from erro
                motivo, ultimo = f"API respondeu {erro.code}", erro.code
                if tentativa < self.max_retentativas:  # na última, não há espera (nem aviso dela)
                    pausa = self.pausa(erro.code, erro.headers, corpo_erro, pausa)
            # OSError cobre tempo esgotado, conexão recusada/caída, SSL e URLError;
            # HTTPException cobre resposta cortada no meio (IncompleteRead).
            except (OSError, http.client.HTTPException) as erro:
                ultimo = None
                causa = getattr(erro, "reason", None) or erro
                if isinstance(causa, TimeoutError):
                    motivo = f"sem resposta em {self.timeout:g} s"
                else:
                    motivo = f"falha de conexão ({getattr(erro, 'reason', None) or type(erro).__name__})"
                if tentativa < self.max_retentativas:
                    self.avisar(f"({self.NOME}: {motivo}; tentando de novo...)")
            if tentativa < self.max_retentativas:
                self.espera(pausa)
        desisti = f"{motivo}; desisti após {self.max_retentativas + 1} tentativas"
        raise ErroLLM(desisti + self.dica_ao_desistir(ultimo), codigo=ultimo, esgotou=True)

    def fatal(self, codigo: int, corpo: str) -> bool:
        """Erro que não muda tentando de novo, nem na próxima pergunta (ex.: chave inválida)."""
        return codigo in STATUS_FATAIS

    def desistir(self, codigo: int, corpo: str) -> bool:
        """Erro temporário em que esperar uns segundos não adianta (veja o Gemini)."""
        return False

    def pausa(self, codigo: int, cabecalhos, corpo: str, padrao: float) -> float:
        return _retry_after(cabecalhos, padrao)

    def dica_ao_desistir(self, codigo: int | None) -> str:
        return ""

    def cabecalhos(self) -> dict[str, str]:
        return {
            "x-api-key": self.chave,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    def _somar_uso(self, resposta) -> None:
        uso = resposta.get("usage") if isinstance(resposta, dict) else None
        if isinstance(uso, dict):
            for chave in self.uso:
                valor = uso.get(chave)
                if isinstance(valor, int) and not isinstance(valor, bool) and valor > 0:
                    self.uso[chave] += valor

    def _requisitar(self, corpo: bytes) -> dict:
        requisicao = urllib.request.Request(
            self.url,
            data=corpo,
            headers=self.cabecalhos(),
            method="POST",
        )
        with urllib.request.urlopen(requisicao, timeout=self.timeout) as resposta:
            bruto = resposta.read()
        try:
            return json.loads(bruto)
        except ValueError as erro:
            raise ErroLLM("A API devolveu uma resposta que não é JSON") from erro


PROVEDORES = ("claude", "gemini")


def escolher_provedor(provedor: str | None = None) -> str:
    """Sem --provedor: usa a Claude se ANTHROPIC_API_KEY existir; senão, o Gemini se GEMINI_API_KEY
    existir. Assim, quem só tem a chave grátis do Gemini não precisa escrever nada a mais."""
    if provedor:
        return provedor
    if os.environ.get("ANTHROPIC_API_KEY", "").strip():
        return "claude"
    if any(os.environ.get(v, "").strip() for v in ("GEMINI_API_KEY", "GOOGLE_API_KEY")):
        return "gemini"
    return "claude"


def criar_cliente(provedor: str | None = None, modelo: str | None = None) -> ClienteClaude:
    if escolher_provedor(provedor) == "gemini":
        from .gemini import ClienteGemini  # noqa: PLC0415 (aqui dentro: gemini.py importa este módulo)

        return ClienteGemini(modelo=modelo)
    return ClienteClaude(modelo=modelo)
