"""Cliente do Google Gemini, que tem plano grátis. Só com a biblioteca padrão.

O assistente fala no formato da Claude API. Este cliente usa a compatibilidade do Gemini com o
formato da OpenAI (/chat/completions) e traduz nos dois sentidos: o pedido sai no formato do
Gemini, e a resposta volta no formato da Claude. Por isso o assistente e os testes dele não mudam.
O assistente não usa ferramentas, mas a tradução cuida delas do mesmo jeito que nos outros
projetos (agente-pedidos e analisador-avaliacoes-ia), para o código ser um só.

Detalhes do Gemini que a tradução cuida:
- Assinatura do raciocínio ("thought signature"): nos modelos Gemini 3 e 4, cada chamada de
  ferramenta pode vir com uma assinatura, que precisa voltar intacta na requisição seguinte,
  senão a API responde 400. O formato da Claude não tem onde guardá-la, então o cliente guarda
  pelo id da chamada e a devolve quando aquela chamada aparece de novo no histórico.
- A resposta de uma ferramenta leva o nome da ferramenta, além do id.
- Uma ferramenta sem argumentos vai sem "parameters" (o Gemini recusa objeto sem propriedades).
- Chave inválida volta como 400, não 401; o limite diário do plano grátis volta como 429. Os dois
  não melhoram tentando de novo, então o cliente para com uma mensagem clara.
- No limite por minuto, o Gemini diz quanto esperar ("retryDelay") no corpo do erro.
- Números inteiros podem chegar como 2.0 (o Gemini guarda todo número como decimal); viram 2,
  senão a validação ("quantidade precisa ser inteiro") recusaria um argumento certo.
- Motivos de parada próprios do Gemini (SAFETY, RECITATION...) viram recusa.
- Dois pontos que a documentação não deixa claros têm um plano B, usado só se o Gemini responder
  400 apontando para eles: um modelo sem raciocínio configurável (o pedido é refeito sem
  "reasoning_effort") e uma chamada sem assinatura (refeito com a assinatura que a documentação
  indica para chamadas que não vieram da API).
"""

import itertools
import json
import re

from .llm import ESFORCO, MAX_TOKENS, ClienteClaude, ErroLLM

URL_BASE_GEMINI = "https://generativelanguage.googleapis.com/v1beta/openai"
# Modelo do plano grátis (lista em ai.google.dev/gemini-api/docs/pricing). Se o limite grátis
# dele acabar, dá para trocar: --modelo gemini-3.5-flash-lite
MODELO_GEMINI = "gemini-3.8-flash"
# Modelo reserva, mais leve e com limite grátis próprio. No primeiro teste real, o principal ficou
# sobrecarregado (503) e o reserva respondeu na hora. Só entra se o usuário não escolheu o --modelo.
MODELO_RESERVA = "gemini-3.5-flash-lite"

DICAS_GEMINI = {
    400: "confira a GEMINI_API_KEY (crie uma em aistudio.google.com/apikey) e o --modelo",
    401: "verifique a GEMINI_API_KEY",
    403: "a chave não tem permissão; confira a GEMINI_API_KEY",
    404: "verifique o nome do modelo (--modelo); a lista está em ai.google.dev/gemini-api/docs/models",
    413: "a pergunta ou os trechos ficaram grandes demais",
    429: "o limite grátis acabou por hoje; tente amanhã ou troque o --modelo (ex.: gemini-3.5-flash-lite)",
}
PARADAS = {"length": "max_tokens", "content_filter": "refusal"}
# Motivos de parada do Gemini por segurança ou direitos autorais: a resposta foi bloqueada.
BLOQUEIOS = ("safety", "recitation", "prohibited", "blocklist", "spii", "language")
# Valor da documentação para pular a conferência da assinatura (chamadas que não vieram da API).
PLANOS_B = 3  # quantos ajustes, no máximo, um pedido recusado com 400 pode ter
ASSINATURA_FICTICIA = {"google": {"thought_signature": "skip_thought_signature_validator"}}


def _texto_do_conteudo(conteudo) -> str:
    """O conteúdo de uma mensagem ou de um resultado de ferramenta: texto ou lista de blocos."""
    if isinstance(conteudo, str):
        return conteudo
    if isinstance(conteudo, list):
        return "".join(b.get("text", "") for b in conteudo if isinstance(b, dict) and b.get("type") == "text")
    return ""


def ferramenta_para_gemini(ferramenta: dict) -> dict:
    funcao = {"name": ferramenta["name"], "description": ferramenta.get("description", "")}
    esquema = ferramenta.get("input_schema") or {}
    if esquema.get("properties"):
        funcao["parameters"] = esquema
    return {"type": "function", "function": funcao}


def _assistente_para_gemini(blocos: list[dict], assinaturas: dict[str, dict], ficticia: bool) -> dict:
    convertida: dict = {"role": "assistant"}
    texto = _texto_do_conteudo(blocos)
    if texto:
        convertida["content"] = texto
    chamadas = []
    for bloco in blocos:
        if bloco.get("type") != "tool_use":
            continue  # blocos de raciocínio da Claude não existem no Gemini
        chamada = {
            "id": bloco["id"],
            "type": "function",
            "function": {
                "name": bloco["name"],
                "arguments": json.dumps(bloco.get("input") or {}, ensure_ascii=False),
            },
        }
        if bloco["id"] in assinaturas:
            chamada["extra_content"] = assinaturas[bloco["id"]]
        chamadas.append(chamada)
    if chamadas:
        if ficticia and "extra_content" not in chamadas[0]:
            chamadas[0]["extra_content"] = ASSINATURA_FICTICIA
        convertida["tool_calls"] = chamadas
    elif not texto:
        convertida["content"] = ""  # nunca uma mensagem sem conteúdo e sem chamada
    return convertida


def mensagens_para_gemini(
    sistema: str, mensagens: list[dict], assinaturas: dict[str, dict], ficticia: bool = False
) -> list[dict]:
    """Histórico no formato da Claude -> lista de mensagens no formato do Gemini (OpenAI).
    Com `ficticia`, a primeira chamada de cada passo que não tem assinatura ganha a fictícia."""
    saida: list[dict] = [{"role": "system", "content": sistema}]
    nomes: dict[str, str] = {}  # id da chamada -> nome da ferramenta, para a resposta da ferramenta
    for mensagem in mensagens:
        conteudo = mensagem["content"]
        if mensagem["role"] == "assistant":
            blocos = conteudo if isinstance(conteudo, list) else [{"type": "text", "text": conteudo}]
            nomes.update({b["id"]: b["name"] for b in blocos if b.get("type") == "tool_use"})
            saida.append(_assistente_para_gemini(blocos, assinaturas, ficticia))
        elif isinstance(conteudo, str):
            saida.append({"role": "user", "content": conteudo})
        else:
            # Respostas das ferramentas viram mensagens "tool", que precisam vir logo depois das
            # chamadas; um texto junto vai numa mensagem "user" em seguida.
            for bloco in conteudo:
                if bloco.get("type") == "tool_result":
                    saida.append(
                        {
                            "role": "tool",
                            "tool_call_id": bloco["tool_use_id"],
                            "name": nomes.get(bloco["tool_use_id"], ""),
                            "content": _texto_do_conteudo(bloco.get("content", "")),
                        }
                    )
            texto = _texto_do_conteudo([b for b in conteudo if b.get("type") == "text"])
            if texto:
                saida.append({"role": "user", "content": texto})
    return saida


def _texto_da_resposta(conteudo) -> str:
    """O texto da resposta: o normal é uma string, mas alguns servidores no formato OpenAI mandam
    uma lista de partes ([{"type": "text", "text": ...}])."""
    if isinstance(conteudo, str):
        return conteudo
    partes = [p.get("text") for p in _lista(conteudo) if isinstance(p, dict) and p.get("type", "text") == "text"]
    return "".join(t for t in partes if isinstance(t, str))


def _dict(valor) -> dict:
    return valor if isinstance(valor, dict) else {}


def _lista(valor) -> list:
    return valor if isinstance(valor, list) else []


def _inteiros(valor):
    """2.0 -> 2, em qualquer profundidade. Os outros valores ficam como estão."""
    if isinstance(valor, float) and valor.is_integer():
        return int(valor)
    if isinstance(valor, dict):
        return {chave: _inteiros(v) for chave, v in valor.items()}
    if isinstance(valor, list):
        return [_inteiros(v) for v in valor]
    return valor


def _recusar(constante: str):
    """NaN e Infinity não são JSON válido; o json do Python aceitaria, e o pedido seguinte, com
    esses argumentos de volta no histórico, levaria erro 400."""
    raise ValueError(constante)


def _argumentos(bruto) -> dict:
    """Argumentos da chamada: o Gemini manda texto JSON. JSON inválido vira {} e a ferramenta
    responde com erro de argumento, que o modelo corrige no passo seguinte."""
    if isinstance(bruto, dict):  # já veio como objeto: só serve se der para mandar de volta como JSON
        try:
            json.dumps(bruto, allow_nan=False)
        except (ValueError, TypeError):
            return {}
        return _inteiros(bruto)
    try:
        valor = json.loads(bruto, parse_constant=_recusar) if isinstance(bruto, str) and bruto.strip() else {}
    except ValueError:
        return {}
    return _inteiros(valor) if isinstance(valor, dict) else {}


class ClienteGemini(ClienteClaude):
    NOME = "Google Gemini"
    VARIAVEIS_CHAVE = ("GEMINI_API_KEY", "GOOGLE_API_KEY")  # a segunda é a que alguns tutoriais usam
    VARIAVEL_URL = "GEMINI_BASE_URL"
    URL_BASE = URL_BASE_GEMINI
    CAMINHO = "/chat/completions"
    MODELO = MODELO_GEMINI
    DICAS = DICAS_GEMINI
    # O plano grátis às vezes demora: no primeiro teste real, respostas passaram de 60 s.
    TIMEOUT = 120.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        escolhido = kwargs.get("modelo") is not None
        # Sem --modelo, o reserva assume quando o principal falha em todas as tentativas.
        self.reserva: str | None = None if escolhido else MODELO_RESERVA
        self.assinaturas: dict[str, dict] = {}  # id da chamada -> assinatura do raciocínio
        self._ids = itertools.count(1)
        self._ids_vistos: set[str] = set()
        self.usar_esforco = True  # vira False se o modelo recusar "reasoning_effort"
        self.assinatura_ficticia = False  # vira True se o Gemini reclamar de assinatura faltando

    def cabecalhos(self) -> dict[str, str]:
        return {"authorization": f"Bearer {self.chave}", "content-type": "application/json"}

    def pedido(self, sistema: str, mensagens: list[dict], ferramentas: list[dict] | None = None) -> dict:
        pedido: dict = {
            "model": self.modelo,
            "max_tokens": MAX_TOKENS,
            "messages": mensagens_para_gemini(sistema, mensagens, self.assinaturas, self.assinatura_ficticia),
        }
        if self.usar_esforco:
            pedido["reasoning_effort"] = ESFORCO
        if ferramentas:
            pedido["tools"] = [ferramenta_para_gemini(f) for f in ferramentas]
        return pedido

    def traduzir(self, resposta) -> dict:
        """Resposta do Gemini (formato OpenAI) -> formato da Claude API. À prova de resposta
        malformada: o que não tiver o formato esperado é ignorado."""
        escolhas = _lista(_dict(resposta).get("choices"))
        escolha = _dict(escolhas[0]) if escolhas else {}
        mensagem = _dict(escolha.get("message"))
        blocos: list[dict] = []
        texto = _texto_da_resposta(mensagem.get("content"))
        if texto:
            blocos.append({"type": "text", "text": texto})
        for chamada in map(_dict, _lista(mensagem.get("tool_calls"))):
            funcao = _dict(chamada.get("function"))
            if not isinstance(funcao.get("name"), str):
                continue
            id_ = chamada.get("id")
            # Sem id, a resposta da ferramenta não teria a quem se ligar. Com um id repetido (de
            # outro passo ou da mesma resposta), a assinatura e o nome de uma chamada iriam para outra.
            if not isinstance(id_, str) or not id_ or id_ in self._ids_vistos:
                id_ = f"{id_ if isinstance(id_, str) and id_ else 'gemini'}-{next(self._ids)}"
            self._ids_vistos.add(id_)
            if isinstance(chamada.get("extra_content"), dict):
                self.assinaturas[id_] = chamada["extra_content"]
            entrada = _argumentos(funcao.get("arguments"))
            blocos.append({"type": "tool_use", "id": id_, "name": funcao["name"], "input": entrada})

        motivo = escolha.get("finish_reason")
        motivo = motivo.lower() if isinstance(motivo, str) else None  # um valor estranho (lista) não quebra
        parada: str | None
        if motivo in PARADAS:
            parada = PARADAS[motivo]
        elif (isinstance(motivo, str) and any(b in motivo for b in BLOQUEIOS)) or mensagem.get("refusal"):
            parada = "refusal"
        elif any(b["type"] == "tool_use" for b in blocos):
            parada = "tool_use"  # o Gemini às vezes manda "stop" junto com chamadas de ferramenta
        else:
            parada = "end_turn" if escolhas else None
        uso = _dict(_dict(resposta).get("usage"))
        return {
            "content": blocos,
            "stop_reason": parada,
            "usage": {"input_tokens": uso.get("prompt_tokens"), "output_tokens": uso.get("completion_tokens")},
        }

    def fatal(self, codigo: int, corpo: str) -> bool:
        return super().fatal(codigo, corpo) or _chave_invalida(codigo, corpo) or _limite_diario(codigo, corpo)

    def desistir(self, codigo: int, corpo: str) -> bool:
        return _limite_diario(codigo, corpo)

    def pausa(self, codigo: int, cabecalhos, corpo: str, padrao: float) -> float:
        # "retryDelay": "36s" nos detalhes do erro; se faltar, "Please retry in 36.2s." na mensagem.
        pedida = re.search(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"', corpo) or re.search(
            r"retry in (\d+(?:\.\d+)?)\s*s", corpo, re.IGNORECASE
        )
        pausa = min(float(pedida.group(1)), 60.0) if pedida else super().pausa(codigo, cabecalhos, corpo, padrao)
        if pausa >= 5:  # sem o aviso, a espera parece o programa travado
            if codigo == 429:
                self.avisar(f"(Limite por minuto do plano grátis do Gemini: esperando {pausa:.0f} s...)")
            else:
                self.avisar(f"(O Gemini respondeu {codigo}: tentando de novo em {pausa:.0f} s...)")
        return pausa

    def _plano_b_extra(self, corpo: str) -> bool:
        """Outro ajuste para um 400 (o analisador usa para a ferramenta forçada). True = refazer."""
        return False

    def dica_ao_desistir(self, codigo: int | None) -> str:
        if codigo == 429:
            return ". Dica: é o limite por minuto do plano grátis; espere um minuto e tente de novo"
        if codigo in (500, 502, 503, 504):
            if self.modelo == MODELO_RESERVA:  # já é o reserva: sugerir ele mesmo não ajudaria
                return ". Dica: o Gemini está sobrecarregado; tente de novo daqui a alguns minutos"
            return f". Dica: o Gemini está sobrecarregado; tente daqui a pouco ou use --modelo {MODELO_RESERVA}"
        return ""

    def enviar(self, sistema: str, mensagens: list[dict], *args, **kwargs) -> dict:
        """Se o modelo principal falhar em todas as tentativas (sobrecarga, sem resposta, limite
        por minuto ou do dia), troca para o reserva, avisa e tenta mais uma vez."""
        try:
            return self._enviar_com_planos_b(sistema, mensagens, *args, **kwargs)
        except ErroLLM as erro:
            limite_do_dia = erro.codigo == 429 and _limite_diario(429, erro.corpo)
            if not self.reserva or self.modelo == self.reserva or not (erro.esgotou or limite_do_dia):
                raise
            self.avisar(f"({self.modelo} não respondeu; trocando para o {self.reserva}, mais leve...)")
            self.modelo = self.reserva
            return self._enviar_com_planos_b(sistema, mensagens, *args, **kwargs)

    def _enviar_com_planos_b(self, sistema: str, mensagens: list[dict], *args, **kwargs) -> dict:
        """Com os planos B: um 400 que aponta o raciocínio ou a assinatura refaz o pedido sem o
        "reasoning_effort" ou com a assinatura fictícia. Cada plano B é usado uma vez só."""
        for _ in range(PLANOS_B):  # cada plano B uma vez; um laço sem fim aqui travaria o programa
            try:
                return super().enviar(sistema, mensagens, *args, **kwargs)
            except ErroLLM as erro:
                if erro.codigo != 400 or _chave_invalida(400, erro.corpo):
                    raise
                if self.usar_esforco and re.search(r"reasoning|thinking", erro.corpo, re.IGNORECASE):
                    self.usar_esforco = False
                elif not self.assinatura_ficticia and "signature" in erro.corpo.lower():
                    self.assinatura_ficticia = True
                elif not self._plano_b_extra(erro.corpo):
                    raise
        return super().enviar(sistema, mensagens, *args, **kwargs)


def _chave_invalida(codigo: int, corpo: str) -> bool:
    return codigo == 400 and ("API_KEY_INVALID" in corpo or "API key not valid" in corpo)


def _limite_diario(codigo: int, corpo: str) -> bool:
    return codigo == 429 and "PerDay" in corpo
