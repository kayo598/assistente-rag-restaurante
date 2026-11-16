"""Mede a qualidade da busca E da resposta, com perguntas de gabarito.

Busca (não depende do LLM):
- acerto@1 e acerto@3: a seção certa aparece em 1º ou entre os 3 primeiros resultados?
- MRR: média de 1/posição da seção certa (1,0 = sempre em 1º).

Resposta (depende do modo: offline ou Claude API):
- resposta correta: a resposta contém o fato esperado (ex.: "R$ 9,00" para a taxa de 5 km)?
- cita a fonte certa: a resposta cita a seção que contém o fato?
- recusa correta: perguntas fora do escopo foram recusadas, em vez de respondidas com invenção?

A primeira versão media só a busca. Por isso, um modo offline que achava o trecho certo
mas respondia com a frase errada ("taxa para 5 km" -> "Até 3 km: R$ 5,00") parecia ótimo.
"""

import json
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .assistente import LIMITE_PERGUNTA, Assistente, Resposta, sem_saudacao
from .busca import IndiceBM25
from .llm import ErroLLM

GABARITO_PADRAO = Path(__file__).parent / "perguntas.json"
# Perguntas como o cliente escreve no chat ("qto ta o frete pra 2km"), escritas antes das
# correções da 3ª rodada e nunca usadas para escolher sinônimos: medem se as correções generalizam.
GABARITO_COLOQUIAL = Path(__file__).parent / "perguntas_coloquiais.json"
# Com a API, uma falha temporária (que sobrou depois das novas tentativas) conta como erro
# daquela pergunta e a avaliação segue. Várias seguidas indicam API fora do ar: aí ela para.
LIMITE_FALHAS_SEGUIDAS = 3


class ErroGabarito(ValueError):
    pass


@dataclass
class Metricas:
    total: int
    acerto_1: float
    acerto_3: float
    mrr: float
    resposta_correta: float
    cita_fonte_certa: float
    recusa_correta: float | None  # None: o gabarito não tem perguntas fora do escopo
    erros: list[str] = field(default_factory=list)
    falhas_api: int = 0  # perguntas que ficaram sem resposta por erro da API


def _normalizar(texto: str) -> str:
    sem_acento = "".join(c for c in unicodedata.normalize("NFKD", texto) if not unicodedata.combining(c))
    return " ".join(sem_acento.casefold().split())


def carregar_gabarito(caminho: Path, indice: IndiceBM25) -> dict[str, Any]:
    """Valida o gabarito: uma seção digitada errada contaria como erro da busca, em silêncio."""
    caminho = Path(caminho)
    try:
        dados = json.loads(caminho.read_text(encoding="utf-8"))
    except OSError as erro:
        raise ErroGabarito(f"não consegui ler {caminho}: {erro.strerror or erro}") from erro
    except ValueError as erro:
        raise ErroGabarito(f"{caminho.name} não é um JSON válido ({erro})") from erro
    if (
        not isinstance(dados, dict)
        or not isinstance(dados.get("com_resposta"), list)
        or not dados["com_resposta"]
        or not isinstance(dados.get("sem_resposta"), list)
    ):
        raise ErroGabarito("o gabarito precisa das listas 'com_resposta' (não vazia) e 'sem_resposta'")

    secoes = {(t.fonte, t.secao) for t in indice.trechos}
    for n, caso in enumerate(dados["com_resposta"], start=1):
        if not isinstance(caso, dict):
            raise ErroGabarito(f"pergunta {n}: precisa ser um objeto com pergunta, fonte, secao e deve_conter")
        faltando = {"pergunta", "fonte", "secao", "deve_conter"} - set(caso)
        if faltando:
            raise ErroGabarito(f"pergunta {n}: faltam os campos {sorted(faltando)}")
        _validar_pergunta(caso["pergunta"], f"pergunta {n}")
        if (caso["fonte"], caso["secao"]) not in secoes:
            raise ErroGabarito(f"pergunta {n}: a seção '{caso['fonte']} › {caso['secao']}' não existe nos documentos")
        # Uma string em vez de lista seria conferida letra por letra: "R$ 9,00" passaria com "R$ 90,00".
        fatos = caso["deve_conter"]
        if not isinstance(fatos, list) or not fatos or not all(isinstance(f, str) and f.strip() for f in fatos):
            raise ErroGabarito(f"pergunta {n}: 'deve_conter' precisa ser uma lista de textos, ex.: [\"R$ 9,00\"]")
    for n, pergunta in enumerate(dados["sem_resposta"], start=1):
        _validar_pergunta(pergunta, f"sem_resposta {n}")
    return dados


def _validar_pergunta(pergunta: object, onde: str) -> None:
    if not isinstance(pergunta, str) or not pergunta.strip():
        raise ErroGabarito(f"{onde}: a pergunta precisa ser um texto não vazio")
    if len(pergunta.strip()) > LIMITE_PERGUNTA:
        raise ErroGabarito(f"{onde}: a pergunta passa de {LIMITE_PERGUNTA} caracteres")


class _Respondedor:
    """Chama o assistente, tolerando falhas temporárias isoladas da API."""

    def __init__(self, assistente: Assistente, erros: list[str]):
        self.assistente, self.erros = assistente, erros
        self.falhas = self.seguidas = 0

    def __call__(self, pergunta: str) -> Resposta | None:
        try:
            resposta = self.assistente.responder(pergunta)
        except ErroLLM as erro:
            if erro.fatal:
                raise
            self.falhas += 1
            self.seguidas += 1
            self.erros.append(f"api    {pergunta!r}: {erro}")
            if self.seguidas >= LIMITE_FALHAS_SEGUIDAS:
                motivo = f"{self.seguidas} perguntas seguidas falharam; a API parece fora do ar ({erro})"
                raise ErroLLM(motivo) from erro
            return None
        self.seguidas = 0
        return resposta


def avaliar(indice: IndiceBM25, caminho: Path, assistente: Assistente) -> Metricas:
    gabarito = carregar_gabarito(caminho, indice)
    casos = gabarito["com_resposta"]
    acertos_1 = acertos_3 = corretas = citou_certo = 0
    soma_rr = 0.0
    erros: list[str] = []
    responder = _Respondedor(assistente, erros)

    for caso in casos:
        # A mesma busca que o assistente faz (sem a saudação do começo), para medir o que o usuário vê.
        resultados = indice.buscar(sem_saudacao(caso["pergunta"]), k=3)
        posicoes = [
            i
            for i, r in enumerate(resultados, start=1)
            if r.trecho.fonte == caso["fonte"] and r.trecho.secao == caso["secao"]
        ]
        if posicoes:
            acertos_1 += posicoes[0] == 1
            acertos_3 += 1
            soma_rr += 1 / posicoes[0]
        if not posicoes or posicoes[0] != 1:
            achado = resultados[0].trecho.referencia if resultados else "nada"
            erros.append(f"busca  {caso['pergunta']!r}: esperado {caso['fonte']} › {caso['secao']}, veio {achado}")

        resposta = responder(caso["pergunta"])
        if resposta is None:  # falha da API: conta como resposta errada
            continue
        if all(_normalizar(fato) in _normalizar(resposta.texto) for fato in caso["deve_conter"]):
            corretas += 1
        else:
            erros.append(f"resposta {caso['pergunta']!r}: faltou {caso['deve_conter']} em {resposta.texto[:60]!r}")
        if any(f.trecho.fonte == caso["fonte"] and f.trecho.secao == caso["secao"] for f in resposta.fontes):
            citou_certo += 1

    fora = gabarito["sem_resposta"]
    recusas = 0
    for pergunta in fora:
        resposta = responder(pergunta)
        if resposta is None:
            continue
        if resposta.recusa:
            recusas += 1
        else:
            erros.append(f"recusa {pergunta!r}: deveria recusar")

    n = len(casos)
    return Metricas(
        total=n,
        acerto_1=acertos_1 / n,
        acerto_3=acertos_3 / n,
        mrr=soma_rr / n,
        resposta_correta=corretas / n,
        cita_fonte_certa=citou_certo / n,
        recusa_correta=recusas / len(fora) if fora else None,
        erros=erros,
        falhas_api=responder.falhas,
    )
