"""Busca BM25 implementada do zero.

BM25 dá mais peso a palavras raras na coleção (idf) e evita que trechos
longos ganhem só por repetirem termos (normalização por tamanho).
"""

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .trechos import PASTA_PADRAO, Trecho, carregar_pasta, carregar_sinonimos

STOPWORDS = set(
    """a o as os um uma uns umas de do da dos das no na nos nas em por para com sem
    e ou que se ao aos à às é são ser foi tem têm ter há qual quais quanto quanta
    quantos quantas como onde quando meu minha meus minhas seu sua seus suas voces
    vocês voce você eu me mim isso esse essa este esta pelo pela pelos pelas mais
    mas muito já tambem também pode posso consigo""".split()
)


def _sem_acento(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", texto) if not unicodedata.combining(c))


# Abreviações de chat para palavras sem conteúdo ("vcs abrem que horas hj?"). Sem isso, cada
# uma contava como palavra desconhecida e derrubava a relevância de uma pergunta boa.
ABREVIACOES = set("vc vcs q oq pq pra pro pras pros qnto qto qdo tb tbm hj mto mt ta tah eh".split())
# "funciona" tem o mesmo radical de "funcionamento": "como funciona o reembolso?" ia para o
# horário de funcionamento. Como verbo genérico, ele não diz nada sobre o assunto da pergunta.
VERBOS_GENERICOS = {"funciona", "funcionam", "funcionar", "funcionando"}
STOPWORDS = {_sem_acento(p) for p in STOPWORDS} | ABREVIACOES | VERBOS_GENERICOS | {"hoje", "agora", "esta", "estao"}


# Escolhido pela avaliação: 4 e 5 recusam pior fora do escopo; sem corte, o acerto@1 cai de 95% para 80%.
TAMANHO_RADICAL = 6


def radical(token: str, tamanho: int = TAMANHO_RADICAL) -> str:
    """Radical por truncamento: cancelar/cancelamento -> cancel, celiaco/celiaca -> celiac.
    Simples, mas funciona bem em português para coleções pequenas."""
    if len(token) > 4 and token.endswith("s"):
        token = token[:-1]  # pizzas -> pizza
    return token[:tamanho]


def palavras(texto: str) -> list[str]:
    """Minúsculas, sem acento, número separado de letra: "Pão de Queijo 5km!" -> ["pao", "de", "queijo", "5", "km"].

    Separar o número importa: "5km" virava uma palavra só, que não existe nos documentos
    ("De 3 a 7 km"), e a relevância da pergunta caía para menos da metade."""
    return re.findall(r"[a-z]+|[0-9]+", _sem_acento(texto.lower()))


def tokenizar(texto: str, tamanho: int = TAMANHO_RADICAL) -> list[str]:
    """Minúsculas, sem acento, sem stopwords, reduzido ao radical."""
    return [radical(p, tamanho) for p in palavras(texto) if p not in STOPWORDS]


@dataclass(frozen=True)
class Resultado:
    trecho: Trecho
    score: float


class IndiceBM25:
    def __init__(
        self, trechos: list[Trecho], k1: float = 1.5, b: float = 0.75, sinonimos: dict[str, str] | None = None
    ):
        if not trechos:
            raise ValueError("o índice precisa de pelo menos um trecho")
        self.trechos = trechos
        # A chave é comparada pelo radical: "frete" também vale para "fretes", "coca" para "cocas".
        self.sinonimos = {radical(" ".join(palavras(k))): " ".join(palavras(v)) for k, v in (sinonimos or {}).items()}
        self.k1, self.b = k1, b
        # O título da seção entra no índice: ele resume bem o assunto do trecho.
        self.docs = [Counter(tokenizar(f"{t.secao} {t.secao} {t.texto}")) for t in trechos]
        self.tamanhos = [sum(d.values()) for d in self.docs]
        self.media = sum(self.tamanhos) / len(self.tamanhos)
        n = len(self.docs)
        df = Counter(termo for d in self.docs for termo in d)
        self.idf = {termo: math.log(1 + (n - f + 0.5) / (f + 0.5)) for termo, f in df.items()}

    def consulta(self, pergunta: str) -> list[str]:
        """Termos da pergunta, com os sinônimos aplicados.

        Palavra que não existe nos documentos ("frete") é TROCADA pela dos documentos ("taxa de
        entrega"): somada, ela contaria como desconhecida e derrubaria a relevância da pergunta.
        Palavra que existe ("delivery", em "não vendemos bebidas alcoólicas pelo delivery")
        fica, e o sinônimo é somado: trocar perderia esse casamento."""
        termos: list[str] = []
        for palavra in palavras(pergunta):
            troca = self.sinonimos.get(radical(palavra))
            if troca is None:
                termos.append(palavra)
            elif tokenizar(palavra) and tokenizar(palavra)[0] in self.idf:
                termos += [palavra, troca]
            else:
                termos.append(troca)
        return tokenizar(" ".join(termos))

    def _score(self, consulta: list[str], i: int) -> float:
        doc, tamanho = self.docs[i], self.tamanhos[i]
        total = 0.0
        for termo in consulta:
            freq = doc.get(termo, 0)
            if not freq:
                continue
            norm = self.k1 * (1 - self.b + self.b * tamanho / self.media)
            total += self.idf[termo] * freq * (self.k1 + 1) / (freq + norm)
        return total

    def relevancia(self, pergunta: str, resultado: Resultado) -> float:
        """Pontuação do resultado dividida pelo máximo possível para esta pergunta (0 a 1).

        A pontuação BM25 crua depende do tamanho da coleção: com poucos documentos, toda
        palavra parece comum e as pontuações caem. Um limiar fixo (a primeira versão usava 1,5)
        funcionava com os documentos do restaurante, mas recusava perguntas boas quando o
        projeto era usado com outros documentos. Dividir pelo máximo deixa o limiar estável.
        Palavras da pergunta que não aparecem em nenhum documento contam como as mais raras:
        são justamente o sinal de que a pergunta é sobre outro assunto."""
        consulta = self.consulta(pergunta)
        if not consulta:
            return 0.0
        idf_desconhecida = math.log(1 + (len(self.docs) + 0.5) / 0.5)
        maximo = sum(self.idf.get(termo, idf_desconhecida) * (self.k1 + 1) for termo in consulta)
        return resultado.score / maximo

    def buscar(self, pergunta: str, k: int = 3) -> list[Resultado]:
        consulta = self.consulta(pergunta)
        resultados = [Resultado(t, self._score(consulta, i)) for i, t in enumerate(self.trechos)]
        resultados = [r for r in resultados if r.score > 0]
        resultados.sort(key=lambda r: r.score, reverse=True)
        return resultados[:k]


def carregar_indice(pasta: Path = PASTA_PADRAO) -> IndiceBM25:
    """Índice dos documentos .md da pasta, com o sinonimos.json dela, se houver."""
    return IndiceBM25(carregar_pasta(pasta), sinonimos=carregar_sinonimos(pasta))
