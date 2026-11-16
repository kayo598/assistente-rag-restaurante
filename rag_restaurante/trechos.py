"""Divide os documentos em trechos pequenos, um por seção.

Trechos curtos deixam a busca mais precisa e o prompt mais barato:
o LLM recebe só as seções que importam para a pergunta.
"""

import json
import re
from itertools import pairwise
from dataclasses import dataclass
from pathlib import Path

PASTA_PADRAO = Path(__file__).parent / "documentos"
ARQUIVO_SINONIMOS = "sinonimos.json"


class ErroDocumentos(ValueError):
    """Pasta de documentos inválida: erro de configuração, com mensagem clara."""


@dataclass(frozen=True)
class Trecho:
    fonte: str  # nome do arquivo, ex.: entrega.md
    secao: str  # título da seção (##)
    texto: str

    @property
    def referencia(self) -> str:
        return f"{self.fonte} › {self.secao}"


TITULO = re.compile(r"^(#{1,2})(?:[ \t]+(.*?))?(?:[ \t]+#+)?[ \t]*$")  # "## Pizzas", "## Pizzas ##"
ITEM_DE_LISTA = re.compile(r"^\s*([-*+]|\d+[.)])\s")


def _sem_metadados(linhas: list[str]) -> list[str]:
    """Tira o bloco de metadados do topo (front matter), comum em Markdown exportado do Notion,
    Obsidian ou Jekyll: "---\ntitle: ...\ntags: ...\n---". Antes, ele virava um trecho."""
    if linhas and linhas[0].strip() == "---":
        for i in range(1, len(linhas)):
            if linhas[i].strip() in ("---", "..."):
                return linhas[i + 1 :]
    return linhas


def _e_sublinhado(linha: str, anteriores: list[str]) -> bool:
    """ "===" ou "---" embaixo de uma linha de texto: a linha de cima é um título.

    "---" depois de linha vazia, item de lista, tabela, citação ou bloco de código é só uma
    linha separadora."""
    marca = linha.rstrip()
    if not marca or not (set(marca) <= {"="} or (len(marca) >= 3 and set(marca) <= {"-"})):
        return False
    if not anteriores or not anteriores[-1].strip():
        return False
    cima = anteriores[-1]
    if len(cima) - len(cima.lstrip(" ")) > 3:  # bloco de código recuado
        return False
    return not (ITEM_DE_LISTA.match(cima) or cima.lstrip().startswith(("|", ">", "```", "~~~", "#")))


def dividir_markdown(texto: str, fonte: str, max_palavras: int = 120) -> list[Trecho]:
    """Cria um trecho por seção `##`. Seções longas são quebradas por parágrafo.

    O texto entre o título (`#`) e a primeira seção também vira um trecho, com o título
    do documento como nome: antes ele era descartado em silêncio.

    Segue o Markdown comum (CommonMark) nos casos que apareceram nos testes:
    - "## Pizzas ##" (com # no fim) e "   ## Pizzas" (até 3 espaços antes) são títulos;
    - "## isto" dentro de um bloco de código (``` ou ~~~) não é título;
    - um título sublinhado com "===" vale como "#", e com "---" vale como "##";
    - o bloco de metadados do topo (front matter) é ignorado."""
    trechos: list[Trecho] = []
    secao_atual = Path(fonte).stem
    linhas: list[str] = []
    cerca: str | None = None  # "```" ou "~~~" de um bloco de código aberto

    def fechar() -> None:
        corpo = "\n".join(linhas).strip()
        if corpo:
            trechos.extend(_quebrar(corpo, fonte, secao_atual, max_palavras))

    for linha in _sem_metadados(texto.splitlines()):
        recuo = len(linha) - len(linha.lstrip(" "))
        limpa = linha.lstrip(" ") if recuo <= 3 else ""  # 4 espaços ou mais é bloco de código
        if cerca is not None:
            if limpa.startswith(cerca):
                cerca = None
            linhas.append(linha)
            continue
        if limpa.startswith(("```", "~~~")):
            cerca = limpa[:3]
            linhas.append(linha)
            continue
        titulo = TITULO.match(limpa) if limpa.startswith("#") else None
        sublinhado = _e_sublinhado(limpa, linhas)
        if titulo:
            # Um segundo título "#" no meio do arquivo também começa outro bloco: antes, o texto
            # da seção anterior era juntado com o do novo título e citado com o nome errado.
            fechar()
            secao_atual, linhas = (titulo.group(2) or "").strip() or Path(fonte).stem, []
        elif sublinhado:
            nome = linhas.pop().strip()
            fechar()
            secao_atual, linhas = nome, []
        else:
            linhas.append(linha)
    fechar()
    return trechos


def _quebrar(corpo: str, fonte: str, secao: str, max_palavras: int) -> list[Trecho]:
    # Cada unidade é (texto, número do parágrafo de origem). Unidades do mesmo parágrafo são
    # juntadas com quebra de linha simples; parágrafos diferentes, com linha vazia.
    unidades: list[tuple[str, int]] = []
    for n, paragrafo in enumerate(p.strip() for p in corpo.split("\n\n")):
        if len(paragrafo.split()) > max_palavras and "\n" in paragrafo:
            # Uma lista longa sem linha vazia (um cardápio colado) virava um trecho só: 519
            # palavras, e a resposta offline para o preço de uma pizza tinha 45 linhas.
            # Quebrar por linha mantém cada item inteiro.
            unidades += [(linha, n) for linha in paragrafo.splitlines() if linha.strip()]
        elif paragrafo:
            unidades.append((paragrafo, n))

    partes: list[list[tuple[str, int]]] = []
    atual: list[tuple[str, int]] = []
    palavras_atual = 0
    for texto, n in unidades:
        tamanho = len(texto.split())
        if atual and palavras_atual + tamanho > max_palavras:
            partes.append(atual)
            atual, palavras_atual = [], 0
        atual.append((texto, n))
        palavras_atual += tamanho
    if atual:
        partes.append(atual)

    def juntar(parte: list[tuple[str, int]]) -> str:
        texto = parte[0][0]
        for (_, origem_anterior), (proximo, origem) in pairwise(parte):
            texto += ("\n" if origem == origem_anterior else "\n\n") + proximo
        return texto

    return [Trecho(fonte, secao, juntar(parte)) for parte in partes]


def _ler(arquivo: Path) -> str:
    """UTF-8 (com ou sem BOM); se o arquivo foi salvo no Windows, Windows-1252."""
    bruto = arquivo.read_bytes()
    try:
        return bruto.decode("utf-8-sig")
    except UnicodeDecodeError:
        return bruto.decode("cp1252", errors="replace")


def carregar_pasta(pasta: Path = PASTA_PADRAO) -> list[Trecho]:
    pasta = Path(pasta)
    if not pasta.is_dir():
        raise ErroDocumentos(f"a pasta de documentos não existe: {pasta}")
    trechos: list[Trecho] = []
    # Arquivos ocultos ficam de fora: a trava que o editor cria enquanto o arquivo está aberto
    # (".#cardapio.md", um link quebrado) derrubava o programa inteiro, e o "._cardapio.md" que o
    # macOS deixa em pendrives e zips é lixo binário que virava trecho.
    for arquivo in sorted(a for a in pasta.glob("*.md") if not a.name.startswith(".") and a.is_file()):
        try:
            trechos.extend(dividir_markdown(_ler(arquivo), arquivo.name))
        except OSError as erro:
            raise ErroDocumentos(f"não consegui ler {arquivo.name}: {erro.strerror or erro}") from erro
    if not trechos:
        raise ErroDocumentos(f"nenhum documento .md com texto em {pasta}")
    return trechos


def carregar_sinonimos(pasta: Path = PASTA_PADRAO) -> dict[str, str]:
    """Lê o sinonimos.json opcional da pasta: {"frete": "taxa de entrega", ...}.

    O cliente escreve "frete" e o documento diz "taxa de entrega". Sem nenhuma palavra em
    comum, a busca não acha a seção certa, e nem o LLM consegue responder. Os sinônimos
    ficam junto dos documentos porque são vocabulário de cada negócio."""
    arquivo = Path(pasta) / ARQUIVO_SINONIMOS
    if not arquivo.is_file():
        return {}
    try:
        dados = json.loads(_ler(arquivo))
    except OSError as erro:
        raise ErroDocumentos(f"não consegui ler {arquivo.name}: {erro.strerror or erro}") from erro
    except ValueError as erro:
        raise ErroDocumentos(f"{arquivo.name} não é um JSON válido ({erro})") from erro
    if not isinstance(dados, dict):
        raise ErroDocumentos(f'{arquivo.name} precisa ser um objeto, ex.: {{"frete": "taxa de entrega"}}')
    sinonimos: dict[str, str] = {}
    for palavra, troca in dados.items():
        if not re.fullmatch(r"[^\W\d_]+", palavra.strip()):  # só letras: "frete", "coca"
            raise ErroDocumentos(f"{arquivo.name}: '{palavra}' precisa ser uma palavra só, sem espaço, hífen ou número")
        if not isinstance(troca, str) or not troca.strip():
            raise ErroDocumentos(f"{arquivo.name}: o sinônimo de '{palavra}' precisa ser um texto não vazio")
        sinonimos[palavra.strip()] = troca
    return sinonimos
