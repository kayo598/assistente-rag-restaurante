"""Linha de comando.

assistente-rag perguntar "Abre segunda?" [--offline]
assistente-rag chat [--offline]
assistente-rag avaliar [--offline]
Sem --offline, usa a Claude API (ANTHROPIC_API_KEY) ou o Google Gemini, que tem plano grátis
(GEMINI_API_KEY, ou --provedor gemini).
(sem instalar: python -m rag_restaurante ...)
"""

import argparse
import re
import sys
from pathlib import Path

from .assistente import NOME_PADRAO, Assistente, Resposta
from .avaliacao import GABARITO_COLOQUIAL, GABARITO_PADRAO, ErroGabarito, avaliar
from .busca import IndiceBM25, carregar_indice
from .llm import PROVEDORES, ErroLLM, criar_cliente
from .trechos import PASTA_PADRAO, ErroDocumentos

PALAVRAS_DE_SAIDA = {"sair", "tchau", "fim", "exit", "quit"}
# No Windows, Ctrl+D não encerra a entrada: chega como o caractere "\x04" (aparece "^D"), que virava
# uma pergunta. Lá, o fim da entrada é Ctrl+Z e Enter. Os dois caracteres também encerram o chat.
TECLA_DE_FIM = "Ctrl+Z e Enter" if sys.platform == "win32" else "Ctrl+D"
CARACTERES_DE_FIM = {"\x04", "\x1a"}
AVISO_CORTADA = "(Atenção: a resposta foi cortada pelo limite de tamanho e pode estar incompleta.)"
AVISO_SEM_FONTE = "(Atenção: esta resposta não citou nenhum trecho dos documentos. Confirme com um atendente.)"


def tolerar_terminal_sem_acentos() -> None:
    """No Windows, com a saída redirecionada, o terminal usa cp1252: caracteres fora dele
    derrubariam o programa. Os que não cabem viram "?" em vez de erro."""
    for fluxo in (sys.stdout, sys.stderr):
        if hasattr(fluxo, "reconfigure"):
            fluxo.reconfigure(errors="replace")


def para_terminal(texto: str) -> str:
    """Tira as marcas de markdown que o terminal mostraria cruas: "**Total:**" vira "Total:" e
    "## Pedido" vira "Pedido". O Gemini costuma formatar assim; listas com "-" ficam como estão."""
    texto = re.sub(r"\*\*(.+?)\*\*|__(.+?)__", lambda m: m.group(1) or m.group(2), texto)
    return re.sub(r"(?m)^#{1,6}\s+", "", texto)


def imprimir(resposta: Resposta) -> None:
    print(para_terminal(resposta.texto))
    for fonte in resposta.fontes:
        print(f"  fonte: {fonte.trecho.referencia} (score {fonte.score:.2f})")
    if resposta.citacoes_invalidas:
        print(f"  (citou trechos que não existem: {resposta.citacoes_invalidas})")
    if resposta.cortada:
        print(f"  {AVISO_CORTADA}")
    if not resposta.fundamentada:
        print(f"  {AVISO_SEM_FONTE}")


def mostrar_uso(cliente) -> None:
    uso = getattr(cliente, "uso", None)
    if uso and any(uso.values()):
        entrada, saida = (f"{uso[k]:,}".replace(",", ".") for k in ("input_tokens", "output_tokens"))
        print(f"Tokens gastos: {entrada} de entrada, {saida} de saída")


def perguntar(assistente: Assistente, pergunta: str) -> int:
    try:
        imprimir(assistente.responder(pergunta))
    except ValueError as erro:  # pergunta vazia ou longa demais
        print(f"Não enviei: {erro}.", file=sys.stderr)
        return 1
    return 0


def nome_da_api(cliente) -> str:
    """'Google Gemini (gemini-3.8-flash)'. Tolerante a clientes de teste sem esses atributos."""
    return f"{getattr(cliente, 'NOME', 'API')} ({getattr(cliente, 'modelo', '?')})"


def chat(assistente: Assistente) -> int:
    quem = f" ({assistente.nome})" if assistente.nome else ""
    if assistente.cliente is not None:
        print(f"Usando {nome_da_api(assistente.cliente)}.")
    print(f"Assistente de atendimento{quem}. Digite 'sair' (ou {TECLA_DE_FIM}) para terminar.")
    anterior = None  # última pergunta respondida: dá contexto para "e quanto custa?"
    try:
        while True:
            pergunta = input("> ").strip()
            if not pergunta:  # um Enter sem querer não encerra a conversa
                continue
            if pergunta.casefold() in PALAVRAS_DE_SAIDA or pergunta in CARACTERES_DE_FIM:
                break
            if assistente.cliente is not None:  # a resposta da API pode demorar: a tela não fica parada
                print("(pensando...)", flush=True)
            try:
                resposta = assistente.responder(pergunta, anterior)
            except ValueError as erro:
                print(f"Não enviei: {erro}.", file=sys.stderr)
            except ErroLLM as erro:
                print(f"Erro: {erro}", file=sys.stderr)
                if erro.fatal:
                    return 1
            else:
                imprimir(resposta)
                # "quanto é?" respondida com ajuda da anterior não vira o novo contexto: senão,
                # um segundo "e quanto é?" perdia o assunto ("tem calabresa?") e era recusado.
                if not resposta.recusa and not resposta.conversa and not resposta.usou_contexto:
                    anterior = pergunta
    except (EOFError, KeyboardInterrupt):
        print()
    print("Até logo!")
    return 0


def cmd_avaliar(indice: IndiceBM25, assistente: Assistente, gabarito: Path, modo: str) -> int:
    m = avaliar(indice, gabarito, assistente)
    print(f"Perguntas com resposta: {m.total}")
    print(f"Busca     acerto@1 {m.acerto_1:.0%} · acerto@3 {m.acerto_3:.0%} · MRR {m.mrr:.2f}")
    print(f"Resposta  ({modo}) correta {m.resposta_correta:.0%} · cita a fonte certa {m.cita_fonte_certa:.0%}")
    if m.recusa_correta is None:  # antes mostrava 100% sem ter medido nada
        print("Recusa correta fora do escopo: não medida (o gabarito não tem perguntas fora do escopo)")
    else:
        print(f"Recusa correta fora do escopo: {m.recusa_correta:.0%}")
    if m.falhas_api:
        print(f"Atenção: {m.falhas_api} pergunta(s) ficaram sem resposta por erro da API e contaram como erro.")
    for erro in m.erros:
        print(f"  - {erro}")
    return 0


def main(argv: list[str] | None = None) -> int:
    tolerar_terminal_sem_acentos()
    p = argparse.ArgumentParser(prog="assistente-rag", description="Assistente de atendimento com RAG.")
    p.add_argument("--docs", type=Path, default=PASTA_PADRAO, help="pasta com os documentos .md")
    p.add_argument(
        "--nome",
        default=None,
        help=f"nome do negócio (padrão: '{NOME_PADRAO}' com os documentos do projeto; nenhum com outra pasta)",
    )
    sub = p.add_subparsers(dest="comando", required=True)
    s_perguntar = sub.add_parser("perguntar", help="responde uma pergunta")
    s_perguntar.add_argument("pergunta")
    s_chat = sub.add_parser("chat", help="conversa no terminal")
    s_avaliar = sub.add_parser("avaliar", help="mede busca e respostas contra o gabarito")
    gabaritos = s_avaliar.add_mutually_exclusive_group()
    gabaritos.add_argument("--gabarito", type=Path, default=GABARITO_PADRAO, help="arquivo JSON com as perguntas")
    gabaritos.add_argument(
        "--coloquial", action="store_true", help="usa as perguntas escritas como no chat (conjunto separado)"
    )
    for s in (s_perguntar, s_chat, s_avaliar):
        s.add_argument("--offline", action="store_true", help="sem LLM: responde com o trecho mais relevante")
        s.add_argument("--modelo", default=None, help="modelo da API (Claude ou Gemini)")
        s.add_argument(
            "--provedor",
            choices=PROVEDORES,
            type=str.lower,  # "Gemini" e "GEMINI" também valem
            default=None,
            help="claude ou gemini. Sem ele: claude se houver ANTHROPIC_API_KEY, senão gemini",
        )
    args = p.parse_args(argv)
    if args.offline and (args.modelo or args.provedor):
        p.error("--modelo e --provedor só valem com a API; tire o --offline ou eles")

    cliente = None
    try:
        indice = carregar_indice(args.docs)
        if not args.offline:
            cliente = criar_cliente(args.provedor, args.modelo)
            if args.comando == "avaliar" and hasattr(cliente, "reserva"):
                cliente.reserva = None  # a avaliação mede um modelo só: sem trocar no meio
        # Com outra pasta de documentos, o assistente não pode se apresentar como a Cantina.
        nome = args.nome if args.nome is not None else (NOME_PADRAO if args.docs == PASTA_PADRAO else None)
        assistente = Assistente(indice, cliente, nome=nome)
        if args.comando == "perguntar":
            return perguntar(assistente, args.pergunta)
        if args.comando == "chat":
            return chat(assistente)
        gabarito = GABARITO_COLOQUIAL if args.coloquial else args.gabarito
        return cmd_avaliar(indice, assistente, gabarito, "offline" if args.offline else nome_da_api(cliente))
    except (ErroDocumentos, ErroGabarito) as erro:
        print(f"Erro: {erro}", file=sys.stderr)
        return 1
    except ErroLLM as erro:
        print(f"Erro: {erro}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # Ctrl+C no meio de uma pergunta ou da avaliação: sem traceback
        print("\nInterrompido.", file=sys.stderr)
        return 130
    finally:
        mostrar_uso(cliente)


if __name__ == "__main__":
    sys.exit(main())
