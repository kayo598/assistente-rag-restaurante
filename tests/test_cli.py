"""Linha de comando de ponta a ponta."""

import json
import importlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from rag_restaurante import __main__ as cli
from rag_restaurante.llm import ErroLLM

from .conftest import ClienteFixo

RAIZ = Path(__file__).resolve().parent.parent


def test_perguntar_offline(capsys):
    assert cli.main(["perguntar", "Qual a taxa de entrega para 5 km?", "--offline"]) == 0
    saida = capsys.readouterr().out
    assert "De 3 a 7 km: taxa de R$ 9,00" in saida and "fonte: entrega.md › Área e taxa de entrega" in saida


def test_perguntar_fora_do_escopo(capsys):
    assert cli.main(["perguntar", "Qual a capital da França?", "--offline"]) == 0
    assert "Não encontrei essa informação" in capsys.readouterr().out


def test_avaliar_offline(capsys):
    assert cli.main(["avaliar", "--offline"]) == 0
    saida = capsys.readouterr().out
    assert "Busca     acerto@1 100%" in saida and "Resposta  (offline) correta 100%" in saida
    assert "Perguntas com resposta: 23" in saida and "Atenção" not in saida


@pytest.fixture
def com_modelo(monkeypatch):
    """Troca a Claude API por um modelo falso que devolve um texto fixo."""

    def usar(texto):
        cliente = ClienteFixo(texto)
        cliente.uso = {"input_tokens": 900, "output_tokens": 40}
        monkeypatch.setattr(cli, "criar_cliente", lambda provedor, modelo: cliente)
        return cliente

    return usar


def test_resposta_sem_citacao_mostra_aviso(com_modelo, capsys):
    com_modelo("Fechamos às segundas.")
    assert cli.main(["perguntar", "Abre segunda?"]) == 0
    saida = capsys.readouterr().out
    assert cli.AVISO_SEM_FONTE in saida and "Tokens gastos: 900 de entrada, 40 de saída" in saida


def test_citacao_inexistente_aparece(com_modelo, capsys):
    com_modelo("Fechamos [1] [9].")
    cli.main(["perguntar", "Abre segunda?"])
    assert "citou trechos que não existem: [9]" in capsys.readouterr().out


def test_erro_da_api_vira_mensagem(com_modelo, capsys):
    com_modelo(ErroLLM("API respondeu 401 (invalid x-api-key)", fatal=True))
    assert cli.main(["perguntar", "Abre segunda?"]) == 1
    assert capsys.readouterr().err.startswith("Erro: API respondeu 401")


def test_pergunta_longa_demais(capsys):
    assert cli.main(["perguntar", "x" * 5000, "--offline"]) == 1
    assert "Não enviei: a pergunta tem 5000 caracteres" in capsys.readouterr().err


@pytest.mark.parametrize("args, mensagem", [(["--docs", "/nao/existe", "avaliar", "--offline"], "não existe")])
def test_pasta_de_documentos_invalida(capsys, args, mensagem):
    assert cli.main(args) == 1
    erro = capsys.readouterr().err
    assert erro.startswith("Erro: ") and mensagem in erro and "Traceback" not in erro


def test_gabarito_invalido(capsys, tmp_path):
    ruim = tmp_path / "g.json"
    ruim.write_text("{x", encoding="utf-8")
    assert cli.main(["avaliar", "--offline", "--gabarito", str(ruim)]) == 1
    assert "não é um JSON válido" in capsys.readouterr().err


def test_offline_com_modelo_e_recusado():
    with pytest.raises(SystemExit) as saida:
        cli.main(["perguntar", "x", "--offline", "--modelo", "claude-opus-5-5"])
    assert saida.value.code == 2


def test_sem_chave(monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert cli.main(["perguntar", "Abre segunda?"]) == 1
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


@pytest.fixture
def digitar(monkeypatch):
    def usar(*linhas):
        entradas = iter(linhas)

        def entrada(prompt=""):
            item = next(entradas)
            if isinstance(item, BaseException):
                raise item
            return item

        monkeypatch.setattr("builtins.input", entrada)

    return usar


def test_chat_ignora_enter_e_sai_com_sair(digitar, capsys):
    digitar("abre segunda?", "", "  ", "vocês vendem cerveja?", "SAIR")
    assert cli.main(["chat", "--offline"]) == 0
    saida = capsys.readouterr().out
    assert "Fechado às segundas" in saida and "Não vendemos bebidas alcoólicas" in saida and "Até logo!" in saida


@pytest.mark.parametrize("fim", [EOFError(), KeyboardInterrupt()])
def test_chat_sai_limpo_com_ctrl_d_ou_ctrl_c(digitar, capsys, fim):
    digitar("abre segunda?", fim)
    assert cli.main(["chat", "--offline"]) == 0
    assert "Até logo!" in capsys.readouterr().out


def test_chat_erro_temporario_continua_e_fatal_encerra(digitar, com_modelo, capsys):
    com_modelo(ErroLLM("API respondeu 529; desisti"))
    digitar("abre segunda?", "sair")
    assert cli.main(["chat"]) == 0
    com_modelo(ErroLLM("API respondeu 401", fatal=True))
    digitar("abre segunda?", "sair")
    assert cli.main(["chat"]) == 1


def test_terminal_do_windows(tmp_path):
    """Saída redirecionada no Windows usa cp1252, que não tem emoji: a resposta não pode quebrar."""
    (tmp_path / "promo.md").write_text("## Promoção\n\nPizza em dobro na terça 🍕", encoding="utf-8")
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "rag_restaurante",
            "--docs",
            str(tmp_path),
            "perguntar",
            "promoção de pizza",
            "--offline",
        ],
        capture_output=True,
        env={**os.environ, "PYTHONIOENCODING": "cp1252", "PYTHONPATH": str(RAIZ)},
        check=False,
    )
    assert r.returncode == 0, r.stderr
    assert b"Pizza em dobro na ter" in r.stdout


def test_chat_recusa_pergunta_longa_e_continua(digitar, capsys):
    digitar("x" * 5000, "abre segunda?", "sair")
    assert cli.main(["chat", "--offline"]) == 0
    saida = capsys.readouterr()
    assert "Não enviei: a pergunta tem 5000 caracteres" in saida.err and "Fechado às segundas" in saida.out


@pytest.mark.parametrize("comando", [["perguntar", "Abre segunda?", "--offline"], ["avaliar", "--offline"]])
def test_ctrl_c_no_meio_sai_sem_traceback(monkeypatch, capsys, comando):
    def interromper(self, pergunta):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli.Assistente, "responder", interromper)
    assert cli.main(comando) == 130
    assert "Interrompido." in capsys.readouterr().err


def test_ctrl_c_mostra_tokens_ja_gastos(com_modelo, monkeypatch, capsys):
    """Os tokens gastos até o Ctrl+C foram cobrados: o usuário precisa vê-los."""
    cliente = com_modelo("x")
    cliente.texto = KeyboardInterrupt()
    assert cli.main(["perguntar", "Abre segunda?"]) == 130
    assert "Tokens gastos: 900 de entrada" in capsys.readouterr().out


def test_ctrl_c_de_verdade(tmp_path):
    """Manda SIGINT para um processo de verdade no meio da avaliação."""
    if sys.platform == "win32":
        pytest.skip("SIGINT só no Unix")
    codigo = (
        "import sys, time, rag_restaurante.assistente as A\n"
        "A.Assistente.responder = lambda self, p: time.sleep(30)\n"
        "from rag_restaurante.__main__ import main\n"
        "sys.exit(main(['avaliar', '--offline']))\n"
    )
    ambiente = {**os.environ, "PYTHONPATH": str(RAIZ)}
    processo = subprocess.Popen(
        [sys.executable, "-c", codigo], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=ambiente
    )
    time.sleep(1.5)
    processo.send_signal(signal.SIGINT)
    _, erro = processo.communicate(timeout=20)
    assert processo.returncode == 130 and "Traceback" not in erro and "Interrompido." in erro


def test_pergunta_vazia(capsys):
    assert cli.main(["perguntar", "   ", "--offline"]) == 1
    assert "Não enviei: a pergunta está vazia." in capsys.readouterr().err


def test_chat_responde_saudacao_e_agradecimento(digitar, capsys):
    digitar("oi", "qual o valor do frete?", "obrigado!", "sair")
    assert cli.main(["chat", "--offline"]) == 0
    saida = capsys.readouterr().out
    assert "Olá!" in saida and "Não encontrei" not in saida and "De 3 a 7 km" in saida and "Por nada!" in saida


def test_avaliar_avisa_falhas_da_api(com_modelo, monkeypatch, capsys):
    cliente = com_modelo("Resposta [1].")
    original = cliente.completar
    chamadas = []

    def instavel(sistema, mensagem):
        chamadas.append(1)
        if len(chamadas) == 4:
            raise ErroLLM("API respondeu 529; desisti após 4 tentativas")
        return original(sistema, mensagem)

    cliente.completar = instavel
    assert cli.main(["avaliar"]) == 0
    saida = capsys.readouterr().out
    assert "Atenção: 1 pergunta(s) ficaram sem resposta por erro da API" in saida


def test_sinonimos_invalidos_viram_mensagem(tmp_path, capsys):
    (tmp_path / "a.md").write_text("## A\n\ntexto", encoding="utf-8")
    (tmp_path / "sinonimos.json").write_text("{x", encoding="utf-8")
    assert cli.main(["--docs", str(tmp_path), "perguntar", "texto", "--offline"]) == 1
    assert capsys.readouterr().err.startswith("Erro: sinonimos.json não é um JSON válido")


def test_chat_lembra_a_pergunta_anterior(digitar, capsys):
    digitar("tem calabresa?", "quanto é?", "sair")
    assert cli.main(["chat", "--offline"]) == 0
    saida = capsys.readouterr().out
    assert saida.count("fonte: cardapio.md › Pizzas") == 2 and "Não encontrei" not in saida


def test_chat_nao_usa_pergunta_recusada_nem_saudacao_como_contexto(digitar, monkeypatch, capsys):
    recebidas = []
    original = cli.Assistente.responder

    def espiar(self, pergunta, anterior=None):
        recebidas.append(anterior)
        return original(self, pergunta, anterior)

    monkeypatch.setattr(cli.Assistente, "responder", espiar)
    digitar("tem calabresa?", "qual a capital da França?", "oi", "quanto é?", "sair")
    assert cli.main(["chat", "--offline"]) == 0
    assert recebidas == [None, "tem calabresa?", "tem calabresa?", "tem calabresa?"]


def test_avaliar_coloquial(capsys):
    assert cli.main(["avaliar", "--offline", "--coloquial"]) == 0
    saida = capsys.readouterr().out
    assert "Perguntas com resposta: 18" in saida and "correta 83%" in saida


def test_coloquial_e_gabarito_juntos_sao_recusados():
    with pytest.raises(SystemExit):
        cli.main(["avaliar", "--coloquial", "--gabarito", "x.json"])


def test_chat_mantem_o_assunto_em_varias_perguntas_de_continuacao(digitar, capsys):
    """O segundo "e quanto é?" perdia o assunto, porque "quanto é?" virava a pergunta anterior."""
    digitar("tem calabresa?", "quanto é?", "e quanto é?", "sair")
    assert cli.main(["chat", "--offline"]) == 0
    saida = capsys.readouterr().out
    assert saida.count("fonte: cardapio.md › Pizzas") == 3 and "Não encontrei" not in saida


def test_outra_pasta_de_documentos_nao_usa_o_nome_da_cantina(tmp_path, com_modelo, digitar, capsys):
    (tmp_path / "academia.md").write_text("## Planos\n\nPlano mensal: R$ 99,00.", encoding="utf-8")
    cliente = com_modelo("R$ 99,00 [1].")
    assert cli.main(["--docs", str(tmp_path), "perguntar", "quanto custa o plano mensal?"]) == 0
    assert "Cantina" not in cliente.recebido[0][0]
    digitar("oi", "sair")
    assert cli.main(["--docs", str(tmp_path), "chat", "--offline"]) == 0
    assert "Cantina" not in capsys.readouterr().out


def test_nome_pela_linha_de_comando(tmp_path, com_modelo, digitar, capsys):
    (tmp_path / "academia.md").write_text("## Planos\n\nPlano mensal: R$ 99,00.", encoding="utf-8")
    cliente = com_modelo("R$ 99,00 [1].")
    assert cli.main(["--docs", str(tmp_path), "--nome", "Academia Forte", "perguntar", "plano mensal?"]) == 0
    assert "um negócio chamado Academia Forte" in cliente.recebido[0][0]
    digitar("oi", "sair")
    assert cli.main(["--docs", str(tmp_path), "--nome", "Academia Forte", "chat", "--offline"]) == 0
    saida = capsys.readouterr().out
    assert "Assistente de atendimento (Academia Forte)." in saida
    assert "(Academia Forte). O que você quer saber?" in saida


def test_documentos_do_projeto_usam_o_nome_da_cantina(com_modelo):
    cliente = com_modelo("ok [1]")
    cli.main(["perguntar", "Abre segunda?"])
    assert "Cantina Sabor da Vila" in cliente.recebido[0][0]


def test_gabarito_sem_perguntas_fora_do_escopo_nao_mostra_100(tmp_path, capsys):
    """Antes aparecia "Recusa correta: 100%" sem nenhuma pergunta medida."""
    (tmp_path / "a.md").write_text("## Entrega\n\nTaxa: R$ 5,00.", encoding="utf-8")
    gabarito = tmp_path / "g.json"
    caso = {"pergunta": "qual a taxa?", "fonte": "a.md", "secao": "Entrega", "deve_conter": ["5,00"]}
    gabarito.write_text(json.dumps({"com_resposta": [caso], "sem_resposta": []}), encoding="utf-8")
    assert cli.main(["--docs", str(tmp_path), "avaliar", "--offline", "--gabarito", str(gabarito)]) == 0
    saida = capsys.readouterr().out
    assert "não medida" in saida and "100%" not in saida.split("Recusa")[1]


@pytest.mark.parametrize("caractere", ["\x04", "\x1a"])
def test_ctrl_d_e_ctrl_z_do_windows_encerram_o_chat(digitar, capsys, caractere):
    """No Windows, Ctrl+D chega como "\\x04" ("^D"): virava pergunta e repetia a resposta anterior."""
    digitar("tem calabresa?", caractere, "isto não deveria ser lido")
    assert cli.main(["chat", "--offline"]) == 0
    saida = capsys.readouterr().out
    assert saida.count("fonte: cardapio.md › Pizzas") == 1 and "Até logo!" in saida


def test_tecla_de_fim_certa_para_cada_sistema(monkeypatch):
    for plataforma, tecla in (("win32", "Ctrl+Z e Enter"), ("linux", "Ctrl+D"), ("darwin", "Ctrl+D")):
        monkeypatch.setattr(sys, "platform", plataforma)
        assert tecla == importlib.reload(cli).TECLA_DE_FIM
    monkeypatch.undo()
    importlib.reload(cli)


def test_chat_mostra_a_tecla_de_fim(digitar, capsys):
    digitar("sair")
    cli.main(["chat", "--offline"])
    assert f"Digite 'sair' (ou {cli.TECLA_DE_FIM}) para terminar." in capsys.readouterr().out


@pytest.mark.parametrize(
    ("markdown", "terminal"),
    [
        ("Fechamos às **segundas** [1].", "Fechamos às segundas [1]."),
        ("## Horário\nTerça a domingo [1].", "Horário\nTerça a domingo [1]."),
        ("2 * 3 = 6", "2 * 3 = 6"),
    ],
)
def test_markdown_sai_limpo_no_terminal(markdown, terminal):
    assert cli.para_terminal(markdown) == terminal


def test_resposta_com_markdown_aparece_limpa(com_modelo, capsys):
    com_modelo("Fechamos às **segundas** [1].")
    assert cli.main(["perguntar", "Abre segunda?"]) == 0
    assert "Fechamos às segundas [1]." in capsys.readouterr().out
