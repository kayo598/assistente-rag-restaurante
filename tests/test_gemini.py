"""Testes do cliente do Gemini.

A API do Gemini não é chamada de verdade: um servidor local imita a compatibilidade OpenAI do
Gemini (/chat/completions), com o formato de resposta e de erro dele.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar

import pytest

from rag_restaurante import __main__ as cli
from rag_restaurante import llm
from rag_restaurante.assistente import Assistente
from rag_restaurante.busca import carregar_indice
from rag_restaurante.gemini import ClienteGemini, ferramenta_para_gemini, mensagens_para_gemini

INDICE = carregar_indice()


def resposta_gemini(texto="Fechamos às segundas [1].", motivo="stop", uso=(1000, 50)):
    return {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": texto}, "finish_reason": motivo}],
        "usage": {"prompt_tokens": uso[0], "completion_tokens": uso[1], "total_tokens": sum(uso)},
    }


class GeminiFalso(BaseHTTPRequestHandler):
    roteiro: ClassVar[list] = []
    recebidos: ClassVar[list] = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        corpo = json.loads(self.rfile.read(int(self.headers["content-length"])))
        self.recebidos.append({"caminho": self.path, "cabecalhos": self.headers, "corpo": corpo})
        codigo, dados = self.roteiro.pop(0) if self.roteiro else (200, resposta_gemini())
        bruto = json.dumps(dados).encode()
        self.send_response(codigo)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(bruto)))
        self.end_headers()
        self.wfile.write(bruto)


@pytest.fixture
def gemini(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    monkeypatch.setenv("GEMINI_API_KEY", "chave-do-usuario")
    GeminiFalso.roteiro, GeminiFalso.recebidos = [], []
    servidor = ThreadingHTTPServer(("127.0.0.1", 0), GeminiFalso)
    threading.Thread(target=servidor.serve_forever, daemon=True).start()
    monkeypatch.setenv("GEMINI_BASE_URL", f"http://127.0.0.1:{servidor.server_port}/v1beta/openai")
    yield GeminiFalso
    servidor.shutdown()
    servidor.server_close()


def cliente(**kw):
    return ClienteGemini(espera=lambda s: None, avisar=lambda t: None, **kw)


def test_pergunta_pela_linha_de_comando(gemini, capsys):
    assert cli.main(["perguntar", "Abre segunda?"]) == 0
    saida = capsys.readouterr().out
    assert "Fechamos às segundas [1]." in saida and "fonte: entrega.md › Horário de funcionamento" in saida
    assert "Tokens gastos: 1.000 de entrada, 50 de saída" in saida
    pedido = gemini.recebidos[0]
    assert pedido["caminho"] == "/v1beta/openai/chat/completions"
    assert pedido["cabecalhos"]["authorization"] == "Bearer chave-do-usuario"
    corpo = pedido["corpo"]
    assert corpo["model"] == "gemini-3.8-flash" and corpo["reasoning_effort"] == "low"
    assert "tools" not in corpo and "temperature" not in corpo
    sistema, usuario = corpo["messages"]
    assert sistema["role"] == "system" and "Cantina" in sistema["content"]
    assert usuario["role"] == "user" and "Abre segunda?" in usuario["content"] and "[1]" in usuario["content"]


def test_chat_mostra_qual_api_esta_usando(gemini, monkeypatch, capsys):
    entradas = iter(["abre segunda?", "sair"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(entradas))
    assert cli.main(["chat", "--modelo", "gemini-3.5-flash-lite"]) == 0
    saida = capsys.readouterr().out
    assert "Usando Google Gemini (gemini-3.5-flash-lite)." in saida and "Fechamos às segundas" in saida
    assert gemini.recebidos[0]["corpo"]["model"] == "gemini-3.5-flash-lite"


def test_avaliar_mostra_o_nome_da_api(gemini, monkeypatch, capsys):
    monkeypatch.setattr(cli, "avaliar", lambda indice, gabarito, assistente: _Metricas())
    assert cli.main(["avaliar"]) == 0
    assert "(Google Gemini (gemini-3.8-flash)) correta 50%" in capsys.readouterr().out


class _Metricas:
    total, acerto_1, acerto_3, mrr = 2, 1.0, 1.0, 1.0
    resposta_correta, cita_fonte_certa, recusa_correta, falhas_api, erros = 0.5, 0.5, None, 0, []


def test_filtro_de_conteudo_vira_recusa(gemini):
    gemini.roteiro = [(200, resposta_gemini("Sobre isso [1]", motivo="content_filter"))]
    resposta = Assistente(INDICE, cliente()).responder("Abre segunda?")
    assert resposta.recusa and resposta.texto.startswith("Não encontrei")


def test_resposta_cortada_aparece(gemini, capsys):
    gemini.roteiro = [(200, resposta_gemini("Fechamos às seg [1]", motivo="length"))]
    assert cli.main(["perguntar", "Abre segunda?"]) == 0
    assert cli.AVISO_CORTADA in capsys.readouterr().out


def test_cortada_sem_texto_e_erro_que_da_para_repetir(gemini):
    gemini.roteiro = [(200, resposta_gemini(None, motivo="length"))]
    with pytest.raises(llm.ErroLLM, match="antes de ter texto") as erro:
        Assistente(INDICE, cliente()).responder("Abre segunda?")
    assert not erro.value.fatal


def test_chave_invalida_encerra_o_chat(gemini, monkeypatch, capsys):
    corpo = [{"error": {"code": 400, "message": "API key not valid.", "status": "INVALID_ARGUMENT"}}]
    gemini.roteiro = [(400, corpo)]
    entradas = iter(["abre segunda?", "abre segunda?", "sair"])  # finito: se não parar, o teste não trava
    monkeypatch.setattr("builtins.input", lambda prompt="": next(entradas))
    assert cli.main(["chat"]) == 1
    erro = capsys.readouterr().err
    assert "INVALID_ARGUMENT: API key not valid." in erro and "GEMINI_API_KEY" in erro
    assert len(gemini.recebidos) == 1


def test_limite_por_minuto_espera_e_avisa(gemini):
    corpo = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "details": [{"retryDelay": "12.5s"}]}}
    gemini.roteiro = [(429, corpo)]
    esperas, avisos = [], []
    c = ClienteGemini(espera=esperas.append, avisar=avisos.append)
    assert "Fechamos" in Assistente(INDICE, c).responder("Abre segunda?").texto
    assert esperas == [12.5] and "esperando 12 s" in avisos[0]


def test_limite_diario_para_na_hora(gemini):
    corpo = {"error": {"code": 429, "message": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}}
    gemini.roteiro = [(429, corpo)]
    with pytest.raises(llm.ErroLLM, match="amanhã") as erro:
        Assistente(INDICE, cliente(modelo="gemini-3.8-flash")).responder("Abre segunda?")
    assert erro.value.fatal and len(gemini.recebidos) == 1


def test_mensagens_de_texto():
    assert mensagens_para_gemini("S", [{"role": "user", "content": "oi"}], {}) == [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "oi"},
    ]


@pytest.mark.parametrize("ruim", [None, {}, {"choices": []}, {"choices": [{"message": {"content": 5}}]}])
def test_resposta_malformada_vira_texto_vazio(ruim):
    c = cliente(chave="k")
    assert c.traduzir(ruim)["content"] == []


def test_escolha_automatica_do_provedor(monkeypatch):
    assert llm.escolher_provedor() == "claude"
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    assert isinstance(llm.criar_cliente(), ClienteGemini)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    assert type(llm.criar_cliente()) is llm.ClienteClaude
    assert isinstance(llm.criar_cliente("gemini"), ClienteGemini)


def test_sem_chave_fala_das_duas_opcoes(capsys):
    assert cli.main(["perguntar", "Abre segunda?"]) == 1
    erro = capsys.readouterr().err
    assert "ANTHROPIC_API_KEY" in erro and "GEMINI_API_KEY" in erro and "--offline" in erro


def test_offline_nao_aceita_provedor(capsys):
    with pytest.raises(SystemExit):
        cli.main(["perguntar", "oi", "--offline", "--provedor", "gemini"])
    assert "só valem com a API" in capsys.readouterr().err


# --- regras da tradução que valem para os três projetos (o gemini.py é o mesmo) ---


def test_traducao_cobre_os_casos_de_borda():
    vazia = ferramenta_para_gemini({"name": "agora", "description": "d", "input_schema": {"type": "object"}})
    assert "parameters" not in vazia["function"]  # o Gemini recusa objeto sem propriedades
    historico = [
        {"role": "user", "content": "oi"},
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "...", "signature": "s"}]},
        {"role": "user", "content": [{"type": "text", "text": "e agora?"}]},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "a", "name": "agora", "input": {"t": "ação"}}]},
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "a", "content": "ok"},
                {"type": "text", "text": "e depois?"},
            ],
        },
    ]
    saida = mensagens_para_gemini("S", historico, {})
    assert saida[2] == {"role": "assistant", "content": ""}  # nunca sem conteúdo e sem chamada
    assert saida[3] == {"role": "user", "content": "e agora?"}
    assert saida[4]["tool_calls"][0]["function"]["arguments"] == '{"t": "ação"}'  # sem \\u00e7
    assert saida[5] == {"role": "tool", "tool_call_id": "a", "name": "agora", "content": "ok"}
    assert saida[6] == {"role": "user", "content": "e depois?"}


def test_chamadas_de_ferramenta_na_resposta_e_de_volta_no_historico():
    c = ClienteGemini(chave="k", espera=lambda s: None, avisar=lambda t: None)
    resposta = {
        "choices": [
            {
                "finish_reason": "stop",  # o Gemini às vezes diz "stop" junto com chamadas
                "message": {
                    "tool_calls": [
                        {
                            "id": "a",
                            "function": {"name": "f", "arguments": '{"x": 1}'},
                            "extra_content": {"google": {"thought_signature": "SIG"}},
                        },
                        {"id": "", "function": {"name": "f", "arguments": "{quebrado"}},
                        {"function": {"name": "f", "arguments": "[1, 2]"}},
                    ]
                },
            }
        ]
    }
    r = c.traduzir(resposta)
    assert r["stop_reason"] == "tool_use"
    ids = [b["id"] for b in r["content"]]
    assert ids[0] == "a" and len(set(ids)) == 3  # ids gerados não se repetem
    assert [b["input"] for b in r["content"]] == [{"x": 1}, {}, {}]  # JSON inválido ou lista viram {}
    historico = [
        {"role": "user", "content": "oi"},
        {"role": "assistant", "content": r["content"]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": i, "content": "ok"} for i in ids]},
    ]
    saida = mensagens_para_gemini("S", historico, c.assinaturas)
    assert saida[2]["tool_calls"][0]["extra_content"] == {"google": {"thought_signature": "SIG"}}
    assert all("extra_content" not in t for t in saida[2]["tool_calls"][1:])


def test_erro_do_gemini_legivel_e_espera_com_teto():
    corpo = '[{"error": {"code": 503, "message": "The model is overloaded.", "status": "UNAVAILABLE"}}]'
    assert llm.mensagem_de_erro(503, corpo).startswith("API respondeu 503 (UNAVAILABLE: The model is overloaded.)")
    avisos = []
    c = ClienteGemini(chave="k", espera=lambda s: None, avisar=avisos.append)
    assert c.pausa(429, {}, '{"retryDelay": "999s"}', 1.0) == 60.0
    assert c.pausa(429, {}, "{}", 2.0) == 2.0 and len(avisos) == 1


# --- erros possíveis no uso de verdade (revisão; o gemini.py é o mesmo nos três projetos) ---


def _http(codigo: int, mensagem: str):
    import io  # noqa: PLC0415
    import urllib.error  # noqa: PLC0415

    corpo = json.dumps([{"error": {"code": codigo, "message": mensagem, "status": "X"}}]).encode()
    return urllib.error.HTTPError("u", codigo, "erro", {}, io.BytesIO(corpo))  # type: ignore[arg-type]


def _roteiro(c, itens):
    enviados = []

    def requisitar(corpo):
        enviados.append(json.loads(corpo))
        item = itens.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    c._requisitar = requisitar
    return enviados


OK = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
EXTRA = ()  # o assistente manda só o sistema e as mensagens


@pytest.mark.parametrize("colada", ["  AIza-chave\n", '"AIza-chave"', "“AIza-chave”"])
def test_chave_colada_com_sobras_e_limpa(monkeypatch, colada):
    monkeypatch.setenv("GEMINI_API_KEY", colada)
    assert ClienteGemini().chave == "AIza-chave"


def test_chave_com_acento_no_meio_da_erro_claro_sem_mostrar_a_chave(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIzá-chave")
    with pytest.raises(llm.ErroLLM) as erro:
        ClienteGemini()
    assert erro.value.fatal and "GEMINI_API_KEY" in str(erro.value) and "AIz" not in str(erro.value)


def test_google_api_key_tambem_vale(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "AIza-google")
    assert llm.escolher_provedor() == "gemini" and llm.criar_cliente().chave == "AIza-google"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "  ")  # só espaços: não conta
    assert llm.escolher_provedor() == "gemini"


def test_sem_chave_no_windows_mostra_o_comando_do_powershell(monkeypatch):
    monkeypatch.setattr(llm.sys, "platform", "win32")
    with pytest.raises(llm.ErroLLM, match=r"\$env:GEMINI_API_KEY"):
        ClienteGemini()


@pytest.mark.parametrize("motivo", ["SAFETY", "RECITATION", "PROHIBITED_CONTENT"])
def test_bloqueios_do_gemini_viram_recusa(motivo):
    c = ClienteGemini(chave="k")
    resposta = {"choices": [{"message": {"content": None}, "finish_reason": motivo}]}
    assert c.traduzir(resposta)["stop_reason"] == "refusal"


def test_inteiro_que_chega_como_decimal_vira_inteiro():
    chamada_ = {"id": "a", "function": {"name": "f", "arguments": '{"n": 4.0, "l": [1.0, 2.5], "x": NaN}'}}
    r = ClienteGemini(chave="k").traduzir({"choices": [{"message": {"tool_calls": [chamada_]}}]})
    assert r["content"][0]["input"] == {}  # NaN não é JSON válido
    chamada_["function"]["arguments"] = '{"n": 4.0, "l": [1.0, 2.5]}'
    r = ClienteGemini(chave="k").traduzir({"choices": [{"message": {"tool_calls": [chamada_]}}]})
    assert r["content"][0]["input"] == {"n": 4, "l": [1, 2.5]}
    assert isinstance(r["content"][0]["input"]["n"], int)


def test_plano_b_sem_raciocinio_e_com_assinatura_ficticia():
    c = ClienteGemini(chave="k", espera=lambda s: None, avisar=lambda t: None)
    enviados = _roteiro(c, [_http(400, "Thinking is not supported for this model."), OK, OK])
    c.enviar("s", [{"role": "user", "content": "oi"}], *EXTRA)
    assert ["reasoning_effort" in e for e in enviados] == [True, False]
    historico = [
        {"role": "user", "content": "oi"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "a", "name": "f", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "x"}]},
    ]
    _roteiro(c, [_http(400, "Function call is missing a thought_signature"), OK])
    enviados = []
    original = c._requisitar

    def guardar(corpo):
        enviados.append(json.loads(corpo))
        return original(corpo)

    c._requisitar = guardar
    c.enviar("s", historico, *EXTRA)
    assert "extra_content" not in enviados[0]["messages"][2]["tool_calls"][0]
    assert enviados[1]["messages"][2]["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == (
        "skip_thought_signature_validator"
    )


def test_outro_400_e_chave_invalida_nao_usam_plano_b():
    c = ClienteGemini(chave="k", espera=lambda s: None, avisar=lambda t: None)
    enviados = _roteiro(c, [_http(400, "API key not valid. thinking")])
    with pytest.raises(llm.ErroLLM) as erro:
        c.enviar("s", [{"role": "user", "content": "oi"}], *EXTRA)
    assert erro.value.fatal and len(enviados) == 1 and c.usar_esforco


def test_desiste_com_dica_de_sobrecarga_sem_aviso_falso():
    avisos: list[str] = []
    c = ClienteGemini(chave="k", modelo="gemini-3.8-flash", espera=lambda s: None, avisar=avisos.append)
    _roteiro(c, [_http(503, "The model is overloaded.") for _ in range(4)])
    with pytest.raises(llm.ErroLLM, match=r"desisti após 4 tentativas\. Dica: o Gemini está sobrecarregado"):
        c.enviar("s", [{"role": "user", "content": "oi"}], *EXTRA)
    assert avisos == []  # esperas de 1, 2 e 4 s: curtas, sem aviso; e nenhum aviso de espera que não houve


def test_inteiros_em_lista_viram_int_de_verdade():
    chamada_ = {"id": "a", "function": {"name": "f", "arguments": '{"n": 4.0, "l": [1.0, 2.5]}'}}
    entrada = ClienteGemini(chave="k").traduzir({"choices": [{"message": {"tool_calls": [chamada_]}}]})["content"][0][
        "input"
    ]
    assert type(entrada["n"]) is int and type(entrada["l"][0]) is int and entrada["l"][1] == 2.5


def test_mensagem_de_erro_longa_e_encurtada():
    corpo = json.dumps({"error": {"code": 429, "message": "x " * 1000, "status": "RESOURCE_EXHAUSTED"}})
    assert len(llm.mensagem_de_erro(429, corpo)) < 400


def test_tempo_de_espera_na_mensagem_e_aviso_certo_para_cada_erro():
    avisos: list[str] = []
    c = ClienteGemini(chave="k", espera=lambda s: None, avisar=avisos.append)
    assert c.pausa(429, {}, '{"message": "Please retry in 7.5s."}', 1.0) == 7.5
    assert c.pausa(503, {}, '{"retryDelay": "10s"}', 1.0) == 10.0
    assert "Limite por minuto" in avisos[0] and "respondeu 503" in avisos[1] and "Limite" not in avisos[1]


def test_os_dois_planos_b_e_depois_desiste_sem_laco_infinito():
    c = ClienteGemini(chave="k", espera=lambda s: None, avisar=lambda t: None)
    enviados = _roteiro(
        c,
        [
            _http(400, "thinking is not supported"),
            _http(400, "missing thought_signature"),
            _http(400, "thinking again"),
        ],
    )
    with pytest.raises(llm.ErroLLM, match="thinking again"):
        c.enviar("s", [{"role": "user", "content": "oi"}], *EXTRA)
    assert len(enviados) == 3


# --- terceira revisão: respostas fora do padrão e ids repetidos ---


def test_texto_em_partes_e_motivo_estranho():
    c = ClienteGemini(chave="k")
    partes = {"choices": [{"message": {"content": [{"type": "text", "text": "Olá, "}, {"text": "tudo bem?"}, 3]}}]}
    assert c.traduzir(partes)["content"] == [{"type": "text", "text": "Olá, tudo bem?"}]
    estranho = {"choices": [{"message": {"content": "oi"}, "finish_reason": ["stop"]}]}
    assert c.traduzir(estranho)["stop_reason"] == "end_turn"


def test_ids_repetidos_viram_ids_unicos():
    c = ClienteGemini(chave="k")

    def resposta(*ids):
        chamadas = [{"id": i, "function": {"name": "f", "arguments": "{}"}} for i in ids]
        return {"choices": [{"message": {"tool_calls": chamadas}}]}

    primeiro = [b["id"] for b in c.traduzir(resposta("function-call-1", "function-call-1"))["content"]]
    segundo = [b["id"] for b in c.traduzir(resposta("function-call-1", ""))["content"]]
    todos = primeiro + segundo
    assert primeiro[0] == "function-call-1" and len(set(todos)) == 4


def test_resposta_aleatoria_nunca_quebra_a_traducao():
    import random  # noqa: PLC0415

    gerador = random.Random(11)
    folhas = [None, 0, 1.5, -1, True, "", "x", "stop", "length", "SAFETY", [], {}, "{bad", '{"a": 2.0}', float("nan")]

    def valor(profundidade=0):
        if profundidade > 3 or gerador.random() < 0.4:
            return gerador.choice(folhas)
        if gerador.random() < 0.5:
            return [valor(profundidade + 1) for _ in range(gerador.randint(0, 3))]
        chaves = [
            "choices",
            "message",
            "content",
            "tool_calls",
            "function",
            "name",
            "arguments",
            "id",
            "extra_content",
            "finish_reason",
            "usage",
            "prompt_tokens",
            "refusal",
            "type",
            "text",
        ]
        return {gerador.choice(chaves): valor(profundidade + 1) for _ in range(gerador.randint(0, 4))}

    c = ClienteGemini(chave="k")
    for _ in range(3000):
        r = c.traduzir(valor())
        assert isinstance(r["content"], list)
        for bloco in r["content"]:
            assert bloco["type"] in ("text", "tool_use")
            if bloco["type"] == "tool_use":
                assert isinstance(bloco["id"], str) and bloco["id"] and isinstance(bloco["input"], dict)
                json.dumps(bloco["input"], allow_nan=False)  # volta ao Gemini como JSON válido


def test_argumentos_ja_em_objeto_com_nan_viram_vazio():
    chamada_ = {"id": "a", "function": {"name": "f", "arguments": {"x": float("nan")}}}
    r = ClienteGemini(chave="k").traduzir({"choices": [{"message": {"tool_calls": [chamada_]}}]})
    assert r["content"][0]["input"] == {}
    chamada_["function"]["arguments"] = {"n": 3.0}
    r = ClienteGemini(chave="k").traduzir({"choices": [{"message": {"tool_calls": [chamada_]}}]})
    assert r["content"][0]["input"] == {"n": 3} and type(r["content"][0]["input"]["n"]) is int


def test_provedor_aceita_maiusculas(monkeypatch):
    recebido = {}

    def fabrica(provedor, modelo):
        recebido["provedor"] = provedor
        raise llm.ErroLLM("parar aqui", fatal=True)

    monkeypatch.setattr(cli, "criar_cliente", fabrica)
    assert cli.main(["chat", "--provedor", "Gemini"]) == 1
    assert recebido == {"provedor": "gemini"}


def test_gemini_espera_mais_que_a_claude():
    """No primeiro teste real, o plano grátis do Gemini passou de 60 s para responder."""
    assert ClienteGemini(chave="k").timeout == 120
    assert llm.ClienteClaude(chave="k").timeout == 60
    assert ClienteGemini(chave="k", timeout=30).timeout == 30


def test_chat_mostra_pensando_antes_da_resposta(gemini, monkeypatch, capsys):
    entradas = iter(["abre segunda?", "sair"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(entradas))
    assert cli.main(["chat"]) == 0
    saida = capsys.readouterr().out
    assert saida.index("(pensando...)") < saida.index("Fechamos às segundas")


def test_chat_offline_nao_mostra_pensando(monkeypatch, capsys):
    entradas = iter(["abre segunda?", "sair"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(entradas))
    assert cli.main(["chat", "--offline"]) == 0
    assert "(pensando...)" not in capsys.readouterr().out


def test_sobrecarga_troca_para_o_modelo_reserva():
    """No 1º teste real, o gemini-3.8-flash ficou sobrecarregado (503) e o reserva respondeu na hora."""
    avisos: list[str] = []
    c = ClienteGemini(chave="k", espera=lambda s: None, avisar=avisos.append)
    enviados = _roteiro(c, [_http(503, "The model is overloaded.") for _ in range(4)] + [OK])
    assert c.completar("s", "abre segunda?") == "ok"
    assert [e["model"] for e in enviados] == ["gemini-3.8-flash"] * 4 + ["gemini-3.5-flash-lite"]
    assert "trocando para o gemini-3.5-flash-lite" in avisos[-1]


def test_com_modelo_escolhido_nao_troca():
    c = ClienteGemini(chave="k", modelo="gemini-3.8-flash", espera=lambda s: None, avisar=lambda t: None)
    _roteiro(c, [_http(503, "overloaded") for _ in range(4)])
    with pytest.raises(llm.ErroLLM, match="desisti"):
        c.completar("s", "oi")


def test_avaliar_nao_troca_de_modelo_no_meio(monkeypatch):
    """As métricas medem um modelo só: trocar no meio misturaria dois."""
    recebido = {}
    monkeypatch.setenv("GEMINI_API_KEY", "k")

    def avaliar_falso(indice, gabarito, assistente):
        recebido["reserva"] = assistente.cliente.reserva
        raise llm.ErroLLM("parar aqui", fatal=True)

    monkeypatch.setattr(cli, "avaliar", avaliar_falso)
    assert cli.main(["avaliar"]) == 1
    assert recebido == {"reserva": None}
    assert ClienteGemini(chave="k").reserva == "gemini-3.5-flash-lite"  # no chat e no perguntar, há reserva


def test_limite_do_dia_troca_e_o_reserva_nao_troca_para_si_mesmo():
    avisos: list[str] = []
    c = ClienteGemini(chave="k", espera=lambda s: None, avisar=avisos.append)
    diario = _http(429, "Quota exceeded: GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    enviados = _roteiro(c, [diario, OK])
    assert c.completar("s", "oi") == "ok"
    assert [e["model"] for e in enviados] == ["gemini-3.8-flash", "gemini-3.5-flash-lite"]
    enviados = _roteiro(c, [_http(503, "overloaded") for _ in range(8)])
    with pytest.raises(llm.ErroLLM, match="desisti"):
        c.completar("s", "oi")
    assert len(enviados) == 4
